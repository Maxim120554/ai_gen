"""
pipeline_v4.py
Многоклассовый конвейер генерализации в ПИКСЕЛЬНЫХ координатах:
кластеризация объектов, обнаруженных детектором (YOLO + SAHI),
в агрегированные области по классам.

Обрабатываемые классы (текущий набор):
  1. small-vehicle — легковые автомобили;
  2. large-vehicle — грузовые автомобили;
  3. ship          — корабли;
  4. plane         — самолёты;
  5. helicopter    — вертолёты.

Этапы для каждого класса:
  1. Извлечение объектов из GeoJSON по атрибуту класса.
  2. Вычисление центров объектов (пиксельные координаты).
  3. Кластеризация HDBSCAN, склейка близких кластеров, привязка шума.
  4. Построение агрегированных контуров (вогнутая оболочка; для сравнения —
     выпуклая), представительный знак в центре области.
  5. Экспорт районов в GeoJSON (по классам и сводный файл).
  6. Обзорный планшет и наложение на исходное изображение.

Особенность: координаты пиксельные (ось Y направлена ВНИЗ), поэтому во всех
функциях отрисовки без imshow ось Y инвертируется вызовом ax.invert_yaxis().

Зависимости:
    pip install shapely scikit-learn matplotlib numpy geopandas pandas
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import geopandas as gpd
from sklearn.cluster import HDBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
from shapely.validation import make_valid

try:
    from shapely import concave_hull
    HAS_CONCAVE = True
except ImportError:
    concave_hull = None
    HAS_CONCAVE = False

# ============================================================
# КОНСТАНТЫ
# ============================================================

# Текущий набор обрабатываемых классов
PROCESSED_CLASSES = [
    "small-vehicle",  # легковые автомобили
    "large-vehicle",  # грузовые автомобили
    "ship",           # корабли
    "plane",          # самолёты
    "helicopter",     # вертолёты
]

# Полный список классов DOTA (для расширения набора):
# "plane", "ship", "storage-tank", "baseball-diamond", "tennis-court",
# "basketball-court", "ground-track-field", "harbor", "bridge",
# "large-vehicle", "small-vehicle", "helicopter", "roundabout",
# "soccer-ball-field", "swimming-pool", "container-crane"

# Возможные имена поля класса в properties
CLASS_FIELDS = ["class_name", "class", "type", "label", "category", "name"]

# Цвет районов по умолчанию (светло-оранжевый)
DISTRICT_FILL = "#ffcc99"
DISTRICT_STROKE = "#cc8833"

# Палитра для раскраски классов на сводных выходах
PALETTE = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe",
    "#008080", "#e6beff", "#9a6324", "#fffac8", "#800000",
    "#aaffc3", "#808000", "#ffd8b1", "#000075", "#808080",
]


# ============================================================
# ШАГ 1. Извлечение объектов по классам
# ============================================================

def get_feature_class(props, classes):
    """Возвращает нормализованное имя класса объекта или None."""
    for field in CLASS_FIELDS:
        val = props.get(field)
        if val is None:
            continue
        val = str(val).strip().lower().replace("_", "-")
        if val in classes:
            return val
    return None


def extract_objects(geojson_path, classes=None, min_confidence=0.35):
    """Читает GeoJSON и возвращает словарь {класс: [геометрии]}.

    Объекты, у которых вероятность детекции (properties.confidence)
    ниже min_confidence, отбрасываются как недостоверные.
    """
    classes = set(classes or PROCESSED_CLASSES)
    with open(geojson_path, encoding="utf-8") as f:
        geojson = json.load(f)

    by_class = {}
    skipped = 0

    for feature in geojson.get("features", []):
        props = feature.get("properties") or {}

        # --- фильтр по вероятности детекции ---
        confidence = props.get("confidence", 1.0)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 1.0
        if confidence < min_confidence:
            skipped += 1
            continue

        # --- фильтр по классу ---
        cls = get_feature_class(props, classes)
        if cls is None:
            continue

        # --- фильтр по типу и корректности геометрии ---
        geometry = feature.get("geometry") or {}
        if geometry.get("type") not in {"Polygon", "MultiPolygon", "Point"}:
            continue
        try:
            geom = shape(geometry)
        except Exception:
            continue
        if not geom.is_valid:
            geom = geom.buffer(0)
            if geom.is_empty or not geom.is_valid:
                continue

        by_class.setdefault(cls, []).append(geom)

    if skipped:
        print(f"[extract] Отброшено объектов с confidence < {min_confidence}: {skipped}")
    return by_class


def build_geodataframes_pixel(geoms, point_buffer=3.0):
    """GeoDataFrame в пиксельных координатах (без геопривязки)."""
    gdf_pixel = gpd.GeoDataFrame(geometry=list(geoms), crs=None)

    # Точечные детекции буферизуем в маленькие полигоны
    if point_buffer:
        gdf_pixel["geometry"] = gdf_pixel.geometry.apply(
            lambda g: g.buffer(point_buffer) if g.geom_type == "Point" else g)

    return gdf_pixel


# ============================================================
# ШАГ 2. Центры объектов и признаки
# ============================================================

def get_centroids(gdf):
    """Возвращает массив координат центров объектов (пиксели)."""
    points = gdf.geometry.representative_point()
    return np.array([[p.x, p.y] for p in points])


def build_features(coords, gdf, use_area=False):
    """Матрица признаков для кластеризации. Опционально добавляет площадь."""
    if not use_area:
        return coords
    areas = gdf.geometry.area.values
    areas_norm = (areas - areas.mean()) / (areas.std() + 1e-12)
    return np.column_stack([coords, areas_norm])


# ============================================================
# ШАГ 3. Кластеризация HDBSCAN + склейка + привязка шума
# ============================================================

def cluster_centroids(features, min_cluster_size=5, min_samples=3, method="leaf"):
    """Кластеризация центров объектов алгоритмом HDBSCAN."""
    if len(features) < min_cluster_size:
        return np.array([-1] * len(features))

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=0.0,
        cluster_selection_method=method,
        metric="euclidean",
    )
    return clusterer.fit_predict(features)


def merge_close_clusters(coords, labels, epsilon):
    """Объединяет кластеры, у которых есть объекты ближе epsilon пикселей
    (замена cluster_selection_epsilon, обход особенности sklearn)."""
    labels = labels.copy()
    unique = sorted(set(labels.tolist()) - {-1})
    if len(unique) < 2 or epsilon <= 0:
        return labels

    nn = NearestNeighbors(radius=epsilon, metric="euclidean").fit(coords)
    _, indices = nn.radius_neighbors(coords, return_distance=True)

    # Union-Find над метками кластеров
    parent = {l: l for l in unique}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(len(coords)):
        li = labels[i]
        if li == -1:
            continue
        for j in indices[i]:
            lj = labels[j]
            if lj == -1 or lj == li:
                continue
            union(li, lj)

    root_to_new = {}
    new_labels = labels.copy()
    counter = 0
    for i in range(len(labels)):
        l = labels[i]
        if l == -1:
            continue
        r = find(l)
        if r not in root_to_new:
            root_to_new[r] = counter
            counter += 1
        new_labels[i] = root_to_new[r]

    return new_labels


def assign_noise(coords, labels, max_distance=150):
    """Привязывает шум к ближайшему кластеру, но не дальше max_distance."""
    labels = labels.copy()
    noise_idx = np.where(labels == -1)[0]
    if len(noise_idx) == 0:
        return labels

    cluster_ids = np.array(sorted(set(labels.tolist()) - {-1}))
    if len(cluster_ids) == 0:
        return labels

    cluster_centers = np.array([coords[labels == k].mean(axis=0) for k in cluster_ids])
    nn = NearestNeighbors(n_neighbors=1).fit(cluster_centers)
    dist, idx = nn.kneighbors(coords[noise_idx])

    for d, i, pos in zip(dist.flatten(), idx.flatten(), noise_idx):
        if d <= max_distance:
            labels[pos] = cluster_ids[i]
    # дальний шум остаётся -1 и не попадает в районы
    return labels


def evaluate_clusters(X, labels):
    """Коэффициент силуэта (шум исключён)."""
    mask = labels != -1
    if mask.sum() < 2 or len(set(labels[mask].tolist())) < 2:
        return None
    try:
        return silhouette_score(X[mask], labels[mask])
    except Exception:
        return None


# ============================================================
# ШАГ 4. Объединение кластеров в агрегированные контуры
# ============================================================

def merge_clusters(gdf, labels, ratio=0.6):
    """Объединяет объекты каждого кластера в один полигон (вогнутая оболочка)."""
    gdf = gdf.copy()
    gdf["cluster"] = labels

    merged = []
    for label, group in gdf.groupby("cluster"):
        if label == -1:
            continue

        union = unary_union(group.geometry.values)

        if HAS_CONCAVE:
            hull = concave_hull(union, ratio=ratio)
        else:
            hull = union.buffer(20).buffer(-20)  # fallback для Shapely < 2.0

        if hull.is_empty or not hull.is_valid:
            hull = make_valid(hull)

        merged.append({
            "label": int(label),
            "geometry": hull,
            "object_count": int(len(group)),
        })

    return merged


def merge_clusters_convex(gdf, labels):
    """То же, но выпуклая оболочка (для сравнения с вогнутой)."""
    gdf = gdf.copy()
    gdf["cluster"] = labels

    merged = []
    for label, group in gdf.groupby("cluster"):
        if label == -1:
            continue

        union = unary_union(group.geometry.values)
        hull = union.convex_hull

        if hull.is_empty or not hull.is_valid:
            hull = make_valid(hull)

        merged.append({
            "label": int(label),
            "geometry": hull,
            "object_count": int(len(group)),
        })

    return merged


# ============================================================
# ШАГ 5. Экспорт результатов
# ============================================================

def save_geojson(gdf, output_path, fill=DISTRICT_FILL, stroke=DISTRICT_STROKE):
    """Сохраняет районы; все атрибутивные колонки попадают в properties."""
    features = []
    for _, row in gdf.iterrows():
        props = {"type": "district"}
        for col in gdf.columns:
            if col == "geometry":
                continue
            v = row[col]
            if isinstance(v, (int, np.integer)):
                v = int(v)
            elif isinstance(v, (float, np.floating)):
                v = float(v)
            props[col] = v
        props.update({"fill": fill, "fill-opacity": 0.5,
                      "stroke": stroke, "stroke-width": 1.5, "stroke-opacity": 0.9})
        features.append({"type": "Feature", "properties": props,
                         "geometry": mapping(row.geometry)})
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": features},
                  f, ensure_ascii=False, indent=2)


# ============================================================
# ШАГ 6. Отрисовка (ось Y инвертирована во всех функциях без imshow)
# ============================================================

def _plot_polygons(ax, geoms, **kwargs):
    """Вспомогательная отрисовка полигонов и мультиполигонов."""
    for g in geoms:
        if g.geom_type == "Polygon":
            x, y = g.exterior.xy
            ax.fill(x, y, **kwargs)
        elif g.geom_type == "MultiPolygon":
            for part in g.geoms:
                x, y = part.exterior.xy
                ax.fill(x, y, **kwargs)


def draw_objects(gdf, output_path, title="Объекты"):
    fig, ax = plt.subplots(figsize=(10, 10))
    _plot_polygons(ax, gdf.geometry, color="#cfe8ff", edgecolor="#1f77b4", linewidth=0.6)
    ax.set_aspect("equal"); ax.axis("off"); ax.set_title(title)
    ax.invert_yaxis()  # пиксельные координаты: ось Y направлена вниз
    plt.tight_layout(); plt.savefig(output_path, dpi=200, bbox_inches="tight"); plt.close(fig)


def draw_centroids(gdf, coords, output_path, title="Центры полигонов"):
    fig, ax = plt.subplots(figsize=(10, 10))
    _plot_polygons(ax, gdf.geometry, color="#cfe8ff", edgecolor="#1f77b4", linewidth=0.6)
    ax.scatter(coords[:, 0], coords[:, 1], s=10, color="#d62728", zorder=5)
    ax.set_aspect("equal"); ax.axis("off"); ax.set_title(title)
    ax.invert_yaxis()  # пиксельные координаты: ось Y направлена вниз
    plt.tight_layout(); plt.savefig(output_path, dpi=200, bbox_inches="tight"); plt.close(fig)


def draw_clusters(gdf, coords, labels, output_path, title="Полученные кластеры"):
    fig, ax = plt.subplots(figsize=(10, 10))
    _plot_polygons(ax, gdf.geometry, color="#f0f0f0", edgecolor="#dddddd", linewidth=0.3)

    cmap = plt.get_cmap("tab20")
    cluster_labels = sorted(set(labels.tolist()) - {-1})
    noise_mask = labels == -1

    if noise_mask.any():
        ax.scatter(coords[noise_mask, 0], coords[noise_mask, 1],
                   s=8, color="#bbbbbb", alpha=0.5, zorder=4)

    for i, label in enumerate(cluster_labels):
        mask = labels == label
        ax.scatter(coords[mask, 0], coords[mask, 1], s=15, color=cmap(i % 20), zorder=5)

    ax.set_aspect("equal"); ax.axis("off"); ax.set_title(title)
    ax.invert_yaxis()  # пиксельные координаты: ось Y направлена вниз
    plt.tight_layout(); plt.savefig(output_path, dpi=200, bbox_inches="tight"); plt.close(fig)


def add_center_icons(ax, merged, icon_path, zoom=0.03):
    """Размещает представительный знак в центре каждой области."""
    if not icon_path or not os.path.exists(icon_path):  # <-- то же самое
        print(f"[icons] Файл {icon_path} не найден — знаки пропущены.")
        return
    img = mpimg.imread(icon_path)
    for item in merged:
        p = item["geometry"].representative_point()
        ab = AnnotationBbox(OffsetImage(img, zoom=zoom), (p.x, p.y),
                            frameon=False, zorder=7)
        ax.add_artist(ab)

def add_icons_for_gdf(ax, gdf, icon_path, zoom=0.03):
    """Знак класса в центре каждой области GeoDataFrame."""
    if not icon_path or not os.path.exists(icon_path):  # <-- добавили проверку None
        return
    img = mpimg.imread(icon_path)
    for geom in gdf.geometry:
        p = geom.representative_point()
        ab = AnnotationBbox(OffsetImage(img, zoom=zoom), (p.x, p.y),
                            frameon=False, zorder=8)
        ax.add_artist(ab)


def draw_cluster_polygons(merged, output_path, title="Внешняя граница кластеров",
                          icon_path=None, icon_zoom=0.03):
    fig, ax = plt.subplots(figsize=(10, 10))
    cmap = plt.get_cmap("tab20")
    for i, item in enumerate(merged):
        _plot_polygons(ax, [item["geometry"]], color=cmap(i % 20), alpha=0.7,
                       edgecolor="black", linewidth=0.5)

    if icon_path:
        add_center_icons(ax, merged, icon_path, zoom=icon_zoom)

    ax.set_aspect("equal"); ax.axis("off"); ax.set_title(title)
    ax.invert_yaxis()  # пиксельные координаты: ось Y направлена вниз
    plt.tight_layout(); plt.savefig(output_path, dpi=200, bbox_inches="tight"); plt.close(fig)


def draw_hull_comparison(merged_concave, merged_convex, output_path,
                         icon_path=None, icon_zoom=0.03):
    """Рядом два изображения: вогнутая оболочка (слева) и выпуклая (справа)."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    cmap = plt.get_cmap("tab20")

    for ax, merged, title in zip(
        axes,
        (merged_concave, merged_convex),
        ("Вогнутая оболочка (concave hull)", "Выпуклая оболочка (convex hull)"),
    ):
        for i, item in enumerate(merged):
            _plot_polygons(ax, [item["geometry"]], color=cmap(i % 20), alpha=0.7,
                           edgecolor="black", linewidth=0.5)

        if icon_path:
            add_center_icons(ax, merged, icon_path, zoom=icon_zoom)

        ax.set_aspect("equal"); ax.axis("off"); ax.set_title(title)
        ax.invert_yaxis()  # пиксельные координаты: ось Y направлена вниз

    plt.tight_layout(); plt.savefig(output_path, dpi=200, bbox_inches="tight"); plt.close(fig)


def draw_overview(all_gdfs, output_path, icon_dir=None, zoom=0.03):
    """Все районы всех классов на одном планшете с легендой и знаками."""
    from matplotlib.patches import Patch
    fig, ax = plt.subplots(figsize=(12, 12))
    for i, (cls, gdf) in enumerate(all_gdfs):
        _plot_polygons(ax, gdf.geometry, color=PALETTE[i % len(PALETTE)],
                       alpha=0.6, edgecolor="black", linewidth=0.4)
        if icon_dir:
            add_icons_for_gdf(ax, gdf, resolve_icon(icon_dir, cls), zoom=zoom)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("Области агрегации по классам объектов")
    handles = [Patch(color=PALETTE[i % len(PALETTE)], label=cls)
               for i, (cls, _) in enumerate(all_gdfs)]
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    ax.invert_yaxis()
    plt.tight_layout(); plt.savefig(output_path, dpi=200, bbox_inches="tight"); plt.close(fig)


def draw_pixel_overlay(all_gdfs, geojson_path, output_path, icon_dir=None, zoom=0.03):
    """Наложение агрегированных областей (вогнутые оболочки) на исходное изображение.
    Сохраняется в родном разрешении растра; толщина линий, знаки и шрифты
    масштабируются коэффициентом k = width / 2000."""
    with open(geojson_path, encoding="utf-8") as f:
        image_path = json.load(f).get("metadata", {}).get("source_image")
    if not image_path or not os.path.exists(image_path):
        print(f"[overlay] Изображение {image_path} не найдено — пропуск.")
        return

    from matplotlib.patches import Patch

    img = mpimg.imread(image_path)
    h, w = img.shape[:2]
    k = w / 2000.0                 # масштаб оформления под крупный холст
    dpi = 100                      # 1 дюйм фигуры = 100 px изображения
    fig, ax = plt.subplots(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax.imshow(img, alpha=0.5)      # origin='upper': ось Y уже направлена вниз

    # Заливка агрегированных областей по классам
    for i, (cls, gdf) in enumerate(all_gdfs):
        color = PALETTE[i % len(PALETTE)]
        for geom in gdf.geometry:
            parts = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
            for p in parts:
                x, y = p.exterior.xy
                ax.fill(x, y, color=color, alpha=0.3,
                        edgecolor=color, linewidth=2 * k)

    # Представительные знаки классов в центрах областей
    if icon_dir:
        for cls, gdf in all_gdfs:
            add_icons_for_gdf(ax, gdf, resolve_icon(icon_dir, cls),
                              zoom=zoom * k)

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Агрегированные области на исходном изображении",
                 fontsize=int(12 * k))
    handles = [Patch(color=PALETTE[i % len(PALETTE)], label=cls)
               for i, (cls, _) in enumerate(all_gdfs)]
    ax.legend(handles=handles, loc="upper right", fontsize=int(10 * k))

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, pad_inches=0)   # без bbox_inches="tight"
    plt.close(fig)
    print(f"[overlay] Сохранено {output_path} ({w}x{h})")


def draw_pixel_overlay_comparison(all_gdfs, all_convex, geojson_path, output_path,
                                  icon_dir=None, zoom=0.03):
    """Исходное изображение + вогнутые оболочки (заливка по классам)
    и выпуклые оболочки (чёрный пунктир) для сравнения.
    Сохраняется в родном разрешении растра."""
    with open(geojson_path, encoding="utf-8") as f:
        image_path = json.load(f).get("metadata", {}).get("source_image")
    if not image_path or not os.path.exists(image_path):
        print(f"[overlay] Изображение {image_path} не найдено — пропуск.")
        return

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    img = mpimg.imread(image_path)
    h, w = img.shape[:2]
    k = w / 2000.0
    dpi = 100
    fig, ax = plt.subplots(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax.imshow(img, alpha=0.5)

    for i, ((cls, gdf), (_, gdf_cvx)) in enumerate(zip(all_gdfs, all_convex)):
        color = PALETTE[i % len(PALETTE)]
        # вогнутая оболочка — заливка
        for geom in gdf.geometry:
            parts = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
            for p in parts:
                x, y = p.exterior.xy
                ax.fill(x, y, color=color, alpha=0.3,
                        edgecolor=color, linewidth=2 * k)
        # выпуклая оболочка — пунктирный контур
        for geom in gdf_cvx.geometry:
            parts = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
            for p in parts:
                x, y = p.exterior.xy
                ax.plot(x, y, color="black", linestyle="--", linewidth=1.5 * k)

    if icon_dir:
        for cls, gdf in all_gdfs:
            add_icons_for_gdf(ax, gdf, resolve_icon(icon_dir, cls),
                              zoom=zoom * k)

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Сравнение оболочек на исходном изображении",
                 fontsize=int(12 * k))
    handles = [Patch(color=PALETTE[i % len(PALETTE)], label=cls)
               for i, (cls, _) in enumerate(all_gdfs)]
    handles.append(Line2D([0], [0], color="black", linestyle="--",
                          linewidth=1.5, label="выпуклая оболочка"))
    ax.legend(handles=handles, loc="upper right", fontsize=int(10 * k))

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, pad_inches=0)
    plt.close(fig)
    print(f"[overlay] Сохранено {output_path} ({w}x{h})")


# ============================================================
# ГЛАВНЫЙ КОНВЕЙЕР
# ============================================================

def resolve_icon(icon_dir, cls):
    """Ищет иконку класса: icons/<cls>.png, иначе icons/default.png, иначе None."""
    if not icon_dir:
        return None
    for name in (f"{cls}.png", "default.png"):
        path = os.path.join(icon_dir, name)
        if os.path.exists(path):
            return path
    return None


def run_class_pipeline_pixel(cls, geoms, output_dir, params, icon_path=None,
                             point_buffer=3.0, compare_hulls=True):
    """Полный цикл для одного класса в пиксельных координатах.

    Возвращает кортеж (gdf_concave, gdf_convex) — GeoDataFrame районов,
    построенных вогнутой и выпуклой оболочками, либо None, если кластеров нет.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Данные класса в пиксельных координатах
    gdf_pixel = build_geodataframes_pixel(geoms, point_buffer=point_buffer)
    draw_objects(gdf_pixel, os.path.join(output_dir, "01_objects.png"),
                 title=f"{cls}: объекты (пиксели)")

    # 2. Центры объектов и признаки
    coords = get_centroids(gdf_pixel)
    draw_centroids(gdf_pixel, coords, os.path.join(output_dir, "02_centroids.png"),
                   title=f"{cls}: центры (пиксели)")

    # 3. Кластеризация: HDBSCAN -> склейка близких -> привязка шума
    features = build_features(coords, gdf_pixel, use_area=params.get("use_area", False))
    labels = cluster_centroids(features, params["min_cluster_size"],
                               params["min_samples"], params["method"])
    labels = merge_close_clusters(coords, labels, params["epsilon"])
    labels = assign_noise(coords, labels, params.get("noise_max_distance", 150.0))
    score = evaluate_clusters(features, labels)
    draw_clusters(gdf_pixel, coords, labels, os.path.join(output_dir, "03_clusters.png"),
                  title=f"{cls}: кластеры (пиксели)")

    # 4. Агрегированные контуры: вогнутая оболочка (основная)
    merged = merge_clusters(gdf_pixel, labels, ratio=params["concave_ratio"])
    if not merged:
        print(f"[{cls}] кластеры не получены (объектов: {len(gdf_pixel)}).")
        return None

    draw_cluster_polygons(merged, os.path.join(output_dir, "04_hulls.png"),
                          title=f"{cls}: области агрегации (пиксели)", icon_path=icon_path)

    # 4b. Выпуклая оболочка — считается всегда (нужна для сравнения на оверлее)
    merged_convex = merge_clusters_convex(gdf_pixel, labels)
    if compare_hulls:
        draw_cluster_polygons(merged_convex, os.path.join(output_dir, "04b_convex_hull.png"),
                              title=f"{cls}: выпуклая оболочка (пиксели)", icon_path=icon_path)
        draw_hull_comparison(merged, merged_convex,
                             os.path.join(output_dir, "04c_hull_comparison.png"),
                             icon_path=icon_path)

    # 5. Экспорт районов (вогнутая оболочка) в GeoJSON
    gdf_out = gpd.GeoDataFrame(
        {
            "cluster_id": [m["label"] for m in merged],
            "object_count": [m["object_count"] for m in merged],
            "class": [cls] * len(merged),
        },
        geometry=[m["geometry"] for m in merged],
    )
    save_geojson(gdf_out, os.path.join(output_dir, "05_districts.geojson"))

    # GeoDataFrame выпуклых оболочек — для сводного сравнения
    gdf_convex = gpd.GeoDataFrame(
        {
            "cluster_id": [m["label"] for m in merged_convex],
            "object_count": [m["object_count"] for m in merged_convex],
            "class": [cls] * len(merged_convex),
        },
        geometry=[m["geometry"] for m in merged_convex],
    )

    print(f"[{cls}] объектов: {len(gdf_pixel)}, районов: {len(merged)}, "
          f"silhouette: {score if score is None else round(score, 3)}")
    return gdf_out, gdf_convex


def run_pipeline_multi_pixel(input_geojson_path, output_dir="results_pixel",
                             classes=None, class_params=None, icon_dir="icons",
                             base_params=None, min_confidence=0.35, zoom=0.05):
    """Оркестратор: прогон конвейера по каждому найденному классу + сводные выходы.

    Для каждого класса создаётся папка output_dir/<класс>/ с покадровыми
    рисунками и GeoJSON; в корне output_dir формируются сводные файлы:
    00_all_districts.geojson, 00_overview.png, 00_overlay_on_image.png,
    00_overlay_convex_comparison.png.
    """
    # Базовые параметры в пикселях; class_params переопределяет их по классам
    base = dict(min_cluster_size=3, min_samples=3, epsilon=150.0,
                method="eom", concave_ratio=0.6, use_area=False)
    if base_params:
        base.update(base_params)

    by_class = extract_objects(input_geojson_path, classes, min_confidence=0.35)
    if not by_class:
        print("Объекты из списка классов не найдены.")
        return {}
    print(f"Найдено классов: {len(by_class)} -> {sorted(by_class)}")

    all_gdfs, all_convex = [], []
    for cls, geoms in sorted(by_class.items()):
        params = dict(base)
        if class_params and cls in class_params:
            params.update(class_params[cls])
        res = run_class_pipeline_pixel(
            cls, geoms, os.path.join(output_dir, cls), params,
            icon_path=resolve_icon(icon_dir, cls),
        )
        if res is not None:
            gdf_out, gdf_convex = res
            all_gdfs.append((cls, gdf_out))
            all_convex.append((cls, gdf_convex))

    if not all_gdfs:
        return {}

    # Сводный GeoJSON всех районов (вогнутые оболочки)
    combined = gpd.GeoDataFrame(pd.concat([g for _, g in all_gdfs], ignore_index=True))
    save_geojson(combined, os.path.join(output_dir, "00_all_districts.geojson"))

    # Сводные визуализации: планшет, оверлей на снимке, сравнение оболочек
    draw_overview(all_gdfs, os.path.join(output_dir, "00_overview.png"),
                  icon_dir=icon_dir, zoom=zoom)
    draw_pixel_overlay(all_gdfs, input_geojson_path,
                       os.path.join(output_dir, "00_overlay_on_image.png"),
                       icon_dir=icon_dir, zoom=zoom)
    draw_pixel_overlay_comparison(all_gdfs, all_convex, input_geojson_path,
                                  os.path.join(output_dir, "00_overlay_convex_comparison.png"),
                                  icon_dir=icon_dir, zoom=zoom)

    print(f"Готово. Результаты в '{output_dir}/'")
    return dict(all_gdfs)
# ============================================================
# ЗАПУСК
# ============================================================

if __name__ == "__main__":
    run_pipeline_multi_pixel(
        input_geojson_path="input_data_geojson/detections8.geojson",
        output_dir="results/results_pixel_8",
        # обрабатываются только: small-vehicle, large-vehicle, ship, plane, helicopter
        base_params={
            "min_cluster_size": 3,
            "epsilon": 150.0,       # пикселей
            "concave_ratio": 0.6,
        },
        class_params={
            "small-vehicle": {"min_cluster_size": 7, "epsilon": 70.0},
            "large-vehicle": {"min_cluster_size": 5, "epsilon": 70.0},
            "ship":          {"min_cluster_size": 5, "epsilon": 70.0},
            "plane":         {"min_cluster_size": 3, "epsilon": 70.0},
            "helicopter":    {"min_cluster_size": 3, "epsilon": 70.0},
        },
    )