"""
pipeline_v2.py
Улучшенный конвейер кластеризации зданий в районы.

Улучшения по сравнению с v1:
  1. Проекция координат в метры (UTM) для точных расстояний.
  2. cluster_selection_epsilon — склейка близких кластеров.
  3. Concave hull (вогнутая оболочка) вместо buffer для точных границ.
  4. Привязка шума к ближайшему кластеру (здания не теряются).
  5. cluster_selection_method='leaf' для стабильных кластеров.
  6. Опциональный признак — площадь здания.
  7. MultiPolygon сохраняется как одно здание.
  8. Метрика качества — silhouette score.
  9. Интерактивная folium-карта.

Зависимости:
    pip install shapely scikit-learn matplotlib numpy geopandas folium
"""

import os
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from sklearn.cluster import HDBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
from shapely.validation import make_valid
import folium

# concave_hull доступна в Shapely >= 2.0
try:
    from shapely import concave_hull
    HAS_CONCAVE = True
except ImportError:
    concave_hull = None
    HAS_CONCAVE = False


# ============================================================
# ШАГ 1. Извлечение зданий (MultiPolygon — одно здание)
# ============================================================

def extract_buildings(geojson_path):
    """Читает GeoJSON и возвращает список геометрий зданий в WGS84."""
    with open(geojson_path, encoding="utf-8") as f:
        geojson = json.load(f)

    geoms = []
    for feature in geojson.get("features", []):
        props = feature.get("properties") or {}
        building = props.get("building")

        if not building or str(building).lower() in {"no", "false", "0", "none"}:
            continue

        geometry = feature.get("geometry") or {}
        if geometry.get("type") not in {"Polygon", "MultiPolygon"}:
            continue

        try:
            geom = shape(geometry)
        except Exception:
            continue

        if not geom.is_valid:
            geom = geom.buffer(0)
            if geom.is_empty or not geom.is_valid:
                continue

        if geom.geom_type not in {"Polygon", "MultiPolygon"}:
            continue

        geoms.append(geom)

    return geoms


def build_geodataframes(geoms):
    """Создаёт GeoDataFrame в WGS84 и его проекцию в UTM (метры)."""
    gdf_wgs = gpd.GeoDataFrame(geometry=geoms, crs="EPSG:4326")
    utm_crs = gdf_wgs.estimate_utm_crs()
    gdf_proj = gdf_wgs.to_crs(utm_crs)
    return gdf_wgs, gdf_proj


def draw_buildings(gdf_proj, output_path):
    fig, ax = plt.subplots(figsize=(10, 10))
    _plot_polygons(ax, gdf_proj.geometry, color="#cfe8ff", edgecolor="#1f77b4", linewidth=0.6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Полигоны зданий")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# ШАГ 2. Центры зданий (в метрах)
# ============================================================

def get_centroids(gdf_proj):
    """Возвращает массив координат центров зданий в метрах."""
    points = gdf_proj.geometry.representative_point()
    return np.array([[p.x, p.y] for p in points])


def build_features(coords, gdf_proj, use_area=False):
    """Матрица признаков для кластеризации. Опционально добавляет площадь."""
    if not use_area:
        return coords
    areas = gdf_proj.geometry.area.values
    areas_norm = (areas - areas.mean()) / (areas.std() + 1e-12)
    return np.column_stack([coords, areas_norm])


def draw_centroids(gdf_proj, coords, output_path):
    fig, ax = plt.subplots(figsize=(10, 10))
    _plot_polygons(ax, gdf_proj.geometry, color="#cfe8ff", edgecolor="#1f77b4", linewidth=0.6)
    ax.scatter(coords[:, 0], coords[:, 1], s=10, color="#d62728", zorder=5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Центры полигонов")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# ШАГ 3. Кластеризация HDBSCAN + привязка шума + метрика
# ============================================================
def merge_close_clusters(coords, labels, epsilon):
    """
    Объединяет кластеры, у которых есть здания ближе epsilon метров.
    Замена cluster_selection_epsilon (обход бага sklearn).
    """
    labels = labels.copy()
    unique = sorted(set(labels.tolist()) - {-1})
    if len(unique) < 2 or epsilon <= 0:
        return labels

    # все пары точек ближе epsilon
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

    # перенумеровываем связные компоненты
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


def cluster_centroids(features, min_cluster_size=5, min_samples=3, method="leaf"):
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


def assign_noise(coords, labels, max_distance=150.0):
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
    """Silhouette score (шум исключён)."""
    mask = labels != -1
    if mask.sum() < 2 or len(set(labels[mask].tolist())) < 2:
        return None
    try:
        return silhouette_score(X[mask], labels[mask])
    except Exception:
        return None


def draw_clusters(gdf_proj, coords, labels, output_path):
    fig, ax = plt.subplots(figsize=(10, 10))
    _plot_polygons(ax, gdf_proj.geometry, color="#f0f0f0", edgecolor="#dddddd", linewidth=0.3)

    cmap = plt.get_cmap("tab20")
    cluster_labels = sorted(set(labels.tolist()) - {-1})
    noise_mask = labels == -1

    if noise_mask.any():
        ax.scatter(coords[noise_mask, 0], coords[noise_mask, 1],
                   s=8, color="#bbbbbb", alpha=0.5, zorder=4)

    for i, label in enumerate(cluster_labels):
        mask = labels == label
        ax.scatter(coords[mask, 0], coords[mask, 1], s=15, color=cmap(i % 20), zorder=5)

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Полученные кластеры")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# ШАГ 4. Объединение кластеров (concave hull)
# ============================================================

def merge_clusters(gdf_proj, labels, ratio=0.3):
    """Объединяет здания каждого кластера в один полигон (вогнутая оболочка)."""
    gdf = gdf_proj.copy()
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
            "building_count": int(len(group)),
        })

    return merged


def draw_cluster_polygons(merged, output_path, title="Внешняя граница кластеров"):
    fig, ax = plt.subplots(figsize=(10, 10))
    cmap = plt.get_cmap("tab20")
    for i, item in enumerate(merged):
        _plot_polygons(ax, [item["geometry"]], color=cmap(i % 20), alpha=0.7,
                       edgecolor="black", linewidth=0.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title)          # <-- было: ax.set_title(f"Внешняя граница кластеров")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# ШАГ 4b. Выпуклая оболочка (для сравнения с вогнутой)
# ============================================================

def merge_clusters_convex(gdf_proj, labels):
    """Объединяет здания каждого кластера в один полигон (выпуклая оболочка).

    Аналог merge_clusters, но вместо concave_hull используется convex_hull.
    Параметр ratio здесь не нужен — выпуклая оболочка строится однозначно.
    """
    gdf = gdf_proj.copy()
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
            "building_count": int(len(group)),
        })

    return merged


def draw_hull_comparison(merged_concave, merged_convex, output_path):
    """Рядом два изображения: вогнутая оболочка (слева) и выпуклая (справа).

    Оба списка строятся из одних и тех же меток через groupby, поэтому
    порядок кластеров (и цвета) совпадает — удобно сравнивать визуально.
    """
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
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# ШАГ 5. Конвертация в WGS84 и сохранение GeoJSON
# ============================================================

def to_wgs84(merged, proj_crs):
    """Конвертирует районы из проекции обратно в WGS84."""
    gdf = gpd.GeoDataFrame(
        {
            "cluster_id": [m["label"] for m in merged],
            "building_count": [m["building_count"] for m in merged],
        },
        geometry=[m["geometry"] for m in merged],
        crs=proj_crs,
    )
    return gdf.to_crs("EPSG:4326")


def save_output_geojson(gdf_wgs, output_path):
    """Сохраняет итоговые районы как GeoJSON."""
    features = []
    for _, row in gdf_wgs.iterrows():
        features.append({
            "type": "Feature",
            "properties": {
                "cluster_id": int(row["cluster_id"]),
                "building_count": int(row["building_count"]),
                "type": "district",
            },
            "geometry": mapping(row.geometry),
        })

    geojson = {"type": "FeatureCollection", "features": features}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False, indent=2)


# ============================================================
# ШАГ 6. Интерактивная folium-карта
# ============================================================

PALETTE = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe",
    "#008080", "#e6beff", "#9a6324", "#fffac8", "#800000",
    "#aaffc3", "#808000", "#ffd8b1", "#000075", "#808080",
]


def save_folium_map(gdf_wgs, output_path):
    """Сохраняет интерактивную карту с районами."""
    try:
        union = gdf_wgs.geometry.union_all()
    except AttributeError:
        union = gdf_wgs.geometry.unary_union

    center = union.centroid
    m = folium.Map(location=[center.y, center.x], zoom_start=14, tiles="CartoDB positron")

    def style(feature):
        cid = feature["properties"]["cluster_id"]
        color = PALETTE[cid % len(PALETTE)]
        return {"fillColor": color, "color": color, "weight": 1.5, "fillOpacity": 0.45}

    folium.GeoJson(
        gdf_wgs,
        style_function=style,
        tooltip=folium.GeoJsonTooltip(fields=["cluster_id", "building_count"]),
    ).add_to(m)

    m.save(output_path)


# ============================================================
# Вспомогательная отрисовка полигонов
# ============================================================

def _plot_polygons(ax, geoms, **kwargs):
    for g in geoms:
        if g.geom_type == "Polygon":
            x, y = g.exterior.xy
            ax.fill(x, y, **kwargs)
        elif g.geom_type == "MultiPolygon":
            for part in g.geoms:
                x, y = part.exterior.xy
                ax.fill(x, y, **kwargs)


# ============================================================
# ГЛАВНЫЙ КОНВЕЙЕР
# ============================================================

def run_pipeline(
    input_geojson_path,
    output_dir="results_v2",
    min_cluster_size=5,
    min_samples=3,
    epsilon=50.0,
    method="leaf",
    concave_ratio=0.3,
    use_area=False,
):
    os.makedirs(output_dir, exist_ok=True)

    # 0. Копия исходника
    shutil.copy(input_geojson_path, os.path.join(output_dir, "00_input.geojson"))
    print(f"[0] Исходник скопирован в {output_dir}/00_input.geojson")

    # 1. Извлечение зданий + проекция в метры
    geoms = extract_buildings(input_geojson_path)
    if not geoms:
        print("Здания не найдены.")
        return None
    gdf_wgs, gdf_proj = build_geodataframes(geoms)
    print(f"[1] Зданий: {len(gdf_proj)}, CRS: {gdf_proj.crs}")
    draw_buildings(gdf_proj, os.path.join(output_dir, "01_buildings_only.png"))

    # 2. Центры зданий
    coords = get_centroids(gdf_proj)
    draw_centroids(gdf_proj, coords, os.path.join(output_dir, "02_centroids.png"))
    print(f"[2] Центров: {len(coords)}")

    # 3. Признаки и кластеризация
    features = build_features(coords, gdf_proj, use_area=use_area)
    labels = cluster_centroids(features, min_cluster_size, min_samples, method)
    n_clusters = len(set(labels.tolist()) - {-1})
    n_noise = int((labels == -1).sum())
    print(f"[3] Кластеров: {n_clusters}, шум: {n_noise}")
    draw_clusters(gdf_proj, coords, labels, os.path.join(output_dir, "03_clusters.png"))

    # 3b. Склейка близких кластеров (замена cluster_selection_epsilon)
    labels = merge_close_clusters(coords, labels, epsilon)
    n_after_merge = len(set(labels.tolist()) - {-1})
    print(f"[3b] Кластеров после склейки (epsilon={epsilon} м): {n_after_merge}")

    # 3c. Привязка шума
    labels = assign_noise(coords, labels)
    print(f"[3c] Кластеров после привязки шума: {len(set(labels.tolist()))}")

    # 3d. Метрика качества
    score = evaluate_clusters(features, labels)
    if score is not None:
        print(f"[3d] Silhouette: {score:.3f}")

    # 4. Объединение в районы
    merged = merge_clusters(gdf_proj, labels, ratio=concave_ratio)
    if not merged:
        print("[4] Районы не получены (все точки — шум или слишком мало зданий).")
        return None
    print(f"[4] Районов: {len(merged)}")
    draw_cluster_polygons(merged, os.path.join(output_dir, "04_cluster_polygons.png"))

    # 4b. Выпуклая оболочка — для сравнения с вогнутой
    merged_convex = merge_clusters_convex(gdf_proj, labels)
    draw_cluster_polygons(
        merged_convex,
        os.path.join(output_dir, "04b_convex_hull.png"),
        title="Выпуклая оболочка кластеров",
    )
    draw_hull_comparison(
        merged,
        merged_convex,
        os.path.join(output_dir, "04c_hull_comparison.png"),
    )

    print(f"[4b] Выпуклая оболочка построена для {len(merged_convex)} кластеров")
    # 5. Конвертация в WGS84 и сохранение GeoJSON
    gdf_districts_wgs = to_wgs84(merged, gdf_proj.crs)
    output_geojson = os.path.join(output_dir, "05_output.geojson")
    save_output_geojson(gdf_districts_wgs, output_geojson)
    print(f"[5] Сохранено: {output_geojson}")

    # 6. Интерактивная карта
    map_path = os.path.join(output_dir, "06_map.html")
    save_folium_map(gdf_districts_wgs, map_path)
    print(f"[6] Сохранено: {map_path}")

    print(f"Готово! Результаты в папке '{output_dir}/'")
    return merged


# ============================================================
# ЗАПУСК
# ============================================================

if __name__ == "__main__":
    run_pipeline(
        input_geojson_path="input_data_geojson/test6.geojson",
        output_dir="results/results6",
        min_cluster_size=10,  # крупнее ядра — меньше случайных кластеров
        min_samples=6,  # главное против chaining
        epsilon=25.0,  # меньше склейка далёких кластеров
        method="eom",  # попробуй вместо leaf — более плотностные кластеры
        concave_ratio=0.15,  # tighter, оболочка плотнее облегает здания
        use_area=False,
    )