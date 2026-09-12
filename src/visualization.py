"""
visualization.py — отрисовка объектов, кластеров, оболочек и оверлеев.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# Палитра для раскраски классов
PALETTE = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe",
    "#008080", "#e6beff", "#9a6324", "#fffac8", "#800000",
    "#aaffc3", "#808000", "#ffd8b1", "#000075", "#808080",
]


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
    ax.invert_yaxis()
    plt.tight_layout(); plt.savefig(output_path, dpi=200, bbox_inches="tight"); plt.close(fig)


def draw_centroids(gdf, coords, output_path, title="Центры полигонов"):
    fig, ax = plt.subplots(figsize=(10, 10))
    _plot_polygons(ax, gdf.geometry, color="#cfe8ff", edgecolor="#1f77b4", linewidth=0.6)
    ax.scatter(coords[:, 0], coords[:, 1], s=10, color="#d62728", zorder=5)
    ax.set_aspect("equal"); ax.axis("off"); ax.set_title(title)
    ax.invert_yaxis()
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
    ax.invert_yaxis()
    plt.tight_layout(); plt.savefig(output_path, dpi=200, bbox_inches="tight"); plt.close(fig)


def add_center_icons(ax, merged, icon_path, zoom=0.03):
    """Размещает представительный знак в центре каждой области."""
    if not icon_path or not os.path.exists(icon_path):
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
    if not icon_path or not os.path.exists(icon_path):
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
    ax.invert_yaxis()
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
        ax.invert_yaxis()

    plt.tight_layout(); plt.savefig(output_path, dpi=200, bbox_inches="tight"); plt.close(fig)


def resolve_icon(icon_dir, cls):
    """Ищет иконку класса: icons/<cls>.png, иначе icons/default.png, иначе None."""
    if not icon_dir:
        return None
    for name in (f"{cls}.png", "default.png"):
        path = os.path.join(icon_dir, name)
        if os.path.exists(path):
            return path
    return None


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
    """Наложение агрегированных областей на исходное изображение."""
    with open(geojson_path, encoding="utf-8") as f:
        image_path = json.load(f).get("metadata", {}).get("source_image")
    if not image_path or not os.path.exists(image_path):
        print(f"[overlay] Изображение {image_path} не найдено — пропуск.")
        return

    from matplotlib.patches import Patch

    img = mpimg.imread(image_path)
    h, w = img.shape[:2]
    k = w / 2000.0
    dpi = 100
    fig, ax = plt.subplots(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax.imshow(img, alpha=0.5)

    for i, (cls, gdf) in enumerate(all_gdfs):
        color = PALETTE[i % len(PALETTE)]
        for geom in gdf.geometry:
            parts = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
            for p in parts:
                x, y = p.exterior.xy
                ax.fill(x, y, color=color, alpha=0.3,
                        edgecolor=color, linewidth=2 * k)

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
    plt.savefig(output_path, dpi=dpi, pad_inches=0)
    plt.close(fig)
    print(f"[overlay] Сохранено {output_path} ({w}x{h})")


def draw_pixel_overlay_comparison(all_gdfs, all_convex, geojson_path, output_path,
                                  icon_dir=None, zoom=0.03):
    """Исходное изображение + вогнутые оболочки (заливка) и выпуклые (пунктир)."""
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
        for geom in gdf.geometry:
            parts = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
            for p in parts:
                x, y = p.exterior.xy
                ax.fill(x, y, color=color, alpha=0.3,
                        edgecolor=color, linewidth=2 * k)
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


def draw_detections_overlay(geojson_path, output_path, min_confidence=0.35):
    """Исходное изображение + цветные рамки обнаруженных объектов.

    Только контур рамки (без заливки и подписей).
    Легенда с цветами классов — в правом верхнем углу.
    Сохраняется в родном разрешении растра.
    """
    with open(geojson_path, encoding="utf-8") as f:
        geojson = json.load(f)

    image_path = geojson.get("metadata", {}).get("source_image")
    if not image_path or not os.path.exists(image_path):
        print(f"[detections_overlay] Изображение {image_path} не найдено — пропуск.")
        return

    img = mpimg.imread(image_path)
    h, w = img.shape[:2]
    k = w / 2000.0
    dpi = 100

    fig, ax = plt.subplots(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax.imshow(img)

    # Словарь цветов по классам
    class_colors = {}
    color_idx = 0

    for feature in geojson.get("features", []):
        props = feature.get("properties", {})
        confidence = props.get("confidence", 1.0)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 1.0
        if confidence < min_confidence:
            continue

        cls = props.get("class_name", "unknown")
        if cls not in class_colors:
            class_colors[cls] = PALETTE[color_idx % len(PALETTE)]
            color_idx += 1

        color = class_colors[cls]
        coords = feature["geometry"]["coordinates"][0]

        # Извлекаем bounding box из координат рамки
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        width = x_max - x_min
        height = y_max - y_min

        # Рисуем только контур рамки (без заливки и подписей)
        from matplotlib.patches import Rectangle
        rect = Rectangle(
            (x_min, y_min), width, height,
            linewidth=1.2 * k,
            edgecolor=color,
            facecolor='none',  # прозрачная заливка
            alpha=0.85,
            zorder=5
        )
        ax.add_patch(rect)

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Обнаруженные объекты", fontsize=int(12 * k), pad=10)

    # Легенда с классами в правом верхнем углу
    from matplotlib.patches import Patch
    handles = [Patch(color=c, label=cls) for cls, c in class_colors.items()]
    ax.legend(handles=handles, loc="upper right", fontsize=int(9 * k),
              framealpha=0.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, pad_inches=0)
    plt.close(fig)
    print(f"[detections_overlay] Сохранено {output_path} ({w}x{h})")


def draw_intersections_resolution(all_gdfs_before, all_gdfs_after, geojson_path, output_path):
    """Сравнение: до и после разрешения пересечений (два планшета рядом)."""
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    for ax, all_gdfs, title in zip(
        axes,
        (all_gdfs_before, all_gdfs_after),
        ("До разрешения пересечений", "После разрешения пересечений"),
    ):
        with open(geojson_path, encoding="utf-8") as f:
            image_path = json.load(f).get("metadata", {}).get("source_image")
        if image_path and os.path.exists(image_path):
            img = mpimg.imread(image_path)
            h, w = img.shape[:2]
            ax.imshow(img, alpha=0.5)
            k = w / 2000.0
        else:
            k = 1.0

        for i, (cls, gdf) in enumerate(all_gdfs.items()):
            if gdf.empty:
                continue
            color = PALETTE[i % len(PALETTE)]
            for geom in gdf.geometry:
                parts = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
                for p in parts:
                    x, y = p.exterior.xy
                    ax.fill(x, y, color=color, alpha=0.3,
                            edgecolor=color, linewidth=1.5 * k)

        ax.set_aspect("equal"); ax.axis("off"); ax.set_title(title)
        handles = [Patch(color=PALETTE[i % len(PALETTE)], label=cls)
                   for i, cls in enumerate(all_gdfs) if not all_gdfs[cls].empty]
        ax.legend(handles=handles, loc="upper right", fontsize=int(9 * k), framealpha=0.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, pad_inches=0)
    plt.close(fig)
    print(f"[intersections] Сохранено {output_path}")