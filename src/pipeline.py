"""
pipeline.py — оркестратор генерализации по классам.
"""
import os
import pandas as pd
import geopandas as gpd

from .extraction import extract_objects, build_geodataframes_pixel
from .clustering import (get_centroids, build_features, cluster_centroids,
                         merge_close_clusters, assign_noise, evaluate_clusters)
from .aggregation import merge_clusters, merge_clusters_convex, resolve_cluster_intersections
from .export import save_geojson
from .visualization import (draw_objects, draw_centroids, draw_clusters,
                            draw_cluster_polygons, draw_hull_comparison,
                            draw_overview, draw_pixel_overlay,
                            draw_pixel_overlay_comparison, resolve_icon)


def run_class_pipeline_pixel(cls, geoms, output_dir, params, icon_path=None,
                             point_buffer=3.0, compare_hulls=True):
    """Полный цикл для одного класса в пиксельных координатах.

    Возвращает кортеж (gdf_concave, gdf_convex) или None.
    """
    os.makedirs(output_dir, exist_ok=True)

    gdf_pixel = build_geodataframes_pixel(geoms, point_buffer=point_buffer)
    draw_objects(gdf_pixel, os.path.join(output_dir, "01_objects.png"),
                 title=f"{cls}: объекты (пиксели)")

    coords = get_centroids(gdf_pixel)
    draw_centroids(gdf_pixel, coords, os.path.join(output_dir, "02_centroids.png"),
                   title=f"{cls}: центры (пиксели)")

    features = build_features(coords, gdf_pixel, use_area=params.get("use_area", False))
    labels = cluster_centroids(features, params["min_cluster_size"],
                               params["min_samples"], params["method"])
    labels = merge_close_clusters(coords, labels, params["epsilon"])
    labels = assign_noise(coords, labels, params.get("noise_max_distance", 150.0))
    score = evaluate_clusters(features, labels)
    draw_clusters(gdf_pixel, coords, labels, os.path.join(output_dir, "03_clusters.png"),
                  title=f"{cls}: кластеры (пиксели)")

    merged = merge_clusters(gdf_pixel, labels, ratio=params["concave_ratio"])
    if not merged:
        print(f"[{cls}] кластеры не получены (объектов: {len(gdf_pixel)}).")
        return None

    draw_cluster_polygons(merged, os.path.join(output_dir, "04_hulls.png"),
                          title=f"{cls}: области агрегации (пиксели)", icon_path=icon_path)

    merged_convex = merge_clusters_convex(gdf_pixel, labels)
    if compare_hulls:
        draw_cluster_polygons(merged_convex, os.path.join(output_dir, "04b_convex_hull.png"),
                              title=f"{cls}: выпуклая оболочка (пиксели)", icon_path=icon_path)
        draw_hull_comparison(merged, merged_convex,
                             os.path.join(output_dir, "04c_hull_comparison.png"),
                             icon_path=icon_path)

    gdf_out = gpd.GeoDataFrame(
        {
            "cluster_id": [m["label"] for m in merged],
            "object_count": [m["object_count"] for m in merged],
            "class": [cls] * len(merged),
        },
        geometry=[m["geometry"] for m in merged],
    )
    save_geojson(gdf_out, os.path.join(output_dir, "05_districts.geojson"))

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
    """Оркестратор: прогон конвейера по каждому найденному классу + сводные выходы."""
    base = dict(min_cluster_size=3, min_samples=3, epsilon=150.0,
                method="eom", concave_ratio=0.6, use_area=False)
    if base_params:
        base.update(base_params)

    by_class = extract_objects(input_geojson_path, classes, min_confidence=min_confidence)
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

    all_gdfs_dict = dict(all_gdfs)
    all_gdfs_dict_before = {cls: gdf.copy() for cls, gdf in all_gdfs_dict.items()}
    all_gdfs_dict = resolve_cluster_intersections(all_gdfs_dict)

    from .visualization import draw_intersections_resolution
    draw_intersections_resolution(
        all_gdfs_dict_before, all_gdfs_dict, input_geojson_path,
        os.path.join(output_dir, "00_intersections_resolution.png"),
    )
    all_gdfs = [(cls, gdf) for cls, gdf in sorted(all_gdfs_dict.items()) if not gdf.empty]
    all_convex = [(cls, gdf) for cls, gdf in all_convex if cls in dict(all_gdfs)]

    if not all_gdfs:
        return {}

    combined = gpd.GeoDataFrame(pd.concat([g for _, g in all_gdfs], ignore_index=True))
    save_geojson(combined, os.path.join(output_dir, "00_all_districts.geojson"))

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