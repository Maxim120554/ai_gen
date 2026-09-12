"""
extraction.py — извлечение объектов из GeoJSON по классам и вероятности.
"""
import json
import numpy as np
import geopandas as gpd
from shapely.geometry import shape

# Возможные имена поля класса в properties
CLASS_FIELDS = ["class_name", "class", "type", "label", "category", "name"]

# Набор обрабатываемых классов по умолчанию
PROCESSED_CLASSES = [
    "small-vehicle",
    "large-vehicle",
    "ship",
    "plane",
    "helicopter",
]


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

    Объекты с confidence < min_confidence отбрасываются.
    """
    classes = set(classes or PROCESSED_CLASSES)
    with open(geojson_path, encoding="utf-8") as f:
        geojson = json.load(f)

    by_class = {}
    skipped = 0

    for feature in geojson.get("features", []):
        props = feature.get("properties") or {}

        confidence = props.get("confidence", 1.0)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 1.0
        if confidence < min_confidence:
            skipped += 1
            continue

        cls = get_feature_class(props, classes)
        if cls is None:
            continue

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
    if point_buffer:
        gdf_pixel["geometry"] = gdf_pixel.geometry.apply(
            lambda g: g.buffer(point_buffer) if g.geom_type == "Point" else g)
    return gdf_pixel