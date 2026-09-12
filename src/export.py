"""
export.py — сохранение районов в GeoJSON.
"""
import json
import numpy as np
from shapely.geometry import mapping

DISTRICT_FILL = "#ffcc99"
DISTRICT_STROKE = "#cc8833"


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