"""
full_pipeline_v2.py — единый конвейер: снимок -> детекция -> генерализация.

Точка входа. Импортирует модули из пакета src.
"""
import os
from pathlib import Path
import yaml

from src.detection import detect_and_save_geojson
from src.pipeline import run_pipeline_multi_pixel


def run_full_pipeline(image_path, model_path, output_dir="results_full",
                      slice_size=640, overlap_ratio=0.2, confidence_threshold=0.35,
                      device="cuda:0", use_georeference=False,
                      detect=True, detections_geojson=None,
                      classes=None, class_params=None, base_params=None,
                      icon_dir="icons", min_confidence=0.35, zoom=0.05):
    """Полный цикл обработки."""
    os.makedirs(output_dir, exist_ok=True)
    geojson_path = detections_geojson or os.path.join(output_dir, "00_detections.geojson")

    if detect:
        detect_and_save_geojson(
            image_path=image_path, model_path=model_path,
            output_geojson=geojson_path,
            slice_size=slice_size, overlap_ratio=overlap_ratio,
            confidence_threshold=confidence_threshold, device=device,
            use_georeference=use_georeference,
        )

    from src.visualization import draw_detections_overlay
    draw_detections_overlay(geojson_path,
                            os.path.join(output_dir, "00_detections_overlay.png"),
                            min_confidence=min_confidence)

    if not os.path.exists(geojson_path):
        print("❌ GeoJSON детекции отсутствует — конвейер остановлен.")
        return {}

    if use_georeference:
        print("️  Получены географические координаты: пиксельная версия генерализации\n"
              "   ожидает пиксельные координаты.")

    return run_pipeline_multi_pixel(
        input_geojson_path=geojson_path,
        output_dir=output_dir,
        classes=classes,
        class_params=class_params,
        icon_dir=icon_dir,
        base_params=base_params,
        min_confidence=min_confidence,
        zoom=zoom,
    )


def load_config(config_path="config.yaml"):
    """Загружает YAML-конфиг и нормализует пути относительно base_dir."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base = Path(cfg["paths"]["base_dir"]).resolve()
    p = cfg["paths"]
    p["base_dir"] = base
    p["image"] = base / p["image"]
    p["model"] = base / p["model"]
    p["output_dir"] = base / p["output_dir"]
    p["icon_dir"] = base / p["icon_dir"]
    p["detections_geojson"] = p["output_dir"] / p["detections_geojson"]

    return cfg


def run(config="config.yaml"):
    cfg = load_config(config)
    p, d, g, o = cfg["paths"], cfg["detection"], cfg["generalization"], cfg["output"]

    class_params = dict(g.get("class_params", {}))

    return run_full_pipeline(
        image_path=str(p["image"]),
        model_path=str(p["model"]),
        output_dir=str(p["output_dir"]),
        slice_size=d["slice_size"],
        overlap_ratio=d["overlap_ratio"],
        confidence_threshold=d["confidence_threshold"],
        device=d["device"],
        use_georeference=d["use_georeference"],
        icon_dir=str(p["icon_dir"]),
        zoom=o["zoom"],
        min_confidence=g["min_confidence"],
        base_params={
            "min_cluster_size": g["min_cluster_size"],
            "min_samples": g["min_samples"],
            "epsilon": g["epsilon"],
            "concave_ratio": g["concave_ratio"],
            "method": g["method"],
            "use_area": g["use_area"],
        },
        class_params=class_params,
    )


if __name__ == "__main__":
    run(config="config.yaml")