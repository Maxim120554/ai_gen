"""
full_pipeline_v2.py — единый конвейер: снимок -> детекция -> генерализация.

Точка входа. Импортирует модули из пакета src.

Использование:
    python full_pipeline_v2.py --config config.yaml
    python full_pipeline_v2.py --config config.yaml --image /data/img.png --weights /data/best.pt --output-dir /results
"""
import os
import argparse
from pathlib import Path

import yaml

from src.detection import detect_and_save_geojson
from src.pipeline import run_pipeline_multi_pixel


# ============================================================
# Конфиг
# ============================================================

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


# ============================================================
# Основной конвейер
# ============================================================

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
        print("⚠️  Получены географические координаты: пиксельная версия генерализации\n"
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


# ============================================================
# Точка входа: конфиг + опциональные переопределения из CLI
# ============================================================

def run(config_path="config.yaml", image=None, weights=None, output_dir=None):
    """Запуск конвейера по конфигу с возможностью переопределить пути.

    Args:
        config_path: путь к config.yaml.
        image:       переопределить путь к изображению (из CLI или None).
        weights:     переопределить путь к весам модели (из CLI или None).
        output_dir:  переопределить выходную папку (из CLI или None).
    """
    cfg = load_config(config_path)
    p, d, g, o = cfg["paths"], cfg["detection"], cfg["generalization"], cfg["output"]

    # Применяем переопределения из CLI (если переданы)
    if image is not None:
        p["image"] = Path(image)
    if weights is not None:
        p["model"] = Path(weights)
    if output_dir is not None:
        p["output_dir"] = Path(output_dir)

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


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Конвейер генерализации карт")
    parser.add_argument("--config", required=True, help="Путь к config.yaml")
    parser.add_argument("--image", default=None,
                        help="Переопределить путь к изображению из конфига")
    parser.add_argument("--weights", default=None,
                        help="Переопределить путь к весам YOLO из конфига")
    parser.add_argument("--output-dir", default=None,
                        help="Переопределить output_dir из конфига")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        config_path=args.config,
        image=args.image,
        weights=args.weights,
        output_dir=args.output_dir,
    )