"""
detection.py — детекция объектов (YOLO OBB + SAHI) и сохранение в GeoJSON.
"""
import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    rasterio = None
    HAS_RASTERIO = False


def read_image(image_path, use_georeference=False):
    """Читает снимок (PNG — OpenCV, TIF — rasterio).
    Возвращает (image HWC uint8, transform, crs)."""
    image_path = Path(image_path)
    transform, crs = None, None

    if image_path.suffix.lower() in (".tif", ".tiff") and HAS_RASTERIO:
        with rasterio.open(image_path) as src:
            if use_georeference:
                transform = src.transform
                crs = str(src.crs) if src.crs else None
            image = src.read()
            if image.shape[0] <= 4:
                image = np.moveaxis(image, 0, -1)
            image = image[:, :, :3] if image.shape[-1] >= 3 \
                else np.stack([image[:, :, 0]] * 3, axis=-1)
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 \
                else image.astype(np.uint8)
    else:
        image = cv2.imread(str(image_path))

    return image, transform, crs


def pixel_to_geo(transform, x_px, y_px):
    """Пиксельные координаты -> географические (аффинная трансформация)."""
    return (transform[0] * x_px + transform[2],
            transform[4] * y_px + transform[5])


def obb_to_geojson_polygon(coords_px, transform=None):
    """OBB (8 значений) -> замкнутый контур GeoJSON Polygon."""
    coords = []
    for i in range(0, 8, 2):
        x, y = coords_px[i], coords_px[i + 1]
        if transform is not None:
            x, y = pixel_to_geo(transform, x, y)
        coords.append([float(x), float(y)])
    coords.append(coords[0])
    return [coords]


def detect_and_save_geojson(image_path, model_path, output_geojson,
                            slice_size=640, overlap_ratio=0.2,
                            confidence_threshold=0.35, device="cuda:0",
                            use_georeference=False, class_names=None):
    """Детекция со слайсингом SAHI и сохранение результата в GeoJSON."""
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"❌ Файл не найден: {image_path}")
        return None

    image, transform, crs = read_image(image_path, use_georeference)
    if image is None:
        print(f"❌ Не удалось загрузить изображение: {image_path}")
        return None
    if transform is not None:
        print(f"📍 Геопривязка: {crs}, трансформация: {transform}")
    print(f"📐 Размер изображения: {image.shape[1]}x{image.shape[0]}")

    print(f"🔄 Загрузка модели: {model_path}")
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=model_path,
        confidence_threshold=confidence_threshold,
        device=device,
    )

    print(f"🔍 SAHI-детекция (тайлы {slice_size}x{slice_size})...")
    result = get_sliced_prediction(
        image,
        detection_model,
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=overlap_ratio,
        overlap_width_ratio=overlap_ratio,
        postprocess_type="GREEDYNMM",
        postprocess_match_threshold=0.5,
        verbose=1,
    )
    print(f"✅ Найдено объектов: {len(result.object_prediction_list)}")

    if class_names is None:
        try:
            from ultralytics import YOLO
            class_names = list(YOLO(model_path).names.values())
        except Exception:
            class_names = [f"class_{i}" for i in range(60)]

    features = []
    for obj in result.object_prediction_list:
        x_min, y_min, x_max, y_max = obj.bbox.to_xyxy()
        obb_coords = [x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]

        cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
        if transform is not None:
            cx, cy = pixel_to_geo(transform, cx, cy)

        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": obb_to_geojson_polygon(obb_coords, transform),
            },
            "properties": {
                "class_id": int(obj.category.id),
                "class_name": obj.category.name,
                "confidence": round(float(obj.score.value), 4),
                "center_x": round(float(cx), 6),
                "center_y": round(float(cy), 6),
                "width_px": round(float(x_max - x_min), 2),
                "height_px": round(float(y_max - y_min), 2),
                "detected_at": datetime.now().isoformat(),
            },
        })

    geojson_data = {
        "type": "FeatureCollection",
        "crs": {"type": "name",
                "properties": {"name": crs} if crs else {"name": "pixel_coordinates"}},
        "metadata": {
            "source_image": str(image_path.absolute()),
            "model": model_path,
            "detection_method": "YOLO + SAHI",
            "slice_size": slice_size,
            "overlap_ratio": overlap_ratio,
            "confidence_threshold": confidence_threshold,
            "total_objects": len(features),
            "created_at": datetime.now().isoformat(),
        },
        "features": features,
    }

    output_path = Path(output_geojson)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(geojson_data, f, ensure_ascii=False, indent=2)

    print(f"✅ GeoJSON сохранён: {output_path.absolute()} ({len(features)} объектов)")
    counts = {}
    for ft in features:
        name = ft["properties"]["class_name"]
        counts[name] = counts.get(name, 0) + 1
    for cls, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"   {cls}: {n}")
    return output_path