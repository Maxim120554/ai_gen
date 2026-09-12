"""
ai_gen/src — пакет модулей конвейера генерализации.
"""
from .detection import detect_and_save_geojson
from .extraction import extract_objects, build_geodataframes_pixel
from .clustering import cluster_centroids, merge_close_clusters, assign_noise, evaluate_clusters
from .aggregation import merge_clusters, merge_clusters_convex
from .export import save_geojson
from .pipeline import run_pipeline_multi_pixel
from full_pipeline_v2 import run_full_pipeline, run

__all__ = [
    "detect_and_save_geojson",
    "extract_objects",
    "build_geodataframes_pixel",
    "cluster_centroids",
    "merge_close_clusters",
    "assign_noise",
    "evaluate_clusters",
    "merge_clusters",
    "merge_clusters_convex",
    "save_geojson",
    "run_pipeline_multi_pixel",
    "run_full_pipeline",
    "run",
]