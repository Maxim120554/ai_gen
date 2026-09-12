"""
clustering.py — кластеризация HDBSCAN, склейка близких кластеров, привязка шума.
"""
import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score


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
    """Объединяет кластеры, у которых есть объекты ближе epsilon пикселей."""
    labels = labels.copy()
    unique = sorted(set(labels.tolist()) - {-1})
    if len(unique) < 2 or epsilon <= 0:
        return labels

    nn = NearestNeighbors(radius=epsilon, metric="euclidean").fit(coords)
    _, indices = nn.radius_neighbors(coords, return_distance=True)

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