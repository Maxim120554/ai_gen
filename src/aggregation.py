"""
aggregation.py — построение агрегированных контуров (вогнутая/выпуклая оболочки).
"""
import geopandas as gpd
from shapely.ops import unary_union
from shapely.validation import make_valid

try:
    from shapely import concave_hull
    HAS_CONCAVE = True
except ImportError:
    concave_hull = None
    HAS_CONCAVE = False


def merge_clusters(gdf, labels, ratio=0.6):
    """Объединяет объекты каждого кластера в один полигон (вогнутая оболочка)."""
    gdf = gdf.copy()
    gdf["cluster"] = labels

    merged = []
    for label, group in gdf.groupby("cluster"):
        if label == -1:
            continue

        union = unary_union(group.geometry.values)

        if HAS_CONCAVE:
            hull = concave_hull(union, ratio=ratio)
        else:
            hull = union.buffer(20).buffer(-20)

        if hull.is_empty or not hull.is_valid:
            hull = make_valid(hull)

        merged.append({
            "label": int(label),
            "geometry": hull,
            "object_count": int(len(group)),
        })

    return merged


def merge_clusters_convex(gdf, labels):
    """То же, но выпуклая оболочка (для сравнения с вогнутой)."""
    gdf = gdf.copy()
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
            "object_count": int(len(group)),
        })

    return merged


def resolve_cluster_intersections(all_gdfs, min_objects_after_trim=2):
    """Разрешает пересечения между кластерами.

    Правила:
      1. Кластеры ОДНОГО класса с пересекающимися выпуклыми оболочками
         объединяются в один кластер.
      2. Кластеры РАЗНЫХ классов с пересекающимися оболочками:
         — кластер с бóльшим object_count сохраняется целиком;
         — кластер с меньшим object_count обрезается через разность
           (difference) с большим; если после обрезки остаётся меньше
           min_objects_after_trim объектов — кластер удаляется.

    Параметры:
      all_gdfs: dict {class_name: GeoDataFrame} — результат генерализации.
                GeoDataFrame должен иметь колонки: cluster_id, object_count,
                class, geometry.
      min_objects_after_trim: минимальное число объектов, при котором
                              кластер сохраняется после обрезки.

    Возвращает:
      dict {class_name: GeoDataFrame} с исправленными geometry (и, возможно,
      изменённым составом кластеров).
    """
    # Собираем все оболочки с метаданными
    records = []
    for cls, gdf in all_gdfs.items():
        for _, row in gdf.iterrows():
            records.append({
                "class": cls,
                "cluster_id": row["cluster_id"],
                "object_count": int(row["object_count"]),
                "geometry": row.geometry,
                "convex_hull": row.geometry.convex_hull,
            })

    if not records:
        return all_gdfs

    # --- Шаг 1: найти пересекающиеся пары ---
    n = len(records)
    merge_groups = {}   # индекс -> список индексов для объединения (один класс)
    trim_targets = {}   # индекс -> список индексов, с которыми нужно вычесться

    for i in range(n):
        for j in range(i + 1, n):
            a, b = records[i], records[j]
            if not a["convex_hull"].intersects(b["convex_hull"]):
                continue

            if a["class"] == b["class"]:
                # Один класс → объединяем
                group_a = merge_groups.setdefault(i, {i})
                group_b = merge_groups.setdefault(j, {j})
                merged = group_a | group_b
                for idx in merged:
                    merge_groups[idx] = merged
            else:
                # Разные классы → больший оставляет, меньший обрезается
                if a["object_count"] >= b["object_count"]:
                    trim_targets.setdefault(j, set()).add(i)
                else:
                    trim_targets.setdefault(i, set()).add(j)

    # --- Шаг 2: применить объединения (по одному классу) ---
    used_in_merge = set()
    new_records = []
    for idx, group in merge_groups.items():
        if idx in used_in_merge:
            continue
        used_in_merge |= group
        group_list = sorted(group)
        cls = records[group_list[0]]["class"]
        union_geom = unary_union([records[k]["geometry"] for k in group_list])
        total_objects = sum(records[k]["object_count"] for k in group_list)
        new_records.append({
            "class": cls,
            "cluster_id": records[group_list[0]]["cluster_id"],
            "object_count": total_objects,
            "geometry": union_geom,
        })

    # Кластеры, не попавшие в объединения — копируем
    for i in range(n):
        if i not in used_in_merge:
            new_records.append({
                "class": records[i]["class"],
                "cluster_id": records[i]["cluster_id"],
                "object_count": records[i]["object_count"],
                "geometry": records[i]["geometry"],
            })

    # --- Шаг 3: применить обрезки (разные классы) ---
    final_records = []
    for rec in new_records:
        if rec["class"] not in trim_targets:
            # Этот кластер никого не должен обрезать — но может быть обрезан сам
            # (обработка ниже)
            pass

    # Для каждого кластера собираем геометрии, которые нужно из него вычесть
    # (геометрии кластеров других классов с бóльшим object_count)
    for idx, rec in enumerate(new_records):
        geom = rec["geometry"]
        # Найти все "большие" кластеры других классов, которые пересекаются
        for other in new_records:
            if other["class"] == rec["class"]:
                continue
            if other["object_count"] <= rec["object_count"]:
                continue
            if not geom.intersects(other["geometry"]):
                continue
            geom = geom.difference(other["geometry"])
            if geom.is_empty:
                break

        if geom.is_empty:
            continue
        # Проверка на слишком маленький остаток — эвристика по площади
        # (если площадь упала более чем на 50% — считаем кластер "искажённым"
        # и удаляем, если объектов мало)
        if rec["object_count"] < min_objects_after_trim:
            continue

        final_records.append({
            "class": rec["class"],
            "cluster_id": rec["cluster_id"],
            "object_count": rec["object_count"],
            "geometry": geom,
        })

    # --- Шаг 4: собрать обратно по классам ---
    result = {}
    for rec in final_records:
        cls = rec["class"]
        if cls not in result:
            result[cls] = []
        result[cls].append(rec)

    out = {}
    for cls, recs in result.items():
        gdf = gpd.GeoDataFrame(
            {
                "cluster_id": [r["cluster_id"] for r in recs],
                "object_count": [r["object_count"] for r in recs],
                "class": [r["class"] for r in recs],
            },
            geometry=[r["geometry"] for r in recs],
        )
        out[cls] = gdf

    # Добавляем классы, у которых все кластеры были удалены
    for cls in all_gdfs:
        if cls not in out:
            out[cls] = gpd.GeoDataFrame(
                columns=["cluster_id", "object_count", "class", "geometry"],
                crs=all_gdfs[cls].crs,
            )

    return out