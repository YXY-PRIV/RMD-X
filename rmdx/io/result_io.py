"""Read-only helpers for COCO result-list files."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any, Optional


def load_result_list(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"{path} is not a COCO result list")
    return data


def _range(values: list[float]) -> dict[str, Optional[float]]:
    if not values:
        return {"min": None, "max": None, "mean": None}
    return {"min": min(values), "max": max(values), "mean": sum(values) / len(values)}


def _count_stats(counter: Counter[int]) -> dict[str, Any]:
    values = list(counter.values())
    if not values:
        return {"groups": 0, "min": 0, "max": 0, "mean": 0, "median": 0, "top10": []}
    return {
        "groups": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "median": median(values),
        "top10": counter.most_common(10),
    }


def summarize_result_list(path: Path) -> dict[str, Any]:
    data = load_result_list(path)
    fields = sorted(set().union(*(det.keys() for det in data[:1000]))) if data else []
    by_image = Counter(det.get("image_id") for det in data)
    by_category = Counter(det.get("category_id") for det in data)
    scores = [float(det["score"]) for det in data if "score" in det]
    dist_inter = [float(det["dist_cosine_inter"]) for det in data if "dist_cosine_inter" in det]
    dist_intra = [float(det["dist_cosine_intra"]) for det in data if "dist_cosine_intra" in det]

    return {
        "path": str(path),
        "top_type": type(data).__name__,
        "file_size_bytes": path.stat().st_size,
        "detection_count": len(data),
        "image_count": len(by_image),
        "category_id_count": len(by_category),
        "fields": fields,
        "sample": data[0] if data else None,
        "required_fields_present_in_sample": {
            field: bool(data and field in data[0])
            for field in ["image_id", "category_id", "bbox", "score", "dist_cosine_inter", "dist_cosine_intra"]
        },
        "score_range": _range(scores),
        "dist_cosine_inter_range": _range(dist_inter),
        "dist_cosine_intra_range": _range(dist_intra),
        "detections_per_image": _count_stats(by_image),
        "category_id_top20": by_category.most_common(20),
    }


def compare_result_counts(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_count = int(before.get("detection_count", 0))
    after_count = int(after.get("detection_count", 0))
    removed = before_count - after_count
    return {
        "before_count": before_count,
        "after_count": after_count,
        "removed_count": removed,
        "removed_ratio": (removed / before_count) if before_count else None,
    }

