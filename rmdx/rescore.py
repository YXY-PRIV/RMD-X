"""Identity-copy RMD-X rescoring helpers.

This module intentionally does not implement reliability, material, or risk
algorithms. It only validates the external wrapper path by adding placeholder
fields while preserving COCO evaluation fields.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from rmdx.reliability import (
    apply_margin_filter,
    apply_reliability_filter_and_rescore,
    apply_reliability_rescore,
    summarize_reliability,
)
from rmdx.material import (
    apply_material_append_only,
    apply_material_filter_and_rescore,
    apply_material_rescore,
)


def load_detections(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    with path.open() as f:
        detections = json.load(f)
    if not isinstance(detections, list):
        raise TypeError(f"{path} is not a COCO result list")
    return detections


def add_rmdx_placeholder_fields(
    dets: Iterable[dict[str, Any]], mode: str = "identity_copy"
) -> list[dict[str, Any]]:
    output = []
    for det in dets:
        copied = dict(det)
        copied["score_rmdx"] = copied["score"]
        copied["rmdx_reliability"] = 1.0
        copied["rmdx_material_score"] = 0.0
        copied["rmdx_risk_score"] = 0.0
        copied["rmdx_mode"] = mode
        copied["rmdx_original_score"] = copied["score"]
        copied["rmdx_original_category_id"] = copied["category_id"]
        output.append(copied)
    return output


def save_detections(dets: Iterable[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(list(dets), f)


def filter_by_image_ids(
    dets: Iterable[dict[str, Any]], image_ids: set[int]
) -> list[dict[str, Any]]:
    return [det for det in dets if int(det["image_id"]) in image_ids]


def summarize_detections(dets: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [float(det["score"]) for det in dets if "score" in det]
    by_image = Counter(int(det["image_id"]) for det in dets)
    by_category = Counter(int(det["category_id"]) for det in dets)
    rmdx_fields = [
        "score_rmdx",
        "rmdx_reliability",
        "rmdx_material_score",
        "rmdx_risk_score",
        "rmdx_mode",
        "rmdx_original_score",
        "rmdx_original_category_id",
    ]
    return {
        "detection_count": len(dets),
        "image_count": len(by_image),
        "category_count": len(by_category),
        "score_range": {
            "min": min(scores) if scores else None,
            "max": max(scores) if scores else None,
            "mean": (sum(scores) / len(scores)) if scores else None,
        },
        "rmdx_fields_present": {
            field: bool(dets and field in dets[0]) for field in rmdx_fields
        },
        "detections_per_image_top10": by_image.most_common(10),
        "categories_top10": by_category.most_common(10),
    }


def export_for_coco_eval(dets: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    keep = {
        "image_id",
        "category_id",
        "bbox",
        "score",
        "dist_cosine_inter",
        "dist_cosine_intra",
    }
    return [{k: v for k, v in det.items() if k in keep} for det in dets]
