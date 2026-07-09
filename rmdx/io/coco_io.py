"""Read-only helpers for COCO-style datasets."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any, Optional


def load_coco_dataset(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"{path} is not a COCO-style dict")
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


def summarize_coco_dataset(path: Path, reference_path: Optional[Path] = None) -> dict[str, Any]:
    data = load_coco_dataset(path)
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])
    annotation_fields = sorted(set().union(*(ann.keys() for ann in annotations[:1000]))) if annotations else []
    bbox_lengths = Counter(len(ann.get("bbox", [])) for ann in annotations)
    by_image = Counter(ann.get("image_id") for ann in annotations)
    by_category = Counter(ann.get("category_id") for ann in annotations)
    scores = [float(ann["score"]) for ann in annotations if "score" in ann]
    image_ids = {img.get("id") for img in images}
    ann_image_ids = set(by_image.keys())

    summary: dict[str, Any] = {
        "path": str(path),
        "top_type": type(data).__name__,
        "top_keys": sorted(data.keys()),
        "image_count": len(images),
        "annotation_count": len(annotations),
        "category_count": len(categories),
        "category_sample": categories[:5],
        "image_sample": images[0] if images else None,
        "annotation_sample": annotations[0] if annotations else None,
        "annotation_fields": annotation_fields,
        "bbox_lengths": dict(bbox_lengths),
        "has_score": bool(scores),
        "score_range": _range(scores),
        "has_mask_field": any("mask" in ann for ann in annotations[:1000]),
        "has_segmentation_field": any("segmentation" in ann for ann in annotations[:1000]),
        "annotations_per_image": _count_stats(by_image),
        "category_id_count": len(by_category),
        "category_id_top20": by_category.most_common(20),
        "image_id_alignment_internal": {
            "image_ids": len(image_ids),
            "annotation_image_ids": len(ann_image_ids),
            "annotation_ids_without_image": len(ann_image_ids - image_ids),
            "images_without_annotations": len(image_ids - ann_image_ids),
        },
    }

    if reference_path:
        ref = load_coco_dataset(reference_path)
        ref_ids = {img.get("id") for img in ref.get("images", [])}
        summary["image_id_alignment_with_reference"] = {
            "reference_path": str(reference_path),
            "reference_image_count": len(ref_ids),
            "current_minus_reference": len(image_ids - ref_ids),
            "reference_minus_current": len(ref_ids - image_ids),
            "annotation_minus_reference": len(ann_image_ids - ref_ids),
        }

    return summary

