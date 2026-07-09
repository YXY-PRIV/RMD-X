#!/usr/bin/env python3
"""Batch19 multi-detector resource inventory.

This script only reads files and writes inventory artifacts. It does not run
DINOv2, RAXO methods, mask extraction, cache building, or COCO evaluation.
"""

from __future__ import annotations

import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


DETECTORS = ["groundingDINO", "detic", "CoDet", "VLDet"]
ANNOTATION = Path("/datasets/xray-datasets/DET-COMPASS/annotations/full_test.json")
IMAGE_DIR = Path("/datasets/xray-datasets/DET-COMPASS/test")
PROTOTYPES = Path("/models/ICCV25_experimentation/known_prototypes_sam2.pt")
MATERIAL_BANK = Path("/workspace/RAXO/outputs_rmdx/material/material_colour_bank_support_full_bbox_only.json")
INITIAL_ROOT = Path("/models/ICCV25_experimentation/initial_detections")
CACHE_ROOT = Path("/workspace/RAXO/outputs_rmdx/cache")
OUT_DIR = Path("/workspace/RAXO/outputs_rmdx/resource_checks")
INVENTORY_JSON = OUT_DIR / "multidetector_resource_inventory.json"
INVENTORY_CSV = OUT_DIR / "multidetector_resource_inventory.csv"
FILE_HEALTH_JSON = OUT_DIR / "multidetector_file_health.json"

FULL_LOAD_LIMIT_BYTES = 650 * 1024 * 1024


def file_size(path: Path) -> int | None:
    return path.stat().st_size if path.exists() and path.is_file() else None


def tail_nonspace(path: Path, n: int = 8192) -> str:
    if not path.exists() or path.stat().st_size == 0:
        return ""
    with path.open("rb") as f:
        f.seek(max(0, path.stat().st_size - n))
        data = f.read()
    text = data.decode("utf-8", errors="ignore").rstrip()
    return text[-1:] if text else ""


def head_text(path: Path, n: int = 65536) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as f:
        return f.read(n).decode("utf-8", errors="ignore")


def light_json_check(path: Path, expected_keywords: list[str] | None = None) -> dict[str, Any]:
    expected_keywords = expected_keywords or []
    exists = path.exists()
    size = file_size(path)
    result = {
        "exists": exists,
        "size": size,
        "nonempty": bool(size and size > 0),
        "first_nonspace": None,
        "last_nonspace": None,
        "keyword_hits": {},
        "appears_closed": False,
        "appears_json_like": False,
    }
    if not exists or not size:
        return result
    head = head_text(path)
    stripped = head.lstrip()
    result["first_nonspace"] = stripped[:1] if stripped else ""
    result["last_nonspace"] = tail_nonspace(path)
    result["keyword_hits"] = {kw: (kw in head) for kw in expected_keywords}
    result["appears_closed"] = result["last_nonspace"] in {"]", "}"}
    result["appears_json_like"] = result["first_nonspace"] in {"[", "{"} and result["appears_closed"]
    return result


def full_json_load(path: Path, *, force: bool = False) -> dict[str, Any]:
    if not path.exists():
        return {"attempted": False, "valid": False, "reason": "missing"}
    size = path.stat().st_size
    if not force and size > FULL_LOAD_LIMIT_BYTES:
        return {
            "attempted": False,
            "valid": None,
            "reason": f"skipped_full_validation_size_gt_{FULL_LOAD_LIMIT_BYTES}",
        }
    try:
        with path.open() as f:
            data = json.load(f)
    except Exception as exc:
        return {"attempted": True, "valid": False, "error": repr(exc)}
    summary: dict[str, Any] = {"attempted": True, "valid": True, "top_type": type(data).__name__}
    if isinstance(data, list):
        summary["count"] = len(data)
        if data:
            summary["first_keys"] = sorted(data[0].keys()) if isinstance(data[0], dict) else None
    elif isinstance(data, dict):
        summary["keys"] = sorted(data.keys())
        for key in ["images", "annotations", "categories"]:
            if key in data and isinstance(data[key], list):
                summary[f"{key}_count"] = len(data[key])
        if data.get("annotations"):
            ann = data["annotations"][0]
            summary["first_annotation_keys"] = sorted(ann.keys()) if isinstance(ann, dict) else None
    return summary


def load_annotation_summary() -> dict[str, Any]:
    data = json.load(ANNOTATION.open())
    image_names = [img["file_name"] for img in data.get("images", [])]
    image_ids = [int(img["id"]) for img in data.get("images", [])]
    category_ids = [int(cat["id"]) for cat in data.get("categories", [])]
    existing = sum(1 for name in image_names if (IMAGE_DIR / name).exists())
    return {
        "exists": ANNOTATION.exists(),
        "file_size": file_size(ANNOTATION),
        "annotation_image_count": len(image_names),
        "annotation_count": len(data.get("annotations", [])),
        "category_count": len(category_ids),
        "image_ids": image_ids,
        "category_ids": category_ids,
        "test_image_count": len([p for p in IMAGE_DIR.iterdir() if p.is_file()]) if IMAGE_DIR.exists() else 0,
        "test_images_cover_annotation_file_name": existing == len(image_names),
        "missing_test_images_count": len(image_names) - existing,
        "sample_file_names": image_names[:10],
    }


def global_resources(annotation: dict[str, Any]) -> dict[str, Any]:
    cache = CACHE_ROOT / "det_compass_groundingDINO_100_0_full"
    return {
        "annotation": str(ANNOTATION),
        "annotation_exists": ANNOTATION.exists(),
        "annotation_file_size": file_size(ANNOTATION),
        "annotation_image_count": annotation["annotation_image_count"],
        "annotation_count": annotation["annotation_count"],
        "category_count": annotation["category_count"],
        "test_image_dir": str(IMAGE_DIR),
        "test_image_dir_exists": IMAGE_DIR.exists(),
        "test_image_count": annotation["test_image_count"],
        "test_images_cover_annotation_file_name": annotation["test_images_cover_annotation_file_name"],
        "missing_test_images_count": annotation["missing_test_images_count"],
        "prototype_path": str(PROTOTYPES),
        "prototype_exists": PROTOTYPES.exists(),
        "prototype_size": file_size(PROTOTYPES),
        "material_bank_path": str(MATERIAL_BANK),
        "material_bank_exists": MATERIAL_BANK.exists(),
        "material_bank_size": file_size(MATERIAL_BANK),
        "groundingDINO_locked_cache": str(cache),
        "groundingDINO_locked_cache_exists": cache.exists(),
        "groundingDINO_locked_cache_manifest_exists": (Path("/workspace/RAXO/outputs_rmdx/cache_manifest/det_compass_groundingDINO_100_0_full_manifest.json")).exists(),
    }


def detector_paths(detector: str) -> dict[str, Path]:
    d = INITIAL_ROOT / detector
    base = d / "coco_results.bbox_DET-COMPASS.json"
    return {
        "detector_dir": d,
        "initial": base,
        "recovered": d / "coco_results.bbox_DET-COMPASS.recovered_from_with_masks.json",
        "truncated_backup": d / "coco_results.bbox_DET-COMPASS.truncated_backup.json",
        "with_masks": d / "coco_results.bbox_DET-COMPASS_with_masks.json",
        "known_final": d / "coco_results.bbox_DET-COMPASS_with_masks_nms_0.8_our_method_known_final.json",
        "known_final_uncertanty": d / "coco_results.bbox_DET-COMPASS_with_masks_nms_0.8_our_method_known_final_uncertanty.json",
        "cache": CACHE_ROOT / f"det_compass_{detector}_100_0_full",
    }


def initial_or_result_summary(path: Path) -> dict[str, Any]:
    light = light_json_check(path)
    full = full_json_load(path, force=False)
    count = full.get("count") if full.get("valid") is True else None
    return {"light": light, "full": full, "count": count}


def with_masks_consistency(path: Path, full_summary: dict[str, Any], annotation: dict[str, Any]) -> dict[str, Any]:
    if full_summary.get("valid") is not True:
        return {"checked": False, "reason": full_summary.get("reason") or full_summary.get("error") or "not_full_loaded"}
    data = json.load(path.open())
    ann_image_ids = set(annotation["image_ids"])
    ann_category_ids = set(annotation["category_ids"])
    wm_image_ids = {int(img["id"]) for img in data.get("images", [])}
    wm_category_ids = {int(cat["id"]) for cat in data.get("categories", [])}
    annotations = data.get("annotations", [])
    proposals_by_image = Counter(int(ann["image_id"]) for ann in annotations)
    empty_images = sorted(list(wm_image_ids - set(proposals_by_image)))
    sample = annotations[:200]
    required = ["image_id", "category_id", "bbox", "score", "mask"]
    required_present = {key: all(key in ann for ann in sample) for key in required} if sample else {key: False for key in required}
    bbox_looks_xywh = all(
        isinstance(ann.get("bbox"), list)
        and len(ann["bbox"]) == 4
        and float(ann["bbox"][2]) >= 0
        and float(ann["bbox"][3]) >= 0
        for ann in sample
    ) if sample else False
    return {
        "checked": True,
        "images_count_matches_full_test": len(wm_image_ids) == len(ann_image_ids),
        "categories_count_matches_full_test": len(wm_category_ids) == len(ann_category_ids),
        "image_id_set_covers_full_test": ann_image_ids.issubset(wm_image_ids),
        "category_id_set_matches_full_test": wm_category_ids == ann_category_ids,
        "avg_proposals_per_image": (len(annotations) / len(wm_image_ids)) if wm_image_ids else None,
        "empty_image_proposal_count": len(empty_images),
        "empty_image_ids_sample": empty_images[:20],
        "bbox_looks_coco_xywh": bbox_looks_xywh,
        "required_fields_present_in_sample": required_present,
    }


def inspect_detector(detector: str, annotation: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = detector_paths(detector)
    notes = []
    alt_xp = paths["detector_dir"] / "coco_results.bbox_DET-COMPASS-XP_with_masks.json"
    if alt_xp.exists() and not paths["with_masks"].exists():
        notes.append(f"found DET-COMPASS-XP_with_masks alternative: {alt_xp}")

    initial = initial_or_result_summary(paths["initial"])
    recovered = initial_or_result_summary(paths["recovered"])
    with_light = light_json_check(paths["with_masks"], ["images", "annotations", "categories"])
    with_full = full_json_load(paths["with_masks"], force=False)
    known = initial_or_result_summary(paths["known_final"])
    known_unc = initial_or_result_summary(paths["known_final_uncertanty"])
    consistency = with_masks_consistency(paths["with_masks"], with_full, annotation)

    row = {
        "detector": detector,
        "detector_dir_exists": paths["detector_dir"].exists(),
        "initial_detection_exists": paths["initial"].exists(),
        "initial_detection_size": file_size(paths["initial"]),
        "initial_detection_json_valid_light": initial["light"]["appears_json_like"],
        "initial_detection_json_valid_full": initial["full"].get("valid"),
        "initial_detection_count_if_valid": initial["count"],
        "recovered_detection_exists": paths["recovered"].exists(),
        "recovered_detection_size": file_size(paths["recovered"]),
        "recovered_detection_valid": recovered["full"].get("valid"),
        "recovered_detection_count_if_valid": recovered["count"],
        "truncated_backup_exists": paths["truncated_backup"].exists(),
        "with_masks_exists": paths["with_masks"].exists(),
        "with_masks_size": file_size(paths["with_masks"]),
        "with_masks_json_valid_light": with_light["appears_json_like"],
        "with_masks_json_valid_full": with_full.get("valid"),
        "with_masks_image_count_if_valid": with_full.get("images_count"),
        "with_masks_annotation_count_if_valid": with_full.get("annotations_count"),
        "with_masks_category_count_if_valid": with_full.get("categories_count"),
        "known_final_exists": paths["known_final"].exists(),
        "known_final_valid": known["full"].get("valid"),
        "known_final_count_if_valid": known["count"],
        "known_final_uncertanty_exists": paths["known_final_uncertanty"].exists(),
        "known_final_uncertanty_valid": known_unc["full"].get("valid"),
        "known_final_uncertanty_count_if_valid": known_unc["count"],
        "existing_cache_dir_exists": paths["cache"].exists(),
        "existing_cache_manifest_exists": (paths["cache"] / "cache_summary.json").exists(),
        "notes": "; ".join(notes),
    }
    health = {
        "detector": detector,
        "paths": {k: str(v) for k, v in paths.items()},
        "initial_detection": initial,
        "recovered_detection": recovered,
        "with_masks_light": with_light,
        "with_masks_full": with_full,
        "with_masks_consistency": consistency,
        "known_final": known,
        "known_final_uncertanty": known_unc,
        "notes": notes,
    }
    return row, health


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    annotation = load_annotation_summary()
    global_info = global_resources(annotation)
    rows = []
    health = {"global": global_info, "detectors": {}}
    for detector in DETECTORS:
        row, det_health = inspect_detector(detector, annotation)
        rows.append(row)
        health["detectors"][detector] = det_health
    INVENTORY_JSON.write_text(json.dumps({"global": global_info, "detectors": rows}, indent=2))
    FILE_HEALTH_JSON.write_text(json.dumps(health, indent=2))
    write_csv(INVENTORY_CSV, rows)
    print(json.dumps({"inventory": str(INVENTORY_JSON), "csv": str(INVENTORY_CSV), "health": str(FILE_HEALTH_JSON)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
