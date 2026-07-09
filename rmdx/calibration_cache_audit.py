"""Audit helpers for non-test Calib-VTC calibration caches.

These utilities only inspect files. They do not generate proposals, masks, or
feature caches, and they do not use full_test data except as leakage reference.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED_CACHE_FILES = [
    "proposal_meta.jsonl",
    "visual_similarity.pt",
    "material_class_similarity.pt",
    "category_meta.json",
    "prototype_meta.json",
    "cache_summary.json",
]


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl_count(path: str | Path) -> int:
    path = Path(path)
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def full_test_basenames(dataset_root: Path, dataset: str) -> set[str]:
    ann = dataset_root / dataset / "annotations" / "full_test.json"
    if not ann.exists():
        return set()
    data = load_json(ann)
    return {Path(img.get("file_name", "")).name for img in data.get("images", [])}


def calibration_manifest_basenames(rows: list[dict[str, str]], dataset: str) -> set[str]:
    return {
        Path(row.get("image_path", "")).name
        for row in rows
        if row.get("dataset") == dataset and row.get("image_path")
    }


def audit_cache_dir(cache_dir: str | Path, dataset: str, manifest_rows: list[dict[str, str]], dataset_root: Path) -> dict[str, Any]:
    cache_dir = Path(cache_dir)
    required_presence = {name: (cache_dir / name).exists() for name in REQUIRED_CACHE_FILES}
    missing_files = [name for name, ok in required_presence.items() if not ok]
    summary: dict[str, Any] = {}
    if (cache_dir / "cache_summary.json").exists():
        try:
            summary = load_json(cache_dir / "cache_summary.json")
        except Exception as exc:
            summary = {"summary_load_error": f"{type(exc).__name__}: {exc}"}

    proposal_meta_path = cache_dir / "proposal_meta.jsonl"
    proposal_count_from_meta = read_jsonl_count(proposal_meta_path)
    calibration_names = calibration_manifest_basenames(manifest_rows, dataset)
    full_test_names = full_test_basenames(dataset_root, dataset)

    cache_file_names: set[str] = set()
    leakage_overlap = set()
    if proposal_meta_path.exists():
        with proposal_meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                file_name = Path(str(row.get("file_name", ""))).name
                if file_name:
                    cache_file_names.add(file_name)
                    if file_name in full_test_names:
                        leakage_overlap.add(file_name)

    cache_images_subset_of_calibration = bool(cache_file_names) and cache_file_names.issubset(calibration_names)
    if not cache_file_names and proposal_count_from_meta == 0:
        cache_images_subset_of_calibration = False

    if not cache_dir.exists():
        status = "MISSING"
        reason = "cache directory missing"
    elif missing_files:
        status = "INCOMPLETE"
        reason = "missing required files: " + ",".join(missing_files)
    elif leakage_overlap:
        status = "BLOCKED_LEAKAGE_RISK"
        reason = "cache proposal_meta contains full_test file_name overlap"
    elif not cache_images_subset_of_calibration:
        status = "BLOCKED_NOT_CALIBRATION_ONLY"
        reason = "cache proposal_meta file_names are not a subset of calibration image manifest"
    else:
        status = "READY"
        reason = ""

    return {
        "dataset": dataset,
        "cache_dir": str(cache_dir),
        "cache_dir_exists": cache_dir.exists(),
        "required_files_present": not missing_files and cache_dir.exists(),
        "missing_files": ",".join(missing_files),
        "proposal_count": summary.get("proposal_count", proposal_count_from_meta),
        "proposal_count_from_meta": proposal_count_from_meta,
        "image_count": summary.get("image_count", ""),
        "category_count": summary.get("category_count", ""),
        "feature_dim": summary.get("feature_dim", ""),
        "cache_file_name_count": len(cache_file_names),
        "calibration_image_count": len(calibration_names),
        "cache_images_subset_of_calibration": cache_images_subset_of_calibration,
        "full_test_overlap_count": len(leakage_overlap),
        "status": status,
        "failure_reason": reason,
    }
