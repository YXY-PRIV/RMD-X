#!/usr/bin/env python3
"""Batch22 multi-dataset resource inventory.

This script only reads resources and writes reports. It does not run DINOv2,
RAXO, pycocotools evaluation, mask extraction, cache build, or file repair.
"""

from __future__ import annotations

import csv
import glob
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path("/workspace/RAXO")
OUT_DIR = ROOT / "outputs_rmdx" / "resource_checks"
DATASET_ROOT = Path("/datasets/xray-datasets")
MODEL_ROOT = Path("/models/ICCV25_experimentation")
DET_ROOT = MODEL_ROOT / "initial_detections"
PROTOTYPES = MODEL_ROOT / "known_prototypes_sam2.pt"
WEB_PROT_DIR = MODEL_ROOT / "prototypes_web_2"
MATERIAL_DIR = ROOT / "outputs_rmdx" / "material"

DATASETS = ["DET-COMPASS", "PIXray", "pidray", "CLCXray", "HiXray", "DvXray"]
DETECTORS = ["groundingDINO", "VLDet", "CoDet"]

INVENTORY_JSON = OUT_DIR / "multidataset_resource_inventory.json"
INVENTORY_CSV = OUT_DIR / "multidataset_resource_inventory.csv"
FILE_HEALTH_JSON = OUT_DIR / "multidataset_file_health.json"
MATERIAL_CSV = OUT_DIR / "multidataset_material_bank_readiness.csv"
MATERIAL_MD = OUT_DIR / "multidataset_material_bank_readiness.md"

FULL_LOAD_LIMIT = 700 * 1024 * 1024


def load_json_safe(path: Path) -> tuple[Any | None, str | None]:
    try:
        with path.open("r") as f:
            return json.load(f), None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def file_size(path: Path) -> int | None:
    return path.stat().st_size if path.exists() else None


def image_files(path: Path) -> set[str]:
    if not path.is_dir():
        return set()
    return {str(p.relative_to(path)) for p in path.rglob("*") if p.is_file()}


def annotation_info(dataset: str) -> dict[str, Any]:
    ds_dir = DATASET_ROOT / dataset
    ann_path = ds_dir / "annotations" / "full_test.json"
    test_dir = ds_dir / "test"
    info: dict[str, Any] = {
        "dataset": dataset,
        "dataset_dir": str(ds_dir),
        "dataset_dir_exists": ds_dir.is_dir(),
        "annotation_path": str(ann_path),
        "annotation_exists": ann_path.is_file(),
        "annotation_valid": False,
        "annotation_error": None,
        "image_count": None,
        "annotation_count": None,
        "category_count": None,
        "category_ids": [],
        "category_names": [],
        "test_dir": str(test_dir),
        "test_dir_exists": test_dir.is_dir(),
        "test_image_count": 0,
        "test_images_cover_annotation_file_name": False,
        "missing_images_count": None,
        "missing_images_sample": [],
        "extra_images_count": None,
        "file_suffix_distribution": {},
        "image_id_type": None,
        "category_id_type": None,
    }
    data, err = load_json_safe(ann_path) if ann_path.is_file() else (None, None)
    if err:
        info["annotation_error"] = err
        return info
    if not isinstance(data, dict):
        if ann_path.is_file():
            info["annotation_error"] = "top-level annotation JSON is not dict"
        return info

    info["annotation_valid"] = True
    images = data.get("images", [])
    anns = data.get("annotations", [])
    cats = data.get("categories", [])
    info["image_count"] = len(images)
    info["annotation_count"] = len(anns)
    info["category_count"] = len(cats)
    info["category_ids"] = sorted([c.get("id") for c in cats if "id" in c], key=lambda x: str(x))
    info["category_names"] = [str(c.get("name")) for c in cats]
    if images:
        info["image_id_type"] = type(images[0].get("id")).__name__
    if cats:
        info["category_id_type"] = type(cats[0].get("id")).__name__
    suffixes = Counter(Path(str(img.get("file_name", ""))).suffix.lower() or "<none>" for img in images)
    info["file_suffix_distribution"] = dict(sorted(suffixes.items()))

    files = image_files(test_dir)
    info["test_image_count"] = len(files)
    expected = {str(img.get("file_name", "")) for img in images}
    missing = sorted(expected - files)
    extra = sorted(files - expected)
    info["missing_images_count"] = len(missing)
    info["missing_images_sample"] = missing[:20]
    info["extra_images_count"] = len(extra)
    info["test_images_cover_annotation_file_name"] = len(missing) == 0 and bool(expected)
    return info


def light_json_check(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "exists": path.is_file(),
        "size": file_size(path),
        "nonempty": False,
        "head_reasonable": False,
        "tail_reasonable": False,
        "full_valid": None,
        "top_type": None,
        "count": None,
        "images": None,
        "annotations": None,
        "categories": None,
        "skipped_full_validation": False,
        "error": None,
    }
    if not path.is_file():
        return info
    size = path.stat().st_size
    info["nonempty"] = size > 0
    if size > 0:
        with path.open("rb") as f:
            head = f.read(8192)
            f.seek(max(0, size - 8192))
            tail = f.read(8192)
        head_text = head.decode("utf-8", errors="replace").lstrip()
        tail_text = tail.decode("utf-8", errors="replace").rstrip()
        info["head_reasonable"] = head_text.startswith(("{", "["))
        info["tail_reasonable"] = tail_text.endswith(("}", "]"))
    if size > FULL_LOAD_LIMIT:
        info["skipped_full_validation"] = True
        return info
    data, err = load_json_safe(path)
    if err:
        info["full_valid"] = False
        info["error"] = err
        return info
    info["full_valid"] = True
    info["top_type"] = type(data).__name__
    if isinstance(data, list):
        info["count"] = len(data)
    elif isinstance(data, dict):
        info["images"] = len(data.get("images", []))
        info["annotations"] = len(data.get("annotations", []))
        info["categories"] = len(data.get("categories", []))
        info["count"] = info["annotations"]
    return info


def with_masks_consistency(path: Path, ann: dict[str, Any]) -> dict[str, Any]:
    result = {
        "can_check": False,
        "image_count_match": None,
        "category_count_match": None,
        "image_id_set_match": None,
        "category_id_set_match": None,
        "has_required_fields": None,
        "bbox_len4": None,
        "score_present": None,
        "mask_present": None,
        "empty_image_proposal_count": None,
        "proposal_count_min": None,
        "proposal_count_mean": None,
        "proposal_count_max": None,
        "error": None,
    }
    if not path.is_file() or path.stat().st_size > FULL_LOAD_LIMIT:
        result["error"] = "missing or skipped full load due to size"
        return result
    data, err = load_json_safe(path)
    if err or not isinstance(data, dict):
        result["error"] = err or "top-level not dict"
        return result
    result["can_check"] = True
    full_images = ann.get("images", [])
    full_cats = ann.get("categories", [])
    wm_images = data.get("images", [])
    wm_cats = data.get("categories", [])
    wm_anns = data.get("annotations", [])
    full_img_ids = {i.get("id") for i in full_images}
    full_cat_ids = {c.get("id") for c in full_cats}
    wm_img_ids = {i.get("id") for i in wm_images}
    wm_cat_ids = {c.get("id") for c in wm_cats}
    result["image_count_match"] = len(full_images) == len(wm_images)
    result["category_count_match"] = len(full_cats) == len(wm_cats)
    result["image_id_set_match"] = full_img_ids == wm_img_ids
    result["category_id_set_match"] = full_cat_ids == wm_cat_ids
    checked = wm_anns[: min(1000, len(wm_anns))]
    result["has_required_fields"] = all({"image_id", "category_id", "bbox", "score", "mask"} <= set(a) for a in checked) if checked else False
    result["bbox_len4"] = all(isinstance(a.get("bbox"), list) and len(a.get("bbox")) == 4 for a in checked) if checked else False
    result["score_present"] = all("score" in a for a in checked) if checked else False
    result["mask_present"] = all("mask" in a for a in checked) if checked else False
    per_img = Counter(a.get("image_id") for a in wm_anns)
    counts = [per_img.get(img_id, 0) for img_id in full_img_ids]
    result["empty_image_proposal_count"] = sum(1 for c in counts if c == 0)
    if counts:
        result["proposal_count_min"] = min(counts)
        result["proposal_count_mean"] = sum(counts) / len(counts)
        result["proposal_count_max"] = max(counts)
    return result


def detector_resource(dataset: str, detector: str, ann_data: dict[str, Any] | None) -> dict[str, Any]:
    ddir = DET_ROOT / detector
    initial = ddir / f"coco_results.bbox_{dataset}.json"
    recovered = ddir / f"coco_results.bbox_{dataset}.recovered_from_with_masks.json"
    backup = ddir / f"coco_results.bbox_{dataset}.truncated_backup.json"
    with_masks = ddir / f"coco_results.bbox_{dataset}_with_masks.json"
    known = ddir / f"coco_results.bbox_{dataset}_with_masks_nms_0.8_our_method_known_final.json"
    known_unc = ddir / f"coco_results.bbox_{dataset}_with_masks_nms_0.8_our_method_known_final_uncertanty.json"
    alternatives = sorted(
        set(glob.glob(str(ddir / f"coco_results.bbox*{dataset}*with_masks*.json"))) - {str(with_masks), str(known), str(known_unc)}
    )
    init_h = light_json_check(initial)
    rec_h = light_json_check(recovered)
    wm_h = light_json_check(with_masks)
    known_h = light_json_check(known)
    unc_h = light_json_check(known_unc)
    consistency = with_masks_consistency(with_masks, ann_data) if ann_data else {"can_check": False}
    return {
        "dataset": dataset,
        "detector": detector,
        "detector_dir_exists": ddir.is_dir(),
        "initial_detection_path": str(initial),
        "initial_detection_exists": initial.is_file(),
        "initial_detection_size": init_h["size"],
        "initial_detection_json_valid_light": init_h["head_reasonable"] and init_h["tail_reasonable"],
        "initial_detection_json_valid_full": init_h["full_valid"],
        "initial_detection_count_if_valid": init_h["count"] if init_h["full_valid"] else None,
        "recovered_detection_exists": recovered.is_file(),
        "recovered_detection_valid": rec_h["full_valid"],
        "truncated_backup_exists": backup.is_file(),
        "with_masks_path": str(with_masks),
        "with_masks_exists": with_masks.is_file(),
        "with_masks_size": wm_h["size"],
        "with_masks_json_valid_light": wm_h["head_reasonable"] and wm_h["tail_reasonable"],
        "with_masks_json_valid_full": wm_h["full_valid"],
        "with_masks_image_count_if_valid": wm_h["images"],
        "with_masks_annotation_count_if_valid": wm_h["annotations"],
        "with_masks_category_count_if_valid": wm_h["categories"],
        "with_masks_avg_proposals_per_image": consistency.get("proposal_count_mean"),
        "with_masks_has_required_fields": consistency.get("has_required_fields"),
        "with_masks_consistency": consistency,
        "known_final_exists": known.is_file(),
        "known_final_valid": known_h["full_valid"],
        "known_final_count_if_valid": known_h["count"] if known_h["full_valid"] else None,
        "known_final_uncertanty_exists": known_unc.is_file(),
        "known_final_uncertanty_valid": unc_h["full_valid"],
        "known_final_uncertanty_count_if_valid": unc_h["count"] if unc_h["full_valid"] else None,
        "alternative_candidates": alternatives,
        "notes": "",
    }


def load_prototypes() -> dict[str, Any]:
    info = {
        "path": str(PROTOTYPES),
        "exists": PROTOTYPES.is_file(),
        "can_load_light": False,
        "prototype_count": None,
        "feature_dim": None,
        "supercategory_count": None,
        "supercategories": [],
        "contains_negative": False,
        "error": None,
    }
    if not PROTOTYPES.is_file():
        return info
    try:
        import torch

        data = torch.load(PROTOTYPES, map_location="cpu")
        info["can_load_light"] = True
        info["prototype_count"] = len(data)
        supers = [str(v.get("supercategory")) for v in data.values() if isinstance(v, dict) and "supercategory" in v]
        info["supercategories"] = sorted(set(supers))
        info["supercategory_count"] = len(set(supers))
        info["contains_negative"] = any("negative" in str(k).lower() or str(v.get("supercategory")).lower() == "negative" for k, v in data.items() if isinstance(v, dict))
        for v in data.values():
            proto = v.get("prototype") if isinstance(v, dict) else None
            if proto is not None and hasattr(proto, "numel"):
                info["feature_dim"] = int(proto.numel())
                break
    except Exception as exc:  # noqa: BLE001
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def prototype_coverage(dataset_infos: dict[str, dict[str, Any]], proto_info: dict[str, Any]) -> dict[str, Any]:
    supercats = set(proto_info.get("supercategories") or [])
    out = {}
    for ds, info in dataset_infos.items():
        names = set(info.get("category_names") or [])
        matched = sorted(names & supercats)
        missing = sorted(names - supercats)
        out[ds] = {
            "matched_category_count": len(matched),
            "missing_category_count": len(missing),
            "missing_category_names_sample": missing[:30],
            "coverage_ratio": (len(matched) / len(names)) if names else None,
        }
    return out


def web_prototypes() -> dict[str, Any]:
    out = {}
    for ds in DATASETS:
        p = WEB_PROT_DIR / f"{ds}_prot_sam2_style2_v2.pt"
        info = {"path": str(p), "exists": p.is_file(), "size": file_size(p), "can_load_light": False, "prototype_count": None, "notes": ""}
        if p.is_file():
            try:
                import torch

                obj = torch.load(p, map_location="cpu")
                info["can_load_light"] = True
                info["prototype_count"] = len(obj) if hasattr(obj, "__len__") else None
            except Exception as exc:  # noqa: BLE001
                info["notes"] = f"{type(exc).__name__}: {exc}"
        out[ds] = info
    return out


def material_readiness(dataset_infos: dict[str, dict[str, Any]], coverage: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    existing_banks = sorted(str(p) for p in MATERIAL_DIR.glob("material_colour_bank_*.json"))
    transfer_candidates = [
        str(MATERIAL_DIR / "material_colour_bank_support_full_bbox_only.json")
    ] if (MATERIAL_DIR / "material_colour_bank_support_full_bbox_only.json").is_file() else []
    leakage_risk_banks = [p for p in existing_banks if p.endswith("material_colour_bank_bbox_only.json")]
    for ds, info in dataset_infos.items():
        ds_dir = DATASET_ROOT / ds
        candidates = [
            ds_dir / "support",
            ds_dir / "support_full",
            ds_dir / "annotations" / "support_full.json",
            ds_dir / "annotations" / "support.json",
            MODEL_ROOT / "support_full",
            MODEL_ROOT / "support_full" / ds,
        ]
        support_images = [str(p) for p in candidates if p.is_dir() and any(x.is_file() for x in p.rglob("*"))]
        support_annotations = [str(p) for p in candidates if p.is_file() and p.suffix.lower() == ".json"]
        dataset_banks = sorted(str(p) for p in MATERIAL_DIR.glob(f"material_colour_bank*{ds}*.json"))
        if ds == "DET-COMPASS" and (MATERIAL_DIR / "material_colour_bank_support_full_bbox_only.json").is_file():
            status = "READY_EXISTING_BANK"
            source = str(MATERIAL_DIR / "material_colour_bank_support_full_bbox_only.json")
        elif dataset_banks:
            status = "READY_EXISTING_BANK"
            source = dataset_banks[0]
        elif support_annotations and support_images:
            status = "READY_CAN_BUILD_FROM_SUPPORT"
            source = support_annotations[0]
        elif support_annotations or support_images:
            status = "NEEDS_SUPPORT_MAPPING"
            source = (support_annotations or support_images)[0]
        elif transfer_candidates:
            status = "ONLY_TRANSFER_BANK_AVAILABLE"
            source = transfer_candidates[0]
        elif existing_banks:
            status = "ONLY_TRANSFER_BANK_AVAILABLE"
            source = existing_banks[0]
        else:
            status = "BLOCKED_NO_SUPPORT_SOURCE"
            source = ""
        notes = "full_test GT is not used for material bank construction"
        if leakage_risk_banks:
            notes += f"; leakage-risk exploratory bank present but not selected: {leakage_risk_banks[:2]}"
        rows.append(
            {
                "dataset": ds,
                "existing_material_bank_found": bool(dataset_banks) or (ds == "DET-COMPASS" and (MATERIAL_DIR / "material_colour_bank_support_full_bbox_only.json").is_file()),
                "support_images_found": bool(support_images),
                "support_annotations_found": bool(support_annotations),
                "possible_support_source": source,
                "category_name_mapping_possible": (coverage.get(ds, {}).get("coverage_ratio") or 0) > 0,
                "expected_category_coverage": coverage.get(ds, {}).get("coverage_ratio"),
                "material_bank_readiness_status": status,
                "notes": notes,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_md_table(path: Path, title: str, rows: list[dict[str, Any]], fields: list[str]) -> None:
    lines = [f"# {title}", "", "| " + " | ".join(fields) + " |", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset_infos = {ds: annotation_info(ds) for ds in DATASETS}
    ann_data = {}
    for ds in DATASETS:
        p = DATASET_ROOT / ds / "annotations" / "full_test.json"
        data, _ = load_json_safe(p) if p.is_file() else (None, None)
        ann_data[ds] = data if isinstance(data, dict) else None
    proto_info = load_prototypes()
    coverage = prototype_coverage(dataset_infos, proto_info)
    web = web_prototypes()
    detector_rows = [
        detector_resource(ds, detector, ann_data.get(ds))
        for ds in DATASETS
        for detector in DETECTORS
    ]
    mat_rows = material_readiness(dataset_infos, coverage)

    inventory = {
        "datasets": dataset_infos,
        "prototypes": proto_info,
        "prototype_coverage": coverage,
        "web_prototypes": web,
        "detector_resources": detector_rows,
        "material_bank_readiness": mat_rows,
        "constraints": {
            "ran_dinov2": False,
            "ran_raxo": False,
            "ran_ap_eval": False,
            "modified_resources": False,
            "used_full_test_material_bank": False,
        },
    }
    INVENTORY_JSON.write_text(json.dumps(inventory, indent=2) + "\n")
    FILE_HEALTH_JSON.write_text(json.dumps({"detector_resources": detector_rows}, indent=2) + "\n")
    ds_fields = [
        "dataset", "dataset_dir_exists", "annotation_exists", "annotation_valid", "image_count",
        "annotation_count", "category_count", "test_dir_exists", "test_image_count",
        "test_images_cover_annotation_file_name", "missing_images_count", "extra_images_count",
        "image_id_type", "category_id_type",
    ]
    write_csv(INVENTORY_CSV, [dataset_infos[ds] for ds in DATASETS], ds_fields)
    mat_fields = [
        "dataset", "existing_material_bank_found", "support_images_found", "support_annotations_found",
        "possible_support_source", "category_name_mapping_possible", "expected_category_coverage",
        "material_bank_readiness_status", "notes",
    ]
    write_csv(MATERIAL_CSV, mat_rows, mat_fields)
    write_md_table(MATERIAL_MD, "Multidataset Material Bank Readiness", mat_rows, mat_fields)
    print(f"inventory_json={INVENTORY_JSON}")
    print(f"inventory_csv={INVENTORY_CSV}")
    print(f"file_health_json={FILE_HEALTH_JSON}")
    print(f"material_csv={MATERIAL_CSV}")
    print(f"material_md={MATERIAL_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
