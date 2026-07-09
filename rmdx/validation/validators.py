"""Lightweight validators for RMD-X input contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def check(name: str, ok: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "detail": detail}


def validate_contracts(
    *,
    prototype_summary: dict[str, Any],
    annotation_summary: dict[str, Any],
    with_masks_summary: dict[str, Any],
    known_final_summary: dict[str, Any],
    image_dir: Path,
    annotation_path: Path,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    checks.append(
        check(
            "prototype_feature_dim_768",
            prototype_summary.get("feature_dim") == 768,
            f"feature_dim={prototype_summary.get('feature_dim')}",
        )
    )

    top_keys = set(with_masks_summary.get("top_keys", []))
    checks.append(
        check(
            "with_masks_has_coco_keys",
            {"images", "categories", "annotations"}.issubset(top_keys),
            f"top_keys={sorted(top_keys)}",
        )
    )

    fields = set(with_masks_summary.get("annotation_fields", []))
    required_with_mask = {"image_id", "category_id", "bbox", "score", "mask"}
    checks.append(
        check(
            "with_masks_annotation_required_fields",
            required_with_mask.issubset(fields),
            f"missing={sorted(required_with_mask - fields)}",
        )
    )

    result_fields = set(known_final_summary.get("fields", []))
    required_result = {"image_id", "category_id", "bbox", "score", "dist_cosine_inter", "dist_cosine_intra"}
    checks.append(
        check(
            "known_final_result_fields",
            required_result.issubset(result_fields),
            f"missing={sorted(required_result - result_fields)}",
        )
    )

    alignment = with_masks_summary.get("image_id_alignment_with_reference", {})
    checks.append(
        check(
            "full_test_with_masks_image_alignment",
            alignment.get("current_minus_reference") == 0
            and alignment.get("reference_minus_current") == 0
            and alignment.get("annotation_minus_reference") == 0,
            str(alignment),
        )
    )

    with annotation_path.open() as f:
        ann = json.load(f)
    file_names = [img["file_name"] for img in ann.get("images", [])]
    missing = [fn for fn in file_names if not (image_dir / fn).is_file()]
    checks.append(
        check(
            "det_compass_test_images_cover_full_test",
            len(missing) == 0,
            f"missing_count={len(missing)} sample={missing[:10]}",
        )
    )

    return checks

