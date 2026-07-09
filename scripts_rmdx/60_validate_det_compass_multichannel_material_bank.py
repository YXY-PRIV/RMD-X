#!/usr/bin/env python3
"""Batch27: validate DET-COMPASS multi-channel material banks."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path("/workspace/RAXO")
if not ROOT.exists():
    ROOT = Path("/mnt/data/yxy/program/RAXO")

OUT_ROOT = ROOT / "outputs_rmdx" / "material_multichannel_det_compass"
BANK_DIR = OUT_ROOT / "banks"
VALID_DIR = OUT_ROOT / "validation"
OLD_COLOUR_BANK = ROOT / "outputs_rmdx" / "material" / "material_colour_bank_support_full_bbox_only.json"
BANK_TYPES = ["colour", "grey", "high", "low", "density", "multi_channel_all"]
FIELDS = [
    "bank_type",
    "bank_path",
    "json_valid",
    "category_count",
    "feature_dim",
    "feature_names_nonempty",
    "numeric_valid",
    "sample_count_min",
    "sample_count_median",
    "sample_count_max",
    "low_sample_category_count",
    "covered_category_count",
    "missing_category_count",
    "coverage_ratio",
    "missing_category_names",
    "old_colour_coverage_category_count",
    "leakage_check",
    "final_status",
    "notes",
]


def dataset_root() -> Path:
    for root in [
        Path("/datasets/xray-datasets"),
        Path("/mnt/data/raxo/datasets/xray-datasets"),
        Path("/data/raxo/datasets/xray-datasets"),
    ]:
        if (root / "DET-COMPASS" / "annotations" / "full_test.json").exists():
            return root
    return Path("/mnt/data/raxo/datasets/xray-datasets")


def full_test_categories() -> dict[str, int]:
    path = dataset_root() / "DET-COMPASS" / "annotations" / "full_test.json"
    with path.open("r", encoding="utf-8") as f:
        ann = json.load(f)
    return {str(cat["name"]): int(cat["id"]) for cat in ann.get("categories", [])}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def numeric_list_valid(values: Any) -> bool:
    if not isinstance(values, list) or not values:
        return False
    for v in values:
        try:
            fv = float(v)
        except Exception:
            return False
        if not math.isfinite(fv):
            return False
    return True


def old_colour_coverage() -> int:
    if not OLD_COLOUR_BANK.exists():
        return 0
    data = load_json(OLD_COLOUR_BANK)
    names = {str(proto.get("category_name")) for proto in data.get("prototypes", {}).values()}
    return len(names - {""})


def bank_path_for(bank_type: str) -> Path:
    if bank_type == "multi_channel_all":
        return BANK_DIR / "material_bank_multichannel_all_support.json"
    return BANK_DIR / f"material_bank_{bank_type}_support.json"


def validate_bank(bank_type: str, full_names: set[str], old_cov: int) -> dict[str, Any]:
    path = bank_path_for(bank_type)
    row = {
        "bank_type": bank_type,
        "bank_path": str(path),
        "json_valid": False,
        "category_count": 0,
        "feature_dim": 0,
        "feature_names_nonempty": False,
        "numeric_valid": False,
        "sample_count_min": "",
        "sample_count_median": "",
        "sample_count_max": "",
        "low_sample_category_count": "",
        "covered_category_count": 0,
        "missing_category_count": len(full_names),
        "coverage_ratio": 0.0,
        "missing_category_names": "",
        "old_colour_coverage_category_count": old_cov,
        "leakage_check": "",
        "final_status": "BLOCKED",
        "notes": "",
    }
    try:
        bank = load_json(path)
        row["json_valid"] = True
    except Exception as exc:
        row["notes"] = repr(exc)
        return row
    prototypes = bank.get("prototypes", {})
    feature_names = bank.get("feature_names", [])
    sample_counts = [int(proto.get("sample_count", 0)) for proto in prototypes.values()]
    names = {str(proto.get("category_name")) for proto in prototypes.values() if proto.get("category_name") is not None}
    missing = sorted(full_names - names)
    numeric_valid = bool(feature_names)
    for proto in prototypes.values():
        numeric_valid = numeric_valid and numeric_list_valid(proto.get("mean_feature")) and numeric_list_valid(proto.get("std_feature"))
        numeric_valid = numeric_valid and len(proto.get("mean_feature", [])) == len(feature_names)
        numeric_valid = numeric_valid and len(proto.get("std_feature", [])) == len(feature_names)
    low_count = sum(1 for c in sample_counts if c < 3)
    coverage = len(names & full_names) / max(len(full_names), 1)
    leakage = bank.get("leakage_check", "")
    if coverage >= 0.999 and numeric_valid and leakage == "no_full_test_bbox_used":
        status = "READY_WITH_LOW_SAMPLE_WARNING" if low_count else "READY_FOR_BATCH28"
    else:
        status = "BLOCKED"
    row.update(
        {
            "category_count": len(prototypes),
            "feature_dim": len(feature_names),
            "feature_names_nonempty": bool(feature_names),
            "numeric_valid": numeric_valid,
            "sample_count_min": min(sample_counts) if sample_counts else "",
            "sample_count_median": sorted(sample_counts)[len(sample_counts) // 2] if sample_counts else "",
            "sample_count_max": max(sample_counts) if sample_counts else "",
            "low_sample_category_count": low_count,
            "covered_category_count": len(names & full_names),
            "missing_category_count": len(missing),
            "coverage_ratio": round(coverage, 6),
            "missing_category_names": ";".join(missing[:50]),
            "leakage_check": leakage,
            "final_status": status,
            "notes": "样本数低于 3 的类别需在 Batch28 表注/消融中说明。" if low_count else "可用于 Batch28。",
        }
    )
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# DET-COMPASS 多通道 material bank 验证",
        "",
        "说明：该验证只检查 bank JSON、类别覆盖、数值合法性和 leakage 标记，不运行 AP evaluation。",
        "",
        "| bank_type | category_count | feature_dim | coverage_ratio | sample_min | sample_median | sample_max | low_sample | leakage_check | final_status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['bank_type']} | {row['category_count']} | {row['feature_dim']} | {row['coverage_ratio']} | "
            f"{row['sample_count_min']} | {row['sample_count_median']} | {row['sample_count_max']} | "
            f"{row['low_sample_category_count']} | {row['leakage_check']} | {row['final_status']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    VALID_DIR.mkdir(parents=True, exist_ok=True)
    full_names = set(full_test_categories().keys())
    old_cov = old_colour_coverage()
    rows = [validate_bank(bank_type, full_names, old_cov) for bank_type in BANK_TYPES]
    write_csv(VALID_DIR / "multichannel_bank_validation.csv", rows)
    write_md(VALID_DIR / "multichannel_bank_validation.md", rows)
    (VALID_DIR / "multichannel_bank_validation.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"validation_rows": len(rows), "ready": [r["bank_type"] for r in rows if str(r["final_status"]).startswith("READY")]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
