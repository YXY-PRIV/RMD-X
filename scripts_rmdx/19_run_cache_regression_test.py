#!/usr/bin/env python3
"""Run Batch18 cache regression checks."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import torch


CACHE_DIR = Path("/workspace/RAXO/outputs_rmdx/cache/det_compass_groundingDINO_100_0_full")
MANIFEST = Path("/workspace/RAXO/outputs_rmdx/cache_manifest/det_compass_groundingDINO_100_0_full_manifest.json")
LOCK_CSV = Path("/workspace/RAXO/outputs_rmdx/rmdx_results/cache_reproduction_lock.csv")
GRID_CSV = Path("/workspace/RAXO/outputs_rmdx/rmdx_results/cache_fusion_grid_results.csv")
OUT_JSON = Path("/workspace/RAXO/outputs_rmdx/rmdx_results/cache_regression_test_summary.json")
OUT_MD = Path("/workspace/RAXO/outputs_rmdx/rmdx_results/cache_regression_test_summary.md")


def add_check(rows: list[dict[str, Any]], name: str, passed: bool, details: Any) -> None:
    rows.append({"check": name, "passed": bool(passed), "details": details})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def main() -> int:
    checks: list[dict[str, Any]] = []
    add_check(checks, "manifest_exists", MANIFEST.exists(), str(MANIFEST))

    required = [
        "proposal_meta.jsonl",
        "proposal_features.pt",
        "visual_similarity.pt",
        "material_class_similarity.pt",
        "prototype_meta.json",
        "category_meta.json",
        "cache_summary.json",
    ]
    missing = [name for name in required if not (CACHE_DIR / name).exists()]
    add_check(checks, "required_cache_files_exist", not missing, {"missing": missing})

    summary = json.load((CACHE_DIR / "cache_summary.json").open())
    proposal_count = int(summary["proposal_count"])
    add_check(checks, "proposal_count_is_68741", proposal_count == 68741, proposal_count)
    add_check(checks, "feature_dim_is_768", int(summary["feature_dim"]) == 768, summary["feature_dim"])

    proposal_features = torch.load(CACHE_DIR / "proposal_features.pt", map_location="cpu")
    visual = torch.load(CACHE_DIR / "visual_similarity.pt", map_location="cpu")
    material = torch.load(CACHE_DIR / "material_class_similarity.pt", map_location="cpu")
    add_check(checks, "proposal_features_shape", list(proposal_features.shape) == [68741, 768], list(proposal_features.shape))
    add_check(checks, "visual_similarity_first_dim", int(visual.shape[0]) == proposal_count, list(visual.shape))
    add_check(checks, "material_similarity_first_dim", int(material.shape[0]) == proposal_count, list(material.shape))

    prototype_meta = json.load((CACHE_DIR / "prototype_meta.json").open())
    category_meta = json.load((CACHE_DIR / "category_meta.json").open())
    add_check(checks, "prototype_meta_readable", isinstance(prototype_meta, list) and len(prototype_meta) == 3128, len(prototype_meta))
    add_check(checks, "category_meta_readable", isinstance(category_meta, list) and len(category_meta) == 307, len(category_meta))

    grid_rows = read_csv(GRID_CSV)
    add_check(checks, "cache_grid_has_9_rows", len(grid_rows) == 9, len(grid_rows))

    lock_rows = read_csv(LOCK_CSV)
    lock_map = {row["check_name"]: row for row in lock_rows}
    add_check(
        checks,
        "visual_only_reproduction_passed",
        lock_map.get("visual_only_baseline", {}).get("passed") == "True",
        lock_map.get("visual_only_baseline"),
    )
    add_check(
        checks,
        "lm0p05_reproduction_passed",
        lock_map.get("material_preargmax_lm0p05", {}).get("passed") == "True",
        lock_map.get("material_preargmax_lm0p05"),
    )
    add_check(
        checks,
        "direct_vs_cache_lm0p05_passed",
        lock_map.get("direct_vs_cache_lm0p05", {}).get("passed") == "True",
        lock_map.get("direct_vs_cache_lm0p05"),
    )

    overall = all(row["passed"] for row in checks)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({"overall_passed": overall, "checks": checks}, indent=2))
    lines = [
        "# Cache Regression Test Summary",
        "",
        f"- overall_passed: `{overall}`",
        "",
        "| check | passed | details |",
        "|---|---:|---|",
    ]
    for row in checks:
        lines.append(f"| {row['check']} | {row['passed']} | `{row['details']}` |")
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(json.dumps({"overall_passed": overall, "json": str(OUT_JSON), "md": str(OUT_MD)}, indent=2))
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
