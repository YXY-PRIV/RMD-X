#!/usr/bin/env python3
"""Validate cache reproduction against locked Batch14/17 metrics."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


CACHE_GRID = Path("/workspace/RAXO/outputs_rmdx/rmdx_results/cache_fusion_grid_results.csv")
DIRECT_SUMMARY = Path("/workspace/RAXO/outputs_rmdx/rmdx_results/preargmax_targeted_summary.csv")
OUT_CSV = Path("/workspace/RAXO/outputs_rmdx/rmdx_results/cache_reproduction_lock.csv")
OUT_MD = Path("/workspace/RAXO/outputs_rmdx/rmdx_results/cache_reproduction_lock.md")

FIELDS = [
    "check_name",
    "source",
    "mode",
    "lambda_material",
    "lambda_reliability",
    "expected_AP",
    "actual_AP",
    "diff_AP",
    "expected_AP50",
    "actual_AP50",
    "diff_AP50",
    "expected_AP75",
    "actual_AP75",
    "diff_AP75",
    "passed",
    "notes",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def find_row(rows: list[dict[str, str]], mode: str, lm: float, lr: float) -> dict[str, str]:
    for row in rows:
        if (
            row["mode"] == mode
            and abs(float(row["lambda_material"]) - lm) <= 1e-9
            and abs(float(row["lambda_reliability"]) - lr) <= 1e-9
        ):
            return row
    raise KeyError((mode, lm, lr))


def check_metric(
    *,
    check_name: str,
    source: str,
    row: dict[str, str],
    expected: dict[str, float],
    tolerance: float,
    notes: str,
) -> dict[str, Any]:
    actual = {k: float(row[k]) for k in ["AP", "AP50", "AP75"]}
    diffs = {k: abs(actual[k] - expected[k]) for k in expected}
    passed = all(v <= tolerance for v in diffs.values())
    return {
        "check_name": check_name,
        "source": source,
        "mode": row["mode"],
        "lambda_material": row["lambda_material"],
        "lambda_reliability": row["lambda_reliability"],
        "expected_AP": expected["AP"],
        "actual_AP": actual["AP"],
        "diff_AP": diffs["AP"],
        "expected_AP50": expected["AP50"],
        "actual_AP50": actual["AP50"],
        "diff_AP50": diffs["AP50"],
        "expected_AP75": expected["AP75"],
        "actual_AP75": actual["AP75"],
        "diff_AP75": diffs["AP75"],
        "passed": passed,
        "notes": notes,
    }


def direct_vs_cache(cache_row: dict[str, str], direct_row: dict[str, str]) -> dict[str, Any]:
    expected = {k: float(direct_row[k]) for k in ["AP", "AP50", "AP75"]}
    return check_metric(
        check_name="direct_vs_cache_lm0p05",
        source="cache_fusion_grid_results.csv vs preargmax_targeted_summary.csv",
        row=cache_row,
        expected=expected,
        tolerance=0.001,
        notes="Cache material_preargmax lm=0.05 must match direct targeted run within 0.001.",
    )


def write_outputs(rows: list[dict[str, Any]]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with OUT_MD.open("w") as f:
        f.write("# Cache Reproduction Lock\n\n")
        f.write("| " + " | ".join(FIELDS) + " |\n")
        f.write("|" + "|".join(["---"] * len(FIELDS)) + "|\n")
        for row in rows:
            f.write("| " + " | ".join(str(row.get(field, "")) for field in FIELDS) + " |\n")


def main() -> int:
    cache_rows = read_rows(CACHE_GRID)
    direct_rows = read_rows(DIRECT_SUMMARY)
    cache_visual = find_row(cache_rows, "visual_only", 0.0, 0.0)
    cache_lm005 = find_row(cache_rows, "material_preargmax", 0.05, 0.0)
    direct_lm005 = find_row(direct_rows, "material_preargmax", 0.05, 0.0)

    rows = [
        check_metric(
            check_name="visual_only_baseline",
            source="cache_fusion_grid_results.csv",
            row=cache_visual,
            expected={"AP": 0.480, "AP50": 0.544, "AP75": 0.488},
            tolerance=0.001,
            notes="Cache visual_only must reproduce RAXO baseline.",
        ),
        check_metric(
            check_name="material_preargmax_lm0p05",
            source="cache_fusion_grid_results.csv",
            row=cache_lm005,
            expected={"AP": 0.501, "AP50": 0.561, "AP75": 0.515},
            tolerance=0.001,
            notes="Cache lm=0.05 must reproduce direct pre-argmax best.",
        ),
        direct_vs_cache(cache_lm005, direct_lm005),
    ]
    write_outputs(rows)
    print(f"wrote {OUT_CSV}")
    return 0 if all(str(row["passed"]) == "True" or row["passed"] is True for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
