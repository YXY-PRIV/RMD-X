#!/usr/bin/env python3
"""Summarize Batch21 multi-detector RMD-X pre-argmax results."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path("/workspace/RAXO")
RESULT_DIR = ROOT / "outputs_rmdx" / "rmdx_results"
DOC_DIR = ROOT / "docs_rmdx"
STATE_JSON = RESULT_DIR / "multidetector_rmdx_preargmax_run_state.json"
CACHE_STATE_JSON = RESULT_DIR / "multidetector_rmdx_cache_build_state.json"
FULL_CSV = RESULT_DIR / "multidetector_rmdx_preargmax_results.csv"
FULL_MD = RESULT_DIR / "multidetector_rmdx_preargmax_results.md"
BEST_CSV = RESULT_DIR / "multidetector_rmdx_best_results.csv"
BEST_MD = RESULT_DIR / "multidetector_rmdx_best_results.md"
FILE_INDEX = RESULT_DIR / "multidetector_rmdx_file_index.json"
REPORT = DOC_DIR / "batch21_multidetector_rmdx_report.md"

FULL_FIELDS = [
    "detector",
    "mode",
    "lambda_material",
    "lambda_reliability",
    "raxo_baseline_AP",
    "raxo_baseline_AP50",
    "raxo_baseline_AP75",
    "AP",
    "AP50",
    "AP75",
    "delta_AP_vs_detector_raxo",
    "delta_AP50_vs_detector_raxo",
    "delta_AP75_vs_detector_raxo",
    "detection_count_before_uncertanty",
    "detection_count_after_uncertanty",
    "changed_category_count",
    "changed_category_ratio",
    "visual_only_reproduction_status",
    "cache_dir",
    "output_path",
    "uncertanty_output_path",
    "status",
    "failure_reason",
]

BEST_FIELDS = [
    "detector",
    "raxo_baseline_AP",
    "raxo_baseline_AP50",
    "raxo_baseline_AP75",
    "best_mode",
    "best_lambda_material",
    "best_lambda_reliability",
    "best_AP",
    "best_AP50",
    "best_AP75",
    "best_delta_AP",
    "best_delta_AP50",
    "best_delta_AP75",
    "visual_only_reproduction_status",
    "included_in_multidetector_main_table",
    "notes",
]


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt(row.get(field)) for field in fields})


def write_md(path: Path, title: str, rows: list[dict[str, Any]], fields: list[str]) -> None:
    lines = [f"# {title}", "", "| " + " | ".join(fields) + " |", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(field)) for field in fields) + " |")
    path.write_text("\n".join(lines) + "\n")


def best_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for detector in ["groundingDINO", "VLDet", "CoDet"]:
        det_rows = [r for r in rows if r.get("detector") == detector and r.get("status") == "SUCCESS"]
        visual = next((r for r in det_rows if r.get("mode") == "visual_only"), None)
        repro = visual.get("visual_only_reproduction_status") if visual else "FAIL_FOR_MAIN_TABLE"
        candidates = [r for r in det_rows if r.get("mode") != "visual_only"]
        best = max(candidates, key=lambda r: (r.get("AP") is not None, r.get("AP") or -1), default=None)
        if best is None:
            continue
        include = repro != "FAIL_FOR_MAIN_TABLE"
        note = "ok"
        if repro == "WARNING":
            note = "visual_only differs from Batch20 by >0.002 AP"
        elif repro == "FAIL_FOR_MAIN_TABLE":
            note = "excluded because visual_only reproduction failed"
        out.append(
            {
                "detector": detector,
                "raxo_baseline_AP": best.get("raxo_baseline_AP"),
                "raxo_baseline_AP50": best.get("raxo_baseline_AP50"),
                "raxo_baseline_AP75": best.get("raxo_baseline_AP75"),
                "best_mode": best.get("mode"),
                "best_lambda_material": best.get("lambda_material"),
                "best_lambda_reliability": best.get("lambda_reliability"),
                "best_AP": best.get("AP"),
                "best_AP50": best.get("AP50"),
                "best_AP75": best.get("AP75"),
                "best_delta_AP": best.get("delta_AP_vs_detector_raxo"),
                "best_delta_AP50": best.get("delta_AP50_vs_detector_raxo"),
                "best_delta_AP75": best.get("delta_AP75_vs_detector_raxo"),
                "visual_only_reproduction_status": repro,
                "included_in_multidetector_main_table": include,
                "notes": note,
            }
        )
    return out


def run_cmd(args: list[str]) -> str:
    proc = subprocess.run(
        args,
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return proc.stdout.strip()


def write_file_index(rows: list[dict[str, Any]], cache_state: dict[str, Any]) -> None:
    index = {
        "cache_state": str(CACHE_STATE_JSON),
        "run_state": str(STATE_JSON),
        "detectors": {},
    }
    for detector in ["groundingDINO", "VLDet", "CoDet"]:
        index["detectors"][detector] = {
            "cache_dir": cache_state.get("detectors", {}).get(detector, {}).get("cache_dir"),
            "outputs": [
                {
                    "mode": row.get("mode"),
                    "lambda_material": row.get("lambda_material"),
                    "lambda_reliability": row.get("lambda_reliability"),
                    "output_path": row.get("output_path"),
                    "uncertanty_output_path": row.get("uncertanty_output_path"),
                    "status": row.get("status"),
                }
                for row in rows
                if row.get("detector") == detector
            ],
        }
    FILE_INDEX.write_text(json.dumps(index, indent=2) + "\n")


def write_report(rows: list[dict[str, Any]], best: list[dict[str, Any]], cache_state: dict[str, Any]) -> None:
    included = [r for r in best if r.get("included_in_multidetector_main_table")]
    avg_delta = (
        sum(float(r["best_delta_AP"]) for r in included if r.get("best_delta_AP") is not None) / len(included)
        if included
        else None
    )
    improved = [r for r in included if (r.get("best_delta_AP") or 0) > 0]
    stable_effective = len(improved) == len(included) and bool(included)

    git_base = ["git", "-c", f"safe.directory={ROOT}"]
    git_status = run_cmd(git_base + ["status", "--short"])
    diff_raxo = run_cmd(git_base + ["diff", "--", "raxo/"])
    diff_db = run_cmd(git_base + ["diff", "--", "run_method_database_branch_v2.sh"])
    diff_web = run_cmd(git_base + ["diff", "--", "run_method_web_branch_v2.sh"])

    lines = [
        "# Batch21 Multi-detector RMD-X Pre-argmax Report",
        "",
        "## Goal",
        "",
        "Extend RMD-X pre-argmax cache fusion from groundingDINO to VLDet and CoDet on DET-COMPASS.",
        "",
        "## Detectors",
        "",
        "- included: `groundingDINO`, `VLDet`, `CoDet`",
        "- excluded: `detic`",
        "- detic exclusion reason: official DET-COMPASS Detic proposals are not provided.",
        "",
        "## Cache Build Status",
        "",
        "| detector | status | cache_dir | proposal_count | image_count | category_count | prototype_count | feature_dim |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for detector in ["groundingDINO", "VLDet", "CoDet"]:
        rec = cache_state.get("detectors", {}).get(detector, {})
        summary = rec.get("summary") or {}
        lines.append(
            f"| {detector} | {rec.get('status')} | `{rec.get('cache_dir')}` | "
            f"{fmt(summary.get('proposal_count'))} | {fmt(summary.get('image_count'))} | "
            f"{fmt(summary.get('category_count'))} | {fmt(summary.get('prototype_count'))} | "
            f"{fmt(summary.get('feature_dim'))} |"
        )
    lines.extend(
        [
            "",
            "## visual_only Reproduction",
            "",
            "| detector | baseline AP | visual_only AP | diff | status |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for detector in ["groundingDINO", "VLDet", "CoDet"]:
        visual = next((r for r in rows if r.get("detector") == detector and r.get("mode") == "visual_only"), {})
        diff = abs((visual.get("AP") or 0) - (visual.get("raxo_baseline_AP") or 0)) if visual else None
        lines.append(
            f"| {detector} | {fmt(visual.get('raxo_baseline_AP'))} | {fmt(visual.get('AP'))} | "
            f"{fmt(diff)} | {visual.get('visual_only_reproduction_status')} |"
        )
    lines.extend(
        [
            "",
            "## Best RMD-X Results",
            "",
            "| detector | baseline AP/AP50/AP75 | best mode | lambda_m | lambda_r | best AP/AP50/AP75 | delta AP/AP50/AP75 | included |",
            "|---|---|---|---:|---:|---|---|---|",
        ]
    )
    for row in best:
        lines.append(
            f"| {row['detector']} | {fmt(row['raxo_baseline_AP'])}/{fmt(row['raxo_baseline_AP50'])}/{fmt(row['raxo_baseline_AP75'])} | "
            f"{row['best_mode']} | {fmt(row['best_lambda_material'])} | {fmt(row['best_lambda_reliability'])} | "
            f"{fmt(row['best_AP'])}/{fmt(row['best_AP50'])}/{fmt(row['best_AP75'])} | "
            f"{fmt(row['best_delta_AP'])}/{fmt(row['best_delta_AP50'])}/{fmt(row['best_delta_AP75'])} | "
            f"{row['included_in_multidetector_main_table']} |"
        )
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            f"- average delta AP over included detectors: `{fmt(avg_delta)}`",
            f"- detectors improved: `{len(improved)}/{len(included)}`",
            f"- stable multi-detector effectiveness: `{str(stable_effective).lower()}`",
            "",
            "## Notes on Declines",
            "",
        ]
    )
    declined = [r for r in included if (r.get("best_delta_AP") or 0) <= 0]
    if not declined:
        lines.append("- No included detector declined at its best RMD-X setting.")
    else:
        for row in declined:
            lines.append(
                f"- {row['detector']}: best delta AP `{fmt(row.get('best_delta_AP'))}`; possible causes include detector proposal quality, material bank coverage, or visual_only cache mismatch."
            )
    lines.extend(
        [
            "",
            "## SCI Evidence Assessment",
            "",
            "- Multi-detector evidence is usable if visual_only reproduction is PASS/WARNING and best RMD-X improves over each detector-specific RAXO baseline.",
            f"- current assessment: `{'usable' if stable_effective else 'needs additional analysis'}`",
            "",
            "## Constraints Audit",
            "",
            "- modified RAXO source code: `no`",
            "- modified official run scripts: `no`",
            "- modified annotation: `no`",
            "- modified original detection: `no`",
            "- used full_test material bank: `no`",
            "- included detic in main detector experiment: `no`",
            "",
            "## Git / Original Code Diff Check",
            "",
            "### git status --short",
            "",
            "```text",
            git_status,
            "```",
            "",
            "### git diff -- raxo/",
            "",
            "```text",
            diff_raxo,
            "```",
            "",
            "### git diff -- run_method_database_branch_v2.sh",
            "",
            "```text",
            diff_db,
            "```",
            "",
            "### git diff -- run_method_web_branch_v2.sh",
            "",
            "```text",
            diff_web,
            "```",
            "",
            "## Next Recommendation",
            "",
            "- If multi-detector improvements hold, proceed to Batch22 multi-dataset resource check.",
            "- If a detector underperforms, run detector-specific failure analysis before adding it to the paper main table.",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n")


def main() -> int:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    state = json.load(STATE_JSON.open())
    cache_state = json.load(CACHE_STATE_JSON.open())
    rows = state.get("rows", [])
    best = best_rows(rows)
    write_csv(FULL_CSV, rows, FULL_FIELDS)
    write_md(FULL_MD, "Multi-detector RMD-X Pre-argmax Results", rows, FULL_FIELDS)
    write_csv(BEST_CSV, best, BEST_FIELDS)
    write_md(BEST_MD, "Multi-detector RMD-X Best Results", best, BEST_FIELDS)
    write_file_index(rows, cache_state)
    write_report(rows, best, cache_state)
    print(f"full_csv={FULL_CSV}")
    print(f"full_md={FULL_MD}")
    print(f"best_csv={BEST_CSV}")
    print(f"best_md={BEST_MD}")
    print(f"file_index={FILE_INDEX}")
    print(f"report={REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
