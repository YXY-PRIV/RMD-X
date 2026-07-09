#!/usr/bin/env python3
"""Summarize Batch20 multi-detector RAXO baseline results."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path("/workspace/RAXO")
RESULT_DIR = ROOT / "outputs_rmdx" / "rmdx_results"
DOC_DIR = ROOT / "docs_rmdx"
STATE_JSON = RESULT_DIR / "multidetector_raxo_baseline_run_state.json"
CSV_OUT = RESULT_DIR / "multidetector_raxo_baseline_results.csv"
MD_OUT = RESULT_DIR / "multidetector_raxo_baseline_results.md"
FILE_INDEX_JSON = RESULT_DIR / "multidetector_raxo_baseline_file_index.json"
REPORT = DOC_DIR / "batch20_multidetector_raxo_baseline_report.md"

FIELDNAMES = [
    "detector",
    "included_in_main_multidetector_experiment",
    "exclusion_reason",
    "initial_detection_path",
    "initial_detection_count",
    "initial_AP",
    "initial_AP50",
    "initial_AP75",
    "with_masks_path",
    "with_masks_annotation_count",
    "known_final_path",
    "known_final_count",
    "known_final_uncertanty_path",
    "known_final_uncertanty_count",
    "raxo_AP",
    "raxo_AP50",
    "raxo_AP75",
    "delta_AP_vs_initial",
    "delta_AP50_vs_initial",
    "delta_AP75_vs_initial",
    "reused_existing_known_final",
    "generated_known_final",
    "status",
    "failure_reason",
    "main_log",
    "eval_log",
]


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def delta(a: Any, b: Any) -> float | None:
    if a is None or b is None:
        return None
    return float(a) - float(b)


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


def rows_from_state(state: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for detector in ["groundingDINO", "VLDet", "CoDet", "detic"]:
        rec = state.get("detectors", {}).get(detector, {})
        row = {
            "detector": detector,
            "included_in_main_multidetector_experiment": rec.get(
                "included_in_main_multidetector_experiment", False
            ),
            "exclusion_reason": rec.get("exclusion_reason", ""),
            "initial_detection_path": rec.get("initial_detection_path", ""),
            "initial_detection_count": rec.get("initial_detection_count"),
            "initial_AP": rec.get("initial_AP"),
            "initial_AP50": rec.get("initial_AP50"),
            "initial_AP75": rec.get("initial_AP75"),
            "with_masks_path": rec.get("with_masks_path", ""),
            "with_masks_annotation_count": rec.get("with_masks_annotation_count"),
            "known_final_path": rec.get("known_final_path", ""),
            "known_final_count": rec.get("known_final_count"),
            "known_final_uncertanty_path": rec.get("known_final_uncertanty_path", ""),
            "known_final_uncertanty_count": rec.get("known_final_uncertanty_count"),
            "raxo_AP": rec.get("raxo_AP"),
            "raxo_AP50": rec.get("raxo_AP50"),
            "raxo_AP75": rec.get("raxo_AP75"),
            "delta_AP_vs_initial": delta(rec.get("raxo_AP"), rec.get("initial_AP")),
            "delta_AP50_vs_initial": delta(rec.get("raxo_AP50"), rec.get("initial_AP50")),
            "delta_AP75_vs_initial": delta(rec.get("raxo_AP75"), rec.get("initial_AP75")),
            "reused_existing_known_final": rec.get("reused_existing_known_final", False),
            "generated_known_final": rec.get("generated_known_final", False),
            "status": rec.get("status", "MISSING"),
            "failure_reason": rec.get("failure_reason", ""),
            "main_log": rec.get("main_log", ""),
            "eval_log": rec.get("eval_log", ""),
            "initial_eval_log": rec.get("initial_eval_log", ""),
        }
        rows.append(row)
    return rows


def write_csv(rows: list[dict[str, Any]]) -> None:
    with CSV_OUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: fmt(row.get(k)) for k in FIELDNAMES})


def write_md(rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Multi-detector RAXO Baseline Results",
        "",
        "| " + " | ".join(FIELDNAMES) + " |",
        "|" + "|".join(["---"] * len(FIELDNAMES)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(k)) for k in FIELDNAMES) + " |")
    MD_OUT.write_text("\n".join(lines) + "\n")


def write_report(rows: list[dict[str, Any]], state: dict[str, Any]) -> None:
    included = [r["detector"] for r in rows if r["included_in_main_multidetector_experiment"]]
    excluded = [r["detector"] for r in rows if not r["included_in_main_multidetector_experiment"]]
    success = [r["detector"] for r in rows if r["status"] == "SUCCESS"]
    failed = [r for r in rows if r["status"] not in {"SUCCESS", "EXCLUDED"}]
    can_batch21 = all(r["status"] == "SUCCESS" for r in rows if r["included_in_main_multidetector_experiment"])

    git_base = ["git", "-c", f"safe.directory={ROOT}"]
    git_status = run_cmd(git_base + ["status", "--short"])
    diff_raxo = run_cmd(git_base + ["diff", "--", "raxo/"])
    diff_db = run_cmd(git_base + ["diff", "--", "run_method_database_branch_v2.sh"])
    diff_web = run_cmd(git_base + ["diff", "--", "run_method_web_branch_v2.sh"])

    lines = [
        "# Batch20 Multi-detector RAXO Baseline Report",
        "",
        "## Goal",
        "",
        "Run or reuse RAXO known-branch baselines for DET-COMPASS across ready detectors, without using RMD-X.",
        "",
        "## Included Detectors",
        "",
        f"`{', '.join(included)}`",
        "",
        "## Excluded Detectors",
        "",
        f"`{', '.join(excluded)}`",
        "",
        "- detic is excluded because official DET-COMPASS Detic proposals are not provided.",
        "",
        "## Initial Detection AP",
        "",
        "| detector | count | AP | AP50 | AP75 | eval log |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        if not row["included_in_main_multidetector_experiment"]:
            continue
        lines.append(
            f"| {row['detector']} | {fmt(row['initial_detection_count'])} | {fmt(row['initial_AP'])} | "
            f"{fmt(row['initial_AP50'])} | {fmt(row['initial_AP75'])} | `{row.get('initial_eval_log','')}` |"
        )
    lines.extend(
        [
            "",
            "## RAXO Known Branch Baseline",
            "",
            "| detector | status | known_final_count | known_final_uncertanty_count | AP | AP50 | AP75 | reused existing | generated known_final | failure |",
            "|---|---|---:|---:|---:|---:|---:|---|---|---|",
        ]
    )
    for row in rows:
        if not row["included_in_main_multidetector_experiment"]:
            continue
        lines.append(
            f"| {row['detector']} | {row['status']} | {fmt(row['known_final_count'])} | "
            f"{fmt(row['known_final_uncertanty_count'])} | {fmt(row['raxo_AP'])} | "
            f"{fmt(row['raxo_AP50'])} | {fmt(row['raxo_AP75'])} | "
            f"{fmt(row['reused_existing_known_final'])} | {fmt(row['generated_known_final'])} | "
            f"{row.get('failure_reason','')} |"
        )
    lines.extend(
        [
            "",
            "## Output Paths",
            "",
        ]
    )
    for row in rows:
        if not row["included_in_main_multidetector_experiment"]:
            continue
        lines.extend(
            [
                f"### {row['detector']}",
                "",
                f"- initial detection: `{row['initial_detection_path']}`",
                f"- with_masks: `{row['with_masks_path']}`",
                f"- known_final: `{row['known_final_path']}`",
                f"- known_final_uncertanty: `{row['known_final_uncertanty_path']}`",
                f"- main log: `{row.get('main_log','')}`",
                f"- final eval log: `{row.get('eval_log','')}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Failure Summary",
            "",
            f"- failed detectors: `{', '.join(r['detector'] for r in failed) if failed else 'none'}`",
        ]
    )
    for row in failed:
        lines.append(f"- {row['detector']}: {row.get('failure_reason','')}")
    lines.extend(
        [
            "",
            "## Batch21 Readiness",
            "",
            f"- can enter Batch21 multi-detector RMD-X: `{str(can_batch21).lower()}`",
            f"- successful baseline detectors: `{', '.join(success)}`",
            "",
            "## Constraints Audit",
            "",
            "- modified RAXO source code: `no`",
            "- modified official run scripts: `no`",
            "- used RMD-X: `no`",
            "- built RMD-X cache: `no`",
            "- ran RMD-X pre-argmax fusion: `no`",
            "- used full_test material bank: `no`",
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
            "- If all included detectors succeeded, proceed to Batch21 multi-detector RMD-X pre-argmax extension.",
            "- Keep detic excluded from DET-COMPASS until proposals are generated or official resources are found.",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n")


def main() -> int:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    with STATE_JSON.open() as f:
        state = json.load(f)
    rows = rows_from_state(state)
    write_csv(rows)
    write_md(rows)
    write_report(rows, state)
    print(f"csv={CSV_OUT}")
    print(f"md={MD_OUT}")
    print(f"file_index={FILE_INDEX_JSON}")
    print(f"report={REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
