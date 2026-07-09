#!/usr/bin/env python3
"""Summarize Batch23 multi-dataset RAXO baseline results."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path("/workspace/RAXO")
RESULT_DIR = ROOT / "outputs_rmdx" / "rmdx_results"
DOC_DIR = ROOT / "docs_rmdx"
STATE_JSON = RESULT_DIR / "multidataset_raxo_baseline_run_state.json"
CSV_OUT = RESULT_DIR / "multidataset_raxo_baseline_results.csv"
MD_OUT = RESULT_DIR / "multidataset_raxo_baseline_results.md"
FILE_INDEX_JSON = RESULT_DIR / "multidataset_raxo_baseline_file_index.json"
FAILURES_JSON = RESULT_DIR / "multidataset_raxo_baseline_failures.json"
REPORT = DOC_DIR / "batch23_multidataset_raxo_baseline_report.md"

DATASETS = ["DET-COMPASS", "PIXray", "pidray", "CLCXray", "HiXray", "DvXray"]

FIELDNAMES = [
    "dataset",
    "image_count",
    "category_count",
    "annotation_count",
    "initial_detection_path_used",
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
    "generated_uncertanty",
    "status",
    "failure_reason",
    "included_in_multidataset_main_table",
    "main_log",
    "eval_log",
]


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


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
    for dataset in DATASETS:
        rec = state.get("datasets", {}).get(dataset, {})
        row = {key: rec.get(key) for key in FIELDNAMES}
        row["dataset"] = dataset
        row["initial_eval_log"] = rec.get("initial_eval_log", "")
        row["included_in_multidataset_main_table"] = rec.get("status") == "SUCCESS"
        rows.append(row)
    return rows


def log_error_summary(path_value: Any) -> str:
    if not path_value:
        return ""
    first_path = str(path_value).split(";")[0].strip()
    path = Path(first_path)
    if not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        return f"could not read log: {type(exc).__name__}: {exc}"
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if "JSONDecodeError" in line:
            return line.strip()
        if "Traceback" in line:
            tail = lines[idx : idx + 12]
            for item in tail:
                if "Error" in item or "Exception" in item:
                    return item.strip()
    for line in reversed(lines[-40:]):
        if line.strip():
            return line.strip()
    return ""


def write_csv(rows: list[dict[str, Any]]) -> None:
    with CSV_OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(row.get(key)) for key in FIELDNAMES})


def write_md(rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Multi-dataset RAXO Baseline Results",
        "",
        "| " + " | ".join(FIELDNAMES) + " |",
        "|" + "|".join(["---"] * len(FIELDNAMES)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(key)) for key in FIELDNAMES) + " |")
    MD_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_failures(rows: list[dict[str, Any]]) -> None:
    failures = [
        {
            "dataset": row["dataset"],
            "status": row.get("status"),
            "failure_reason": row.get("failure_reason", ""),
            "error_summary": log_error_summary(row.get("main_log", "")),
            "main_log": row.get("main_log", ""),
            "eval_log": row.get("eval_log", ""),
        }
        for row in rows
        if row.get("status") != "SUCCESS"
    ]
    FAILURES_JSON.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")


def write_report(rows: list[dict[str, Any]], state: dict[str, Any]) -> None:
    success = [row for row in rows if row.get("status") == "SUCCESS"]
    failed = [row for row in rows if row.get("status") != "SUCCESS"]
    reused = [row["dataset"] for row in rows if row.get("reused_existing_known_final")]
    generated = [row["dataset"] for row in rows if row.get("generated_known_final")]
    can_batch24 = len(success) >= 1
    can_batch25 = len(success) == len(DATASETS)

    git_base = ["git", "-c", f"safe.directory={ROOT}"]
    git_status = run_cmd(git_base + ["status", "--short"])
    diff_raxo = run_cmd(git_base + ["diff", "--", "raxo/"])
    diff_db = run_cmd(git_base + ["diff", "--", "run_method_database_branch_v2.sh"])
    diff_web = run_cmd(git_base + ["diff", "--", "run_method_web_branch_v2.sh"])

    lines = [
        "# Batch23 Multi-dataset RAXO Baseline Report",
        "",
        "## Goal",
        "",
        "Run or reuse groundingDINO RAXO known/database branch baselines across six datasets.",
        "",
        "## Runtime Context",
        "",
        f"- container: `{state.get('container', 'raxo_iccv25_gpu_mnt')}`",
        f"- detector: `{state.get('detector', 'groundingDINO')}`",
        f"- dataset mount: `{state.get('dataset_root', '/datasets/xray-datasets')}`",
        f"- model root: `/models/ICCV25_experimentation`",
        "",
        "## Test Image Readiness",
        "",
        "| dataset | image_count | annotation_count | category_count | test_image_status |",
        "|---|---:|---:|---:|---|",
    ]
    for row in rows:
        pre = state.get("preflight", {}).get("datasets", {}).get(row["dataset"], {}).get("annotation", {})
        lines.append(
            f"| {row['dataset']} | {fmt(pre.get('image_count', row.get('image_count')))} | "
            f"{fmt(pre.get('annotation_count', row.get('annotation_count')))} | "
            f"{fmt(pre.get('category_count', row.get('category_count')))} | "
            f"{pre.get('test_image_status', '')} |"
        )

    lines.extend(
        [
            "",
            "## Initial Detection AP",
            "",
            "| dataset | count | AP | AP50 | AP75 | detection path | eval log |",
            "|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {fmt(row.get('initial_detection_count'))} | {fmt(row.get('initial_AP'))} | "
            f"{fmt(row.get('initial_AP50'))} | {fmt(row.get('initial_AP75'))} | "
            f"`{row.get('initial_detection_path_used', '')}` | `{row.get('initial_eval_log', '')}` |"
        )

    lines.extend(
        [
            "",
            "## RAXO Known Branch Baseline",
            "",
            "| dataset | status | known_final_count | known_final_uncertanty_count | AP | AP50 | AP75 | delta AP | reused existing | generated known_final | failure |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row.get('status', '')} | {fmt(row.get('known_final_count'))} | "
            f"{fmt(row.get('known_final_uncertanty_count'))} | {fmt(row.get('raxo_AP'))} | "
            f"{fmt(row.get('raxo_AP50'))} | {fmt(row.get('raxo_AP75'))} | "
            f"{fmt(row.get('delta_AP_vs_initial'))} | {fmt(row.get('reused_existing_known_final'))} | "
            f"{fmt(row.get('generated_known_final'))} | {row.get('failure_reason', '')} |"
        )

    lines.extend(
        [
            "",
            "## Output Paths",
            "",
        ]
    )
    for row in rows:
        lines.extend(
            [
                f"### {row['dataset']}",
                "",
                f"- with_masks: `{row.get('with_masks_path', '')}`",
                f"- known_final: `{row.get('known_final_path', '')}`",
                f"- known_final_uncertanty: `{row.get('known_final_uncertanty_path', '')}`",
                f"- main log: `{row.get('main_log', '')}`",
                f"- final eval log: `{row.get('eval_log', '')}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Reuse / Generation Summary",
            "",
            f"- reused existing known_final or known_final_uncertanty: `{', '.join(reused) if reused else 'none'}`",
            f"- newly generated known_final: `{', '.join(generated) if generated else 'none'}`",
            "",
            "## Failures",
            "",
            f"- failed datasets: `{', '.join(row['dataset'] for row in failed) if failed else 'none'}`",
        ]
    )
    for row in failed:
        detail = log_error_summary(row.get("main_log", ""))
        suffix = f" ({detail})" if detail else ""
        lines.append(f"- {row['dataset']}: {row.get('failure_reason', '')}{suffix}")

    lines.extend(
        [
            "",
            "## Next Step Readiness",
            "",
            f"- can enter Batch24 material bank design: `{str(can_batch24).lower()}`",
            f"- can enter Batch25 multi-dataset RMD-X for all six datasets: `{str(can_batch25).lower()}`",
            "",
            "## Constraints Audit",
            "",
            "- modified RAXO source code: `no`",
            "- modified official run scripts: `no`",
            "- modified annotation/detection/with_masks: `no`",
            "- used RMD-X: `no`",
            "- built RMD-X cache: `no`",
            "- ran RMD-X pre-argmax fusion: `no`",
            "- used full_test material bank: `no`",
            "- ran mask extraction: `no`",
            "",
            "## Generated Result Files",
            "",
            f"- CSV: `{CSV_OUT}`",
            f"- Markdown table: `{MD_OUT}`",
            f"- file index: `{FILE_INDEX_JSON}`",
            f"- failures: `{FAILURES_JSON}`",
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
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    if not STATE_JSON.exists():
        raise SystemExit(f"missing state json: {STATE_JSON}")
    state = json.loads(STATE_JSON.read_text(encoding="utf-8"))
    rows = rows_from_state(state)
    write_csv(rows)
    write_md(rows)
    write_failures(rows)
    write_report(rows, state)
    print(f"wrote {CSV_OUT}")
    print(f"wrote {MD_OUT}")
    print(f"wrote {REPORT}")


if __name__ == "__main__":
    main()
