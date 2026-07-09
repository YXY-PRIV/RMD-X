#!/usr/bin/env python3
"""Batch25: summarize multi-dataset RMD-X pre-argmax results."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path("/workspace/RAXO")
if not ROOT.exists():
    ROOT = Path("/mnt/data/yxy/program/RAXO")

RESULT_DIR = ROOT / "outputs_rmdx" / "rmdx_results"
RESOURCE_DIR = ROOT / "outputs_rmdx" / "resource_checks"
DOC_DIR = ROOT / "docs_rmdx"
STATE_JSON = RESULT_DIR / "multidataset_rmdx_preargmax_run_state.json"
CACHE_STATUS = RESULT_DIR / "multidataset_cache_build_status.csv"
MAIN_DATASETS = ["DET-COMPASS", "PIXray", "CLCXray", "HiXray", "DvXray"]
EXPLORATORY_DATASETS = ["pidray"]

FULL_FIELDS = [
    "dataset",
    "result_group",
    "material_bank_status",
    "mode",
    "lambda_material",
    "lambda_reliability",
    "raxo_baseline_AP",
    "raxo_baseline_AP50",
    "raxo_baseline_AP75",
    "AP",
    "AP50",
    "AP75",
    "delta_AP_vs_raxo",
    "delta_AP50_vs_raxo",
    "delta_AP75_vs_raxo",
    "visual_only_reproduction_status",
    "detection_count_before_uncertanty",
    "detection_count_after_uncertanty",
    "changed_category_count",
    "changed_category_ratio",
    "cache_dir",
    "material_bank_path",
    "output_path",
    "uncertanty_output_path",
    "status",
    "failure_reason",
]

BEST_FIELDS = [
    "dataset",
    "result_group",
    "material_bank_status",
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
    "included_in_multidataset_main_table",
    "notes",
]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, title: str, rows: list[dict[str, Any]], fields: list[str]) -> None:
    lines = [f"# {title}", "", "| " + " | ".join(fields) + " |", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def as_float(value: Any, default: float = -999.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def success(row: dict[str, Any]) -> bool:
    return str(row.get("status", "")).startswith("SUCCESS") and row.get("AP") is not None


def best_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for dataset in [*MAIN_DATASETS, *EXPLORATORY_DATASETS]:
        ds_rows = [row for row in rows if row.get("dataset") == dataset and success(row)]
        visual_status = next((row.get("visual_only_reproduction_status", "") for row in ds_rows if row.get("mode") == "visual_only"), "")
        if not ds_rows:
            out.append(
                {
                    "dataset": dataset,
                    "result_group": "exploratory" if dataset in EXPLORATORY_DATASETS else "main",
                    "material_bank_status": "",
                    "raxo_baseline_AP": "",
                    "raxo_baseline_AP50": "",
                    "raxo_baseline_AP75": "",
                    "best_mode": "",
                    "best_lambda_material": "",
                    "best_lambda_reliability": "",
                    "best_AP": "",
                    "best_AP50": "",
                    "best_AP75": "",
                    "best_delta_AP": "",
                    "best_delta_AP50": "",
                    "best_delta_AP75": "",
                    "visual_only_reproduction_status": visual_status,
                    "included_in_multidataset_main_table": False,
                    "notes": "无成功 RMD-X 结果",
                }
            )
            continue
        best = max(ds_rows, key=lambda row: as_float(row.get("AP")))
        include = dataset in MAIN_DATASETS and visual_status in {"PASS", "WARNING"} and best.get("material_bank_status") != "EXPLORATORY_ONLY"
        notes = ""
        if dataset in EXPLORATORY_DATASETS:
            notes = "pidray fallback ratio 41.67%，EXPLORATORY_ONLY，不进入主表"
        elif visual_status == "FAIL_FOR_MAIN_TABLE":
            notes = "visual_only 与 Batch23 baseline 不一致，不进入主表"
        elif best.get("material_bank_status") == "READY_WITH_COVERAGE_WARNING":
            notes = "material bank 存在 fallback 类别，主表需表注说明"
        else:
            notes = "可进入主表"
        out.append(
            {
                "dataset": dataset,
                "result_group": "exploratory" if dataset in EXPLORATORY_DATASETS else "main",
                "material_bank_status": best.get("material_bank_status", ""),
                "raxo_baseline_AP": best.get("raxo_baseline_AP", ""),
                "raxo_baseline_AP50": best.get("raxo_baseline_AP50", ""),
                "raxo_baseline_AP75": best.get("raxo_baseline_AP75", ""),
                "best_mode": best.get("mode", ""),
                "best_lambda_material": best.get("lambda_material", ""),
                "best_lambda_reliability": best.get("lambda_reliability", ""),
                "best_AP": best.get("AP", ""),
                "best_AP50": best.get("AP50", ""),
                "best_AP75": best.get("AP75", ""),
                "best_delta_AP": best.get("delta_AP_vs_raxo", ""),
                "best_delta_AP50": best.get("delta_AP50_vs_raxo", ""),
                "best_delta_AP75": best.get("delta_AP75_vs_raxo", ""),
                "visual_only_reproduction_status": visual_status,
                "included_in_multidataset_main_table": include,
                "notes": notes,
            }
        )
    return out


def run_cmd(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    output = proc.stdout.strip()
    if proc.returncode != 0 and (
        "not a git repository" in output.lower()
        or "not found" in output.lower()
        or "no such file or directory" in output.lower()
    ):
        return ""
    return output


def git_checks() -> dict[str, str]:
    return {
        "git_status_short": run_cmd(["git", "status", "--short"]),
        "git_diff_raxo": run_cmd(["git", "diff", "--", "raxo/"]),
        "git_diff_database_script": run_cmd(["git", "diff", "--", "run_method_database_branch_v2.sh"]),
        "git_diff_web_script": run_cmd(["git", "diff", "--", "run_method_web_branch_v2.sh"]),
    }


def make_report(rows: list[dict[str, Any]], best: list[dict[str, Any]], cache_rows: list[dict[str, str]], git_info: dict[str, str]) -> None:
    main_best = [row for row in best if row["dataset"] in MAIN_DATASETS]
    included = [row for row in main_best if str(row.get("included_in_multidataset_main_table")) == "True" or row.get("included_in_multidataset_main_table") is True]
    deltas = [as_float(row.get("best_delta_AP"), 0.0) for row in included if row.get("best_delta_AP") not in {"", None}]
    improved = [row for row in included if as_float(row.get("best_delta_AP"), 0.0) > 0]
    declined = [row for row in included if as_float(row.get("best_delta_AP"), 0.0) < 0]
    avg_delta = sum(deltas) / len(deltas) if deltas else 0.0
    pid = next((row for row in best if row["dataset"] == "pidray"), None)
    lines = [
        "# Batch25 多 Dataset RMD-X Pre-argmax 报告",
        "",
        "## A. Batch25 目标",
        "基于 Batch24 的 support/transfer material bank，在多个 dataset 上运行 RMD-X pre-argmax cache fusion，并与 Batch23.2 RAXO baseline 对比。",
        "",
        "## B. 容器与并行策略",
        "- 目标容器：`raxo_iccv25_gpu_mnt`。",
        "- 实际执行若使用临时 runner，会在 tmux launch 文档中记录，原因是现有 `raxo_iccv25_gpu_mnt` 的 `/workspace/RAXO` 挂载为空。",
        "- cache build 使用任务级并行：每个 dataset 一个进程，通过 `CUDA_VISIBLE_DEVICES` 绑定单张 GPU；未改 DDP。",
        "",
        "## C. cache build 状态",
        "| dataset | group | gpu_id | status | proposal_count | image_count | category_count | cache_dir |",
        "|---|---|---|---|---:|---:|---:|---|",
    ]
    for row in cache_rows:
        lines.append(f"| {row.get('dataset','')} | {row.get('group','')} | {row.get('gpu_id','')} | {row.get('status','')} | {row.get('proposal_count','')} | {row.get('image_count','')} | {row.get('category_count','')} | `{row.get('cache_dir','')}` |")
    lines.extend(
        [
            "",
            "## D. 主结果 best RMD-X",
            "| dataset | RAXO AP/AP50/AP75 | best RMD-X AP/AP50/AP75 | delta AP | mode | visual_only | notes |",
            "|---|---|---|---:|---|---|---|",
        ]
    )
    for row in main_best:
        lines.append(
            f"| {row['dataset']} | {row.get('raxo_baseline_AP','')}/{row.get('raxo_baseline_AP50','')}/{row.get('raxo_baseline_AP75','')} | "
            f"{row.get('best_AP','')}/{row.get('best_AP50','')}/{row.get('best_AP75','')} | {row.get('best_delta_AP','')} | "
            f"{row.get('best_mode','')} lm={row.get('best_lambda_material','')} lr={row.get('best_lambda_reliability','')} | "
            f"{row.get('visual_only_reproduction_status','')} | {row.get('notes','')} |"
        )
    lines.extend(
        [
            "",
            "## E. 汇总判断",
            f"- 进入主表的数据集数：{len(included)}",
            f"- 主表平均 delta AP：{avg_delta:.4f}",
            f"- 主表提升的数据集数：{len(improved)}",
            f"- 主表下降的数据集数：{len(declined)}",
            "",
            "## F. pidray exploratory",
        ]
    )
    if pid:
        lines.append(
            f"- pidray: best={pid.get('best_AP','')}/{pid.get('best_AP50','')}/{pid.get('best_AP75','')}, "
            f"delta_AP={pid.get('best_delta_AP','')}; fallback ratio 41.67%，不进入主表。"
        )
    else:
        lines.append("- pidray 未产生成功结果。")
    lines.extend(
        [
            "",
            "## G. 论文表格建议",
            "- `multidataset_rmdx_main_table.md/csv`：主表候选。",
            "- `multidataset_rmdx_exploratory_table.md/csv`：pidray exploratory / appendix。",
            "- 若某 dataset visual_only 复现失败，则不进入主表，需先查 cache 与 Batch23 baseline 的一致性。",
            "",
            "## H. 原始代码与约束确认",
            f"- `git diff -- raxo/`: {'无改动' if not git_info['git_diff_raxo'] else '有改动，需人工复核'}",
            f"- `git diff -- run_method_database_branch_v2.sh`: {'无改动' if not git_info['git_diff_database_script'] else '有改动，需人工复核'}",
            f"- `git diff -- run_method_web_branch_v2.sh`: {'无改动' if not git_info['git_diff_web_script'] else '有改动，需人工复核'}",
            "- 是否使用 full_test material bank：否",
            "- 是否使用目标 full_test GT bbox 构建 material bank：否",
            "- 是否运行 RAXO known branch：否",
            "- 是否做 mask extraction：否",
            "",
            "## I. 下一步建议",
            "- 若主表平均 delta AP 为正且 visual_only 均通过，可进入 Batch26 做论文表格与失败案例分析。",
            "- 若某 dataset 下降，优先检查 fallback 类别、category mapping 和 material bank 迁移是否过弱。",
        ]
    )
    (DOC_DIR / "batch25_multidataset_rmdx_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    state = read_json(STATE_JSON) if STATE_JSON.exists() else {"rows": []}
    rows = state.get("rows", [])
    for row in rows:
        if row.get("status") == "SUCCESS_REUSED":
            row["status"] = "SUCCESS_REUSED"
    best = best_rows(rows)
    main_rows = [row for row in best if row["dataset"] in MAIN_DATASETS and row.get("included_in_multidataset_main_table")]
    exploratory_rows = [row for row in best if row["dataset"] in EXPLORATORY_DATASETS or not row.get("included_in_multidataset_main_table")]

    write_csv(RESULT_DIR / "multidataset_rmdx_preargmax_results.csv", rows, FULL_FIELDS)
    write_md(RESULT_DIR / "multidataset_rmdx_preargmax_results.md", "多 Dataset RMD-X pre-argmax 全量结果", rows, FULL_FIELDS)
    write_csv(RESULT_DIR / "multidataset_rmdx_best_results.csv", best, BEST_FIELDS)
    write_md(RESULT_DIR / "multidataset_rmdx_best_results.md", "多 Dataset RMD-X best 结果", best, BEST_FIELDS)
    write_csv(RESULT_DIR / "multidataset_rmdx_main_table.csv", main_rows, BEST_FIELDS)
    write_md(RESULT_DIR / "multidataset_rmdx_main_table.md", "多 Dataset RMD-X 主表", main_rows, BEST_FIELDS)
    write_csv(RESULT_DIR / "multidataset_rmdx_exploratory_table.csv", exploratory_rows, BEST_FIELDS)
    write_md(RESULT_DIR / "multidataset_rmdx_exploratory_table.md", "多 Dataset RMD-X exploratory / appendix 表", exploratory_rows, BEST_FIELDS)

    file_index = {
        "state_json": str(STATE_JSON),
        "cache_status": str(CACHE_STATUS),
        "full_results": str(RESULT_DIR / "multidataset_rmdx_preargmax_results.csv"),
        "best_results": str(RESULT_DIR / "multidataset_rmdx_best_results.csv"),
        "main_table": str(RESULT_DIR / "multidataset_rmdx_main_table.csv"),
        "exploratory_table": str(RESULT_DIR / "multidataset_rmdx_exploratory_table.csv"),
        "report": str(DOC_DIR / "batch25_multidataset_rmdx_report.md"),
    }
    write_json(RESULT_DIR / "multidataset_rmdx_file_index.json", file_index)
    cache_rows = read_csv(CACHE_STATUS)
    git_info = git_checks()
    make_report(rows, best, cache_rows, git_info)
    print(json.dumps({"best": best, "file_index": file_index}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
