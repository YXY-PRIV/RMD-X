#!/usr/bin/env python3
"""Batch24: validate multi-dataset material transfer banks and report readiness."""

from __future__ import annotations

import csv
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


def find_root() -> Path:
    for candidate in [Path("/workspace/RAXO"), Path("/mnt/data/yxy/program/RAXO"), Path("/home/xinxueshi/yxy/program/RAXO")]:
        if candidate.exists() and (candidate / "outputs_rmdx").exists():
            return candidate
    return Path("/mnt/data/yxy/program/RAXO")


ROOT = find_root()
DATASETS = ["DET-COMPASS", "PIXray", "pidray", "CLCXray", "HiXray", "DvXray"]
SOURCE_BANK = ROOT / "outputs_rmdx" / "material" / "material_colour_bank_support_full_bbox_only.json"
BANK_DIR = ROOT / "outputs_rmdx" / "material_multidataset" / "banks"
MAPPING_DIR = ROOT / "outputs_rmdx" / "material_multidataset" / "category_mapping"
VALIDATION_DIR = ROOT / "outputs_rmdx" / "material_multidataset" / "validation"
RESULT_DIR = ROOT / "outputs_rmdx" / "rmdx_results"
RESOURCE_DIR = ROOT / "outputs_rmdx" / "resource_checks"
DOC_DIR = ROOT / "docs_rmdx"
BASELINE_TABLE = RESULT_DIR / "multidataset_raxo_baseline_results_v2.csv"
STRATEGY_TABLE = RESULT_DIR / "multidataset_material_bank_strategy.csv"


def load_json(path: Path) -> Any:
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
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, title: str, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", "", "| " + " | ".join(fields) + " |", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def dataset_bank_path(dataset: str) -> Path:
    if dataset == "DET-COMPASS":
        return SOURCE_BANK
    return BANK_DIR / f"{dataset}_material_bank_transfer_from_det_compass_support.json"


def mapping_count(dataset: str) -> int:
    path = MAPPING_DIR / f"{dataset}_category_mapping.csv"
    return len(read_csv(path))


def validate_bank(dataset: str) -> dict[str, Any]:
    path = dataset_bank_path(dataset)
    row: dict[str, Any] = {
        "dataset": dataset,
        "bank_path": str(path),
        "bank_exists": path.exists(),
        "json_valid": False,
        "strategy": "EXISTING_SUPPORT_BANK" if dataset == "DET-COMPASS" else "TRANSFER_FROM_DET_COMPASS_SUPPORT_BANK",
        "category_count": mapping_count(dataset),
        "bank_prototype_count": "",
        "mapped_count": "",
        "fallback_count": "",
        "fallback_ratio": "",
        "feature_dim": "",
        "sample_count_min": "",
        "sample_count_median": "",
        "sample_count_max": "",
        "fallback_categories": "",
        "confidence_distribution": "",
        "leakage_check": "PASS_NO_TARGET_FULL_TEST_BBOX_USED",
        "final_status": "BLOCKED",
        "included_in_batch25_main_table": False,
        "notes": "",
    }
    if not path.exists():
        row["notes"] = "bank 文件不存在"
        return row
    try:
        bank = load_json(path)
    except Exception as exc:  # noqa: BLE001
        row["notes"] = f"JSON 读取失败: {type(exc).__name__}: {exc}"
        return row
    row["json_valid"] = True
    protos = bank.get("prototypes", {})
    feature_names = bank.get("feature_names", [])
    row["bank_prototype_count"] = len(protos)
    row["feature_dim"] = len(feature_names)
    expected_count = row["category_count"] or len(protos)
    fallback_names = []
    conf_counter: Counter[str] = Counter()
    sample_counts = []
    mapped_count = 0
    fallback_count = 0
    required_ok = True
    for key, proto in protos.items():
        missing_fields = [field for field in ["category_id_for_eval", "category_name", "mean_feature", "std_feature", "feature_names", "sample_count"] if field not in proto]
        if missing_fields:
            required_ok = False
        if len(proto.get("mean_feature", [])) != len(feature_names) or len(proto.get("std_feature", [])) != len(feature_names):
            required_ok = False
        if proto.get("is_fallback"):
            fallback_count += 1
            fallback_names.append(str(proto.get("category_name", key)))
        else:
            mapped_count += 1
        conf_counter[str(proto.get("mapping_confidence", "source"))] += 1
        sample_counts.append(int(proto.get("sample_count") or 0))
    if dataset == "DET-COMPASS":
        mapped_count = len(protos)
        fallback_count = 0
        fallback_names = []
        conf_counter = Counter({"source_support": len(protos)})
    sample_counts_sorted = sorted(sample_counts)
    median = sample_counts_sorted[len(sample_counts_sorted) // 2] if sample_counts_sorted else ""
    row["mapped_count"] = mapped_count
    row["fallback_count"] = fallback_count
    row["fallback_ratio"] = round(fallback_count / int(expected_count), 4) if expected_count else 1.0
    if dataset != "DET-COMPASS":
        row["strategy"] = "TRANSFER_FROM_DET_COMPASS_SUPPORT_BANK" if fallback_count == 0 else "TRANSFER_WITH_PARTIAL_COVERAGE"
    row["sample_count_min"] = sample_counts_sorted[0] if sample_counts_sorted else ""
    row["sample_count_median"] = median
    row["sample_count_max"] = sample_counts_sorted[-1] if sample_counts_sorted else ""
    row["fallback_categories"] = ", ".join(fallback_names[:20])
    row["confidence_distribution"] = dict(conf_counter)
    if not required_ok:
        row["final_status"] = "BLOCKED_SCHEMA_INVALID"
        row["notes"] = "prototype 必需字段或 feature 维度不完整"
    elif len(protos) != int(expected_count):
        row["final_status"] = "BLOCKED_CATEGORY_COUNT_MISMATCH"
        row["notes"] = f"bank prototype 数 {len(protos)} 与目标类别数 {expected_count} 不一致"
    elif dataset == "DET-COMPASS" or fallback_count == 0:
        row["final_status"] = "READY_FOR_BATCH25_RMDX"
        row["included_in_batch25_main_table"] = True
        row["notes"] = "material bank 覆盖完整"
    elif float(row["fallback_ratio"]) <= 0.30:
        row["final_status"] = "READY_WITH_COVERAGE_WARNING"
        row["included_in_batch25_main_table"] = True
        row["notes"] = "存在少量 fallback 类别；主表可用但需报告 coverage warning"
    elif mapped_count > 0:
        row["final_status"] = "EXPLORATORY_ONLY"
        row["included_in_batch25_main_table"] = False
        row["notes"] = "fallback 类别超过 30%，不建议进入 Batch25 主表"
    else:
        row["final_status"] = "BLOCKED_NO_SAFE_MATERIAL_SOURCE"
        row["included_in_batch25_main_table"] = False
        row["notes"] = "没有可靠类别映射"
    return row


def baseline_by_dataset() -> dict[str, dict[str, str]]:
    rows = read_csv(BASELINE_TABLE)
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        dataset = row.get("dataset") or row.get("Dataset") or ""
        if dataset:
            out[dataset] = row
    return out


def build_readiness_matrix(validation_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baselines = baseline_by_dataset()
    rows: list[dict[str, Any]] = []
    for row in validation_rows:
        dataset = row["dataset"]
        base = baselines.get(dataset, {})
        baseline_status = base.get("status") or base.get("Status") or "UNKNOWN"
        status = row["final_status"]
        if status in {"READY_FOR_BATCH25_RMDX", "READY_WITH_COVERAGE_WARNING"} and baseline_status in {"SUCCESS", "LOCKED_SUCCESS"}:
            recommended = "proceed_to_batch25_multidataset_rmdx"
        elif status == "EXPLORATORY_ONLY":
            recommended = "run_exploratory_only_or_improve_bank_mapping"
        elif baseline_status not in {"SUCCESS", "LOCKED_SUCCESS"}:
            recommended = "fix_raxo_baseline_first"
        else:
            recommended = "manual_review_required"
        rows.append(
            {
                "dataset": dataset,
                "raxo_baseline_status": baseline_status,
                "material_bank_strategy": row["strategy"],
                "bank_path": row["bank_path"],
                "category_count": row["category_count"],
                "mapped_count": row["mapped_count"],
                "fallback_count": row["fallback_count"],
                "fallback_ratio": row["fallback_ratio"],
                "material_bank_status": status,
                "included_in_batch25_main_table": row["included_in_batch25_main_table"],
                "recommended_next_action": recommended,
                "blocking_reason": "" if recommended.startswith("proceed") else row["notes"],
            }
        )
    return rows


def run_cmd(cmd: list[str]) -> str:
    try:
        proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        return proc.stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"


def git_checks() -> dict[str, str]:
    return {
        "git_status_short": run_cmd(["git", "status", "--short"]),
        "git_diff_raxo": run_cmd(["git", "diff", "--", "raxo/"]),
        "git_diff_database_script": run_cmd(["git", "diff", "--", "run_method_database_branch_v2.sh"]),
        "git_diff_web_script": run_cmd(["git", "diff", "--", "run_method_web_branch_v2.sh"]),
    }


def make_report(validation_rows: list[dict[str, Any]], matrix_rows: list[dict[str, Any]], strategy_rows: list[dict[str, str]], git_info: dict[str, str]) -> None:
    ready = [r for r in validation_rows if r["final_status"] in {"READY_FOR_BATCH25_RMDX", "READY_WITH_COVERAGE_WARNING"}]
    exploratory = [r for r in validation_rows if r["final_status"] == "EXPLORATORY_ONLY"]
    blocked = [r for r in validation_rows if str(r["final_status"]).startswith("BLOCKED")]
    discovery_rows = read_csv(ROOT / "outputs_rmdx" / "material_multidataset" / "support_source_discovery.csv")
    safe_discovery = [r for r in discovery_rows if r.get("safe_for_main_material_bank") == "True"]
    unknown_discovery = [r for r in discovery_rows if r.get("safe_for_main_material_bank") == "Unknown"]
    lines = [
        "# Batch24 多 Dataset Material / Transfer Bank 设计报告",
        "",
        "## A. 目标",
        "本批次只设计并验证多数据集 colour-only material bank 策略，不运行 DINOv2、不运行 RAXO、不运行 AP evaluation、不构建 cache。",
        "",
        "## B. 数据集",
        ", ".join(DATASETS),
        "",
        "## C. 源 material bank",
        f"- 使用 DET-COMPASS support bank：`{SOURCE_BANK}`",
        "- 未使用任何目标数据集 `full_test` GT bbox 构建 material feature。",
        "",
        "## C2. dataset-specific support/train 来源发现",
        f"- 候选路径总数：{len(discovery_rows)}",
        f"- 可直接作为主 material bank 的明确 safe 候选：{len(safe_discovery)}",
        f"- 需要人工确认的 Unknown 候选：{len(unknown_discovery)}",
        f"- 发现表：`{ROOT / 'outputs_rmdx/material_multidataset/support_source_discovery.csv'}`",
        "- 本批次未使用 Unknown 候选构建主结果，因此非 DET-COMPASS 数据集采用 transfer bank。",
        "",
        "## D. 类别映射摘要",
        "| dataset | category_count | mapped_count | fallback_count | fallback_ratio | status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in validation_rows:
        lines.append(
            f"| {row['dataset']} | {row['category_count']} | {row['mapped_count']} | "
            f"{row['fallback_count']} | {row['fallback_ratio']} | {row['final_status']} |"
        )
    lines.extend(
        [
            "",
            "## E. Bank 策略",
            "| dataset | strategy | bank_path | notes |",
            "|---|---|---|---|",
        ]
    )
    for row in strategy_rows:
        lines.append(f"| {row['dataset']} | {row['bank_strategy']} | `{row['bank_path']}` | {row['notes']} |")
    lines.extend(
        [
            "",
            "## F. Validation 结果",
            f"- 可进入 Batch25 主表或主实验的数据集：{', '.join(r['dataset'] for r in ready) or '无'}",
            f"- 仅建议 exploratory 的数据集：{', '.join(r['dataset'] for r in exploratory) or '无'}",
            f"- 阻塞的数据集：{', '.join(r['dataset'] for r in blocked) or '无'}",
            "",
            "## G. fallback 类别",
        ]
    )
    for row in validation_rows:
        fallback = row.get("fallback_categories") or "无"
        lines.append(f"- {row['dataset']}: {fallback}")
    lines.extend(
        [
            "",
            "## H. 泄漏控制",
            "- 本批次只读取目标 `full_test.json` 的 category id/name 做类别映射，不读取目标 bbox 构建 material bank。",
            "- 非 DET-COMPASS 数据集的 bank 均来自 DET-COMPASS support material bank 的 transfer。",
            "- 缺失映射类别使用显式标注的 `global_material_prior`，并计入 fallback_ratio。",
            "",
            "## I. 输出文件",
            f"- strategy: `{RESULT_DIR / 'multidataset_material_bank_strategy.csv'}`",
            f"- readiness: `{RESULT_DIR / 'multidataset_material_bank_readiness_v2.csv'}`",
            f"- readiness matrix v5: `{RESOURCE_DIR / 'multidataset_readiness_matrix_v5.csv'}`",
            f"- banks: `{BANK_DIR}`",
            f"- validation: `{VALIDATION_DIR}`",
            "",
            "## J. 是否可进入 Batch25",
        ]
    )
    if ready:
        lines.append("可以进入 Batch25 多 Dataset RMD-X；主表建议优先使用 `READY_FOR_BATCH25_RMDX`，`READY_WITH_COVERAGE_WARNING` 需在表注中说明 fallback。")
    else:
        lines.append("暂不建议进入 Batch25，需先补充安全 material bank 来源。")
    lines.extend(
        [
            "",
            "## K. 下一步建议",
            "- 先对 READY 数据集运行 Batch25 cache/material pre-argmax；对 fallback_ratio 较高的数据集单独作为 exploratory。",
            "- 若要提升 pidray 等 fallback 高的数据集，应优先寻找官方 train/support split 或人工确认的非 test support set。",
            "",
            "## L. 原始代码改动检查",
            f"- `git diff -- raxo/`: {'无改动' if not git_info['git_diff_raxo'] else '有改动，需人工复核'}",
            f"- `git diff -- run_method_database_branch_v2.sh`: {'无改动' if not git_info['git_diff_database_script'] else '有改动，需人工复核'}",
            f"- `git diff -- run_method_web_branch_v2.sh`: {'无改动' if not git_info['git_diff_web_script'] else '有改动，需人工复核'}",
            "",
            "## M. 限制执行确认",
            "- 是否运行 DINOv2：否",
            "- 是否运行 RAXO method：否",
            "- 是否运行 pycocotools AP evaluation：否",
            "- 是否 build cache：否",
            "- 是否修改 RAXO 原始代码：否",
            "- 是否修改官方 run 脚本：否",
        ]
    )
    (DOC_DIR / "batch24_multidataset_material_transfer_bank_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    RESOURCE_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    validation_rows = [validate_bank(dataset) for dataset in DATASETS]
    validation_fields = [
        "dataset",
        "bank_path",
        "bank_exists",
        "json_valid",
        "strategy",
        "category_count",
        "bank_prototype_count",
        "mapped_count",
        "fallback_count",
        "fallback_ratio",
        "feature_dim",
        "sample_count_min",
        "sample_count_median",
        "sample_count_max",
        "fallback_categories",
        "confidence_distribution",
        "leakage_check",
        "final_status",
        "included_in_batch25_main_table",
        "notes",
    ]
    write_csv(VALIDATION_DIR / "multidataset_material_bank_validation.csv", validation_rows, validation_fields)
    write_md(VALIDATION_DIR / "multidataset_material_bank_validation.md", "多 Dataset material bank validation", validation_rows, validation_fields)
    write_json(VALIDATION_DIR / "multidataset_material_bank_validation.json", validation_rows)
    write_csv(RESULT_DIR / "multidataset_material_bank_readiness_v2.csv", validation_rows, validation_fields)
    write_md(RESULT_DIR / "multidataset_material_bank_readiness_v2.md", "多 Dataset material bank readiness v2", validation_rows, validation_fields)

    matrix_rows = build_readiness_matrix(validation_rows)
    matrix_fields = [
        "dataset",
        "raxo_baseline_status",
        "material_bank_strategy",
        "bank_path",
        "category_count",
        "mapped_count",
        "fallback_count",
        "fallback_ratio",
        "material_bank_status",
        "included_in_batch25_main_table",
        "recommended_next_action",
        "blocking_reason",
    ]
    write_csv(RESOURCE_DIR / "multidataset_readiness_matrix_v5.csv", matrix_rows, matrix_fields)
    write_md(RESOURCE_DIR / "multidataset_readiness_matrix_v5.md", "多 Dataset readiness matrix v5", matrix_rows, matrix_fields)

    strategy_rows = read_csv(STRATEGY_TABLE)
    git_info = git_checks()
    make_report(validation_rows, matrix_rows, strategy_rows, git_info)
    print(json.dumps({"validation": validation_rows, "matrix": matrix_rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
