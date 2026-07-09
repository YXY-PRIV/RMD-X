#!/usr/bin/env python3
"""Batch27: compare old colour-only bank with new multi-channel banks."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path("/workspace/RAXO")
if not ROOT.exists():
    ROOT = Path("/mnt/data/yxy/program/RAXO")

OUT_ROOT = ROOT / "outputs_rmdx" / "material_multichannel_det_compass"
BANK_DIR = OUT_ROOT / "banks"
FEATURE_DIR = OUT_ROOT / "features"
VALID_DIR = OUT_ROOT / "validation"
DOC_DIR = ROOT / "docs_rmdx"
OLD_COLOUR_BANK = ROOT / "outputs_rmdx" / "material" / "material_colour_bank_support_full_bbox_only.json"
REPORT = DOC_DIR / "batch27_det_compass_multichannel_material_bank_report.md"

COMPARE_FIELDS = [
    "bank_name",
    "bank_path",
    "exists",
    "category_count",
    "feature_dim",
    "sample_count_min",
    "sample_count_median",
    "sample_count_max",
    "low_sample_category_count",
    "coverage_vs_full_test_categories",
    "source_split",
    "leakage_check",
    "recommended_batch28_usage",
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


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def full_category_names() -> set[str]:
    path = dataset_root() / "DET-COMPASS" / "annotations" / "full_test.json"
    with path.open("r", encoding="utf-8") as f:
        ann = json.load(f)
    return {str(cat["name"]) for cat in ann.get("categories", [])}


def sample_stats(bank: dict[str, Any]) -> tuple[Any, Any, Any, int]:
    counts = [int(proto.get("sample_count", 0)) for proto in bank.get("prototypes", {}).values()]
    if not counts:
        return "", "", "", 0
    counts_sorted = sorted(counts)
    return min(counts), counts_sorted[len(counts_sorted) // 2], max(counts), sum(1 for c in counts if c < 3)


def bank_row(name: str, path: Path, usage: str, notes: str) -> dict[str, Any]:
    full = full_category_names()
    bank = load_json(path)
    if not bank:
        return {
            "bank_name": name,
            "bank_path": str(path),
            "exists": False,
            "category_count": 0,
            "feature_dim": "",
            "sample_count_min": "",
            "sample_count_median": "",
            "sample_count_max": "",
            "low_sample_category_count": "",
            "coverage_vs_full_test_categories": 0.0,
            "source_split": "",
            "leakage_check": "",
            "recommended_batch28_usage": "blocked",
            "notes": "文件不存在",
        }
    names = {str(proto.get("category_name")) for proto in bank.get("prototypes", {}).values() if proto.get("category_name") is not None}
    mn, med, mx, low = sample_stats(bank)
    feature_dim = len(bank.get("feature_names", []))
    return {
        "bank_name": name,
        "bank_path": str(path),
        "exists": True,
        "category_count": len(bank.get("prototypes", {})),
        "feature_dim": feature_dim,
        "sample_count_min": mn,
        "sample_count_median": med,
        "sample_count_max": mx,
        "low_sample_category_count": low,
        "coverage_vs_full_test_categories": round(len(names & full) / max(len(full), 1), 6),
        "source_split": bank.get("source_split", "legacy_support_full"),
        "leakage_check": bank.get("leakage_check", "legacy_bank_unknown_or_support_full"),
        "recommended_batch28_usage": usage,
        "notes": notes,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COMPARE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# colour-only 与 multi-channel material bank 对比",
        "",
        "说明：该表用于 Batch28 bank 选择，不包含任何 AP 结果。",
        "",
        "| bank_name | category_count | feature_dim | sample_min | sample_median | sample_max | coverage | source_split | recommended_batch28_usage | notes |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['bank_name']} | {row['category_count']} | {row['feature_dim']} | {row['sample_count_min']} | "
            f"{row['sample_count_median']} | {row['sample_count_max']} | {row['coverage_vs_full_test_categories']} | "
            f"{row['source_split']} | {row['recommended_batch28_usage']} | {row['notes']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_cmd(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    return proc.stdout.strip()


def git_checks() -> dict[str, str]:
    return {
        "git_status_short": run_cmd(["git", "status", "--short"]),
        "git_diff_raxo": run_cmd(["git", "diff", "--", "raxo/"]),
        "git_diff_database_script": run_cmd(["git", "diff", "--", "run_method_database_branch_v2.sh"]),
        "git_diff_web_script": run_cmd(["git", "diff", "--", "run_method_web_branch_v2.sh"]),
    }


def validation_rows_by_type() -> dict[str, dict[str, str]]:
    return {row["bank_type"]: row for row in read_csv(VALID_DIR / "multichannel_bank_validation.csv")}


def write_report(compare_rows: list[dict[str, Any]]) -> None:
    feature_summary = load_json(FEATURE_DIR / "support_multichannel_feature_summary.json", {})
    validation = validation_rows_by_type()
    git = git_checks()
    ready = [name for name, row in validation.items() if str(row.get("final_status", "")).startswith("READY")]
    low_counts = {
        name: int(row.get("low_sample_category_count") or 0)
        for name, row in validation.items()
    }
    lines = [
        "# Batch27 DET-COMPASS Multi-channel Support Material Bank 报告",
        "",
        "## A. Batch27 目标",
        "基于 Batch26 导出的 DET-COMPASS support/prototype 五通道图像和 parquet-derived metadata，构建 multi-channel support material bank，为 Batch28 pre-argmax fusion 做准备。",
        "",
        "## B. 输入数据来源",
        "- support/prototype 图像：`/datasets/xray-datasets/DET-COMPASS/multichannel/support_full/{colour,grey,high,low,density}/`。",
        "- metadata：Batch26 `support_multichannel_manifest.csv`，来源为 `raxo/prototypes-*.parquet` 的 `objects.categories` 与 `objects.xray_bbox`。",
        "- full_test annotation 仅用于类别名称/id 对齐，不读取 full_test bbox。",
        "",
        "## C. 为什么 support/prototype split 是安全来源",
        "- `raxo/prototypes-*.parquet` 是非 full_test split，Batch26 已确认 1227 行、307 类、五通道完整。",
        "- 本批次 feature extraction 使用 parquet-derived support/prototype bbox/category，不使用 test annotation bbox。",
        "",
        "## D. 通道列表",
        "- colour, grey, high, low, density, multi_channel_all。",
        "",
        "## E. object-level feature extraction 统计",
        f"- support_row_count：{feature_summary.get('support_row_count')}",
        f"- object_count_total：{feature_summary.get('object_count_total')}",
        f"- object_count_valid：{feature_summary.get('object_count_valid')}",
        f"- skipped_no_bbox：{feature_summary.get('skipped_no_bbox')}",
        f"- skipped_bad_bbox：{feature_summary.get('skipped_bad_bbox')}",
        f"- skipped_missing_category：{feature_summary.get('skipped_missing_category')}",
        f"- leakage_check：{feature_summary.get('leakage_check')}",
        "",
        "## F. 每个 bank 的 coverage / feature_dim / sample_count",
        "| bank_type | category_count | feature_dim | coverage_ratio | sample_min | sample_median | sample_max | final_status |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for bank_type, row in validation.items():
        lines.append(
            f"| {bank_type} | {row.get('category_count')} | {row.get('feature_dim')} | {row.get('coverage_ratio')} | "
            f"{row.get('sample_count_min')} | {row.get('sample_count_median')} | {row.get('sample_count_max')} | {row.get('final_status')} |"
        )
    lines.extend(
        [
            "",
            "## G. colour-only 旧 bank 与新 bank 对比",
            "| bank_name | category_count | feature_dim | coverage | source_split | recommended_usage |",
            "|---|---:|---:|---:|---|---|",
        ]
    )
    for row in compare_rows:
        lines.append(
            f"| {row['bank_name']} | {row['category_count']} | {row['feature_dim']} | {row['coverage_vs_full_test_categories']} | "
            f"{row['source_split']} | {row['recommended_batch28_usage']} |"
        )
    lines.extend(
        [
            "",
            "## H. all-channel bank 的新增信息",
            "- all-channel bank 组合五个通道的强度/纹理/区域特征，并加入 high-low、density-grey、colour-density 等 cross-channel 差异特征。",
            "- 其 feature_dim 高于 legacy colour-only 22D，需要 Batch28 更新 fusion 端的多通道 feature 读取逻辑。",
            "",
            "## I. 可进入 Batch28 的 bank",
            f"- READY bank：{', '.join(ready)}。",
            "- 建议 Batch28 使用 existing colour-only bank 作为历史 baseline，new colour-only bank 作为 controlled rebuild，grey/high/low/density/all-channel 作为多通道消融。",
            "",
            "## J. sample_count 低的类别",
            f"- validation 中 low_sample_category_count：{low_counts}。",
            "- 这些类别不是阻塞项，但 Batch28 结果表需要标注 support/prototype 样本数不足风险。",
            "",
            "## K. leakage-risk",
            "- 新 bank 的 leakage_check 均为 `no_full_test_bbox_used`。",
            "- 未使用 full_test GT bbox 构建 material bank。",
            "",
            "## L. 约束确认",
            f"- 是否修改 RAXO 原始代码：{'否' if not git['git_diff_raxo'] else '需人工复核'}",
            f"- 是否修改官方 run 脚本：{'否' if not git['git_diff_database_script'] and not git['git_diff_web_script'] else '需人工复核'}",
            "- 是否修改 annotation / detection / with_masks：否",
            "- 是否运行 DINOv2 / RAXO / AP evaluation / cache：否",
            "- 是否使用 full_test GT bbox：否",
            "- 是否使用 GPU：否",
            "",
            "## M. 下一步建议",
            "- 进入 Batch28：先做 multi-channel pre-argmax smoke。",
            "- Batch28 需要让 fusion 端支持非 22D material bank，并从多通道 test 目录提取 proposal material feature。",
        ]
    )
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    VALID_DIR.mkdir(parents=True, exist_ok=True)
    rows = [
        bank_row(
            "existing_colour_only_support_bank",
            OLD_COLOUR_BANK,
            "baseline_reference",
            "历史 colour-only support bank，仅作 Batch28 baseline 对照。",
        ),
        bank_row(
            "new_colour_only_support_bank",
            BANK_DIR / "material_bank_colour_support.json",
            "controlled_rebuild",
            "22D legacy-compatible colour feature，来源为 raxo/prototypes support/prototype split。",
        ),
        bank_row(
            "new_multichannel_all_support_bank",
            BANK_DIR / "material_bank_multichannel_all_support.json",
            "multichannel_ablation_primary",
            "包含五通道和 cross-channel feature，需要 Batch28 多通道 fusion 支持。",
        ),
    ]
    write_csv(VALID_DIR / "colour_only_vs_multichannel_bank_comparison.csv", rows)
    write_md(VALID_DIR / "colour_only_vs_multichannel_bank_comparison.md", rows)
    write_report(rows)
    print(json.dumps({"comparison_rows": len(rows), "report": str(REPORT)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
