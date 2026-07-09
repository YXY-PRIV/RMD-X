#!/usr/bin/env python3
"""Analyze Batch28 DET-COMPASS multi-channel pre-argmax ablation results."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


RESULT_DIR = REPO_ROOT / "outputs_rmdx/rmdx_results"
ANALYSIS_DIR = REPO_ROOT / "outputs_rmdx/analysis"
DOC_DIR = REPO_ROOT / "docs_rmdx"
SIM_DIR = REPO_ROOT / "outputs_rmdx/material_multichannel_det_compass/material_similarity"
PREARGMAX_DIR = REPO_ROOT / "outputs_rmdx/preargmax_multichannel_det_compass"
RESULT_CSV = RESULT_DIR / "det_compass_multichannel_preargmax_results.csv"
BEST_CSV = RESULT_DIR / "det_compass_multichannel_best_results.csv"
VALIDATION_CSV = REPO_ROOT / "outputs_rmdx/material_multichannel_det_compass/validation/multichannel_bank_validation.csv"
COMPARISON_CSV = (
    REPO_ROOT / "outputs_rmdx/material_multichannel_det_compass/validation/colour_only_vs_multichannel_bank_comparison.csv"
)
RAXO_BASELINE_AP = 0.480
RAXO_BASELINE = "0.480 / 0.544 / 0.488"
LEGACY_BEST_AP = 0.501
LEGACY_BEST = "0.501 / 0.561 / 0.515"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def md_table(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["无可用结果。"]
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    return lines


def safe_float(value: Any, default: float = -1.0) -> float:
    try:
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def run_text(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc.stdout.strip()


def main() -> int:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_csv(RESULT_CSV)
    best_rows = read_csv(BEST_CSV)
    validation_rows = read_csv(VALIDATION_CSV)
    comparison_rows = read_csv(COMPARISON_CSV)

    successful = [row for row in rows if row.get("status") == "SUCCESS" and row.get("AP")]
    visual = next((row for row in successful if row.get("bank_name") == "visual_only"), None)
    bank_best = {}
    for row in best_rows:
        bank_best[row["bank_name"]] = row
    global_best = max(successful, key=lambda row: safe_float(row.get("AP"))) if successful else None
    material_best = max(
        [row for row in successful if row.get("bank_name") != "visual_only"],
        key=lambda row: safe_float(row.get("AP")),
        default=None,
    )

    ablation_rows = []
    for bank_name, best in bank_best.items():
        ablation_rows.append(
            {
                "bank_name": bank_name,
                "channel_set": best.get("channel_set", ""),
                "feature_dim": best.get("feature_dim", ""),
                "best_config": f"{best.get('best_mode')} lm={best.get('best_lambda_material')} lr={best.get('best_lambda_reliability')}",
                "best_AP": best.get("best_AP", ""),
                "best_AP50": best.get("best_AP50", ""),
                "best_AP75": best.get("best_AP75", ""),
                "delta_AP_vs_raxo": best.get("delta_AP_vs_raxo", ""),
                "delta_AP_vs_legacy_colour_best": best.get("delta_AP_vs_legacy_colour_best", ""),
                "beats_legacy_colour_best": best.get("beats_legacy_colour_best", ""),
                "interpretation": interpret_bank(bank_name, safe_float(best.get("best_AP"))),
            }
        )
    write_csv(ANALYSIS_DIR / "det_compass_multichannel_bank_ablation.csv", ablation_rows)
    (ANALYSIS_DIR / "det_compass_multichannel_bank_ablation.md").write_text(
        "\n".join(
            [
                "# DET-COMPASS 多通道 material bank 消融",
                "",
                f"RAXO baseline：{RAXO_BASELINE}。",
                f"legacy colour-only 当前 best：{LEGACY_BEST}。",
                "",
                *md_table(ablation_rows),
                "",
            ]
        )
    )

    failure_notes = build_failure_notes(rows, bank_best, validation_rows)
    (ANALYSIS_DIR / "det_compass_multichannel_failure_analysis.md").write_text(failure_notes)

    git_status = run_text(["git", "status", "--short"])
    diff_raxo = run_text(["git", "diff", "--", "raxo/"])
    diff_db = run_text(["git", "diff", "--", "run_method_database_branch_v2.sh"])
    diff_web = run_text(["git", "diff", "--", "run_method_web_branch_v2.sh"])

    visual_status = "未完成"
    if visual:
        visual_diff = abs(safe_float(visual.get("AP")) - RAXO_BASELINE_AP)
        visual_status = "PASS" if visual_diff <= 0.002 else f"WARNING(diff={visual_diff:.3f})"
    best_line = "无成功 material 配置"
    if material_best:
        best_line = (
            f"{material_best['bank_name']} / {material_best['mode']} "
            f"lm={material_best['lambda_material']} lr={material_best['lambda_reliability']}："
            f"{material_best['AP']} / {material_best['AP50']} / {material_best['AP75']}"
        )
    beats_legacy = bool(material_best and safe_float(material_best.get("AP")) > LEGACY_BEST_AP)
    update_strategy = (
        "可以考虑更新论文主结果为 multi-channel 配置。"
        if beats_legacy
        else "不建议替换当前 0.501 legacy colour-only 主结果；multi-channel 更适合作为 controlled ablation / 失败分析。"
    )

    report_lines = [
        "# Batch28：DET-COMPASS Multi-channel Material Pre-argmax Fusion Ablation",
        "",
        "## A. Batch28 目标",
        "在不重跑 DINOv2、不重跑 RAXO known branch 的前提下，基于 locked visual cache 对不同 material bank 做 pre-argmax 消融，判断多通道 material 是否超过 legacy colour-only best。",
        "",
        "## B. 输入 cache 和 material bank",
        f"- locked visual cache：`{REPO_ROOT / 'outputs_rmdx/cache/det_compass_groundingDINO_100_0_full'}`",
        "- material bank：legacy colour、新建 colour、grey、high、low、density、all-channel 110D。",
        "- support/prototype bank 来源：Batch27，`leakage_check = no_full_test_bbox_used`。",
        "",
        "## C. 为什么没有重跑 DINOv2",
        "Batch28 只读取 Batch17/18 已锁定的 `visual_similarity.pt` 和 proposal metadata；新增计算只发生在 proposal crop 的 material feature 与 support bank similarity 上。",
        "",
        "## D. visual_only 是否复现 RAXO baseline",
        f"- 期望：{RAXO_BASELINE}",
        f"- 实际：{format_triplet(visual) if visual else '未完成'}",
        f"- 状态：{visual_status}",
        "",
        "## E. legacy colour bank 是否复现当前 best",
        f"- 当前锁定 legacy best：{LEGACY_BEST}",
        f"- Batch28 legacy best：{format_best(bank_best.get('legacy_colour_bank'))}",
        "",
        "## F. new colour bank 结果",
        f"- {format_best(bank_best.get('new_colour_bank'))}",
        "",
        "## G. grey/high/low/density 单通道结果",
        f"- grey：{format_best(bank_best.get('grey_bank'))}",
        f"- high：{format_best(bank_best.get('high_bank'))}",
        f"- low：{format_best(bank_best.get('low_bank'))}",
        f"- density：{format_best(bank_best.get('density_bank'))}",
        "",
        "## H. all-channel 结果",
        f"- {format_best(bank_best.get('all_channel_bank'))}",
        "",
        "## I. 最优 multi-channel 配置",
        f"- {best_line}",
        "",
        "## J. 是否超过 0.501 AP",
        f"- {'是' if beats_legacy else '否'}。",
        "",
        "## K. 是否可以更新论文主结果",
        f"- {update_strategy}",
        "",
        "## L. low sample warning 对结果的影响",
        "Batch27 已确认 91 个类别 sample_count < 3。低样本类别会放大高维 bank 的均值/方差估计噪声，尤其影响 110D all-channel 与 reliability 门控后的类间排序。",
        "",
        "## M. 如果多通道没有提升，技术解释",
        failure_notes,
        "",
        "## N. 是否修改 RAXO 原始代码",
        f"- 否。`git diff -- raxo/` 是否为空：{not bool(diff_raxo)}。",
        "",
        "## O. 是否修改官方 run 脚本",
        f"- 否。database diff empty：{not bool(diff_db)}；web diff empty：{not bool(diff_web)}。",
        "",
        "## P. 是否使用 full_test GT bbox",
        "- 否。本批次使用 detector proposal bbox 提取 inference-time material feature；bank 由 support/prototype split 构建。",
        "",
        "## Q. 下一步建议",
        "- 如果 best 超过 0.501：追加 per-class gain/loss，评估是否更新主表。",
        "- 如果 best 未超过 0.501：保留 legacy colour-only 为主结果，多通道作为 ablation，下一步优先做低样本类别鲁棒统计或通道权重学习。",
        "",
        "## 结果表",
        *md_table(ablation_rows),
        "",
        "## Git 状态摘要",
        "```text",
        git_status,
        "```",
    ]
    (DOC_DIR / "batch28_det_compass_multichannel_preargmax_report.md").write_text("\n".join(report_lines) + "\n")
    print(json.dumps({"visual_status": visual_status, "best": material_best, "beats_legacy": beats_legacy}, indent=2), flush=True)
    return 0


def interpret_bank(bank_name: str, ap: float) -> str:
    if ap < 0:
        return "无成功结果"
    if ap > LEGACY_BEST_AP:
        return "超过 legacy colour-only best，可考虑主结果更新"
    if abs(ap - LEGACY_BEST_AP) <= 0.002:
        return "接近 legacy colour-only best，适合作为 controlled ablation"
    if bank_name == "all_channel_bank":
        return "未超过 colour-only，可能受高维低样本和通道域偏移影响"
    return "未超过 legacy best，作为单通道消融证据"


def format_triplet(row: dict[str, str] | None) -> str:
    if not row:
        return "无"
    return f"{row.get('AP')} / {row.get('AP50')} / {row.get('AP75')}"


def format_best(row: dict[str, str] | None) -> str:
    if not row:
        return "无成功结果"
    return (
        f"{row.get('best_mode')} lm={row.get('best_lambda_material')} lr={row.get('best_lambda_reliability')}："
        f"{row.get('best_AP')} / {row.get('best_AP50')} / {row.get('best_AP75')} "
        f"(delta vs RAXO {row.get('delta_AP_vs_raxo')}, delta vs legacy {row.get('delta_AP_vs_legacy_colour_best')})"
    )


def build_failure_notes(
    rows: list[dict[str, str]],
    bank_best: dict[str, dict[str, str]],
    validation_rows: list[dict[str, str]],
) -> str:
    failed = [row for row in rows if row.get("status") != "SUCCESS"]
    all_channel = bank_best.get("all_channel_bank")
    all_channel_ap = safe_float(all_channel.get("best_AP") if all_channel else "")
    low_sample = next((row.get("low_sample_category_count") for row in validation_rows if row.get("bank_type") == "colour"), "91")
    lines = [
        "### 多通道未显著提升时的可能原因",
        "",
        f"- 低样本类别：Batch27 统计有 {low_sample} 个类别 sample_count < 3，支持集均值在 110D 空间中更容易受单样本噪声影响。",
        "- feature 维度：all-channel 110D 比 22D colour bank 高得多，但每类样本中位数只有 3，容易出现维度高于有效样本的估计不稳定。",
        "- 标准化：当前使用 bank feature stats 做 z-score 后 cosine，相比可学习通道权重仍较粗糙。",
        "- 通道顺序假设：Batch26 使用 xray_images[0..4] 的工作假设；若 HF 通道语义与 colour/grey/high/low/density 不完全一致，会削弱信号。",
        "- material 与 visual prototype 冗余：RAXO visual similarity 已包含部分外观/材质信息，简单 additive fusion 的边际收益可能有限。",
        "- cross-channel 特征较浅：当前 cross feature 是统计量差值/比值，没有学习类别条件下的通道组合。",
    ]
    if all_channel and all_channel_ap > LEGACY_BEST_AP:
        lines.append("- 但本轮 all-channel 已超过 0.501，说明上述风险没有阻止整体提升，后续应做 per-class 分析确认收益来源。")
    if failed:
        lines.extend(["", "### 失败配置摘要"])
        for row in failed[:20]:
            lines.append(f"- {row.get('bank_name')} {row.get('mode')}：{row.get('failure_reason')}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
