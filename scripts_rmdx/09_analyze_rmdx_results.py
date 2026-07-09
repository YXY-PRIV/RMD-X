#!/usr/bin/env python3
"""Analyze RAXO baseline, support material, and RMD-X fusion results."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rmdx.analysis import (  # noqa: E402
    evaluate_per_category,
    group_by_image,
    image_level_comparison,
    load_json,
    load_result_list,
    read_best_from_grid,
    result_basic_stats,
    save_json,
    score_correlation_summary,
    validate_result_alignment,
    write_csv,
)
from rmdx.visualize import render_case_study  # noqa: E402


DEFAULT_ANN = Path("/datasets/xray-datasets/DET-COMPASS/annotations/full_test.json")
DEFAULT_IMAGE_DIR = Path("/datasets/xray-datasets/DET-COMPASS/test")
DEFAULT_BASELINE = Path(
    "/models/ICCV25_experimentation/initial_detections/groundingDINO/"
    "coco_results.bbox_DET-COMPASS_with_masks_nms_0.8_our_method_known_final_uncertanty.json"
)
DEFAULT_MATERIAL_GRID = Path(
    "/workspace/RAXO/outputs_rmdx/rmdx_results/material_colour_support_bank_grid_results.csv"
)
DEFAULT_FUSION_GRID = Path(
    "/workspace/RAXO/outputs_rmdx/rmdx_results/material_reliability_fusion_grid_results.csv"
)
DEFAULT_ANALYSIS_DIR = Path("/workspace/RAXO/outputs_rmdx/analysis")
DEFAULT_FIGURE_DIR = Path("/workspace/RAXO/outputs_rmdx/figures")
REPORT_PATH = Path("/workspace/RAXO/docs_rmdx/batch12_rmdx_analysis_report.md")

MAIN_RESULTS = {
    "baseline": {"AP": 0.480, "AP50": 0.544, "AP75": 0.488},
    "reliability_only_best": {"AP": 0.480, "AP50": 0.544, "AP75": 0.488},
    "support_material_best": {"AP": 0.490, "AP50": 0.554, "AP75": 0.499},
    "fusion_best": {"AP": 0.496, "AP50": 0.559, "AP75": 0.505},
    "full_test_material_exploratory": {"AP": 0.496, "AP50": 0.564, "AP75": 0.506},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch12 RMD-X analysis")
    parser.add_argument("--annotation", default=str(DEFAULT_ANN))
    parser.add_argument("--baseline-result", default=str(DEFAULT_BASELINE))
    parser.add_argument("--material-grid", default=str(DEFAULT_MATERIAL_GRID))
    parser.add_argument("--fusion-grid", default=str(DEFAULT_FUSION_GRID))
    parser.add_argument("--image-dir", default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("--analysis-dir", default=str(DEFAULT_ANALYSIS_DIR))
    parser.add_argument("--figure-dir", default=str(DEFAULT_FIGURE_DIR))
    return parser.parse_args()


def md_table(rows: list[dict[str, Any]], fields: list[str]) -> list[str]:
    lines = ["| " + " | ".join(fields) + " |", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return lines


def category_maps(annotation: dict[str, Any]) -> tuple[dict[int, str], dict[int, int]]:
    cat_names = {int(cat["id"]): cat.get("name", str(cat["id"])) for cat in annotation["categories"]}
    gt_counts = Counter(int(ann["category_id"]) for ann in annotation["annotations"])
    return cat_names, dict(gt_counts)


def write_per_class_outputs(
    *,
    annotation_path: Path,
    annotation: dict[str, Any],
    baseline_path: Path,
    material_path: Path,
    fusion_path: Path,
    analysis_dir: Path,
) -> list[dict[str, Any]]:
    cat_names, gt_counts = category_maps(annotation)
    baseline = evaluate_per_category(annotation_path, baseline_path)
    material = evaluate_per_category(annotation_path, material_path)
    fusion = evaluate_per_category(annotation_path, fusion_path)
    rows = []
    for category_id in sorted(cat_names):
        b = baseline.get(category_id, {})
        m = material.get(category_id, {})
        f = fusion.get(category_id, {})
        row = {
            "category_id": category_id,
            "category_name": cat_names[category_id],
            "gt_count": gt_counts.get(category_id, 0),
            "baseline_AP": b.get("AP"),
            "material_AP": m.get("AP"),
            "fusion_AP": f.get("AP"),
            "material_delta_AP": None
            if b.get("AP") is None or m.get("AP") is None
            else m["AP"] - b["AP"],
            "fusion_delta_AP": None
            if b.get("AP") is None or f.get("AP") is None
            else f["AP"] - b["AP"],
            "fusion_delta_vs_material": None
            if m.get("AP") is None or f.get("AP") is None
            else f["AP"] - m["AP"],
            "baseline_AP50": b.get("AP50"),
            "material_AP50": m.get("AP50"),
            "fusion_AP50": f.get("AP50"),
            "baseline_AP75": b.get("AP75"),
            "material_AP75": m.get("AP75"),
            "fusion_AP75": f.get("AP75"),
        }
        rows.append(row)
    write_csv(rows, analysis_dir / "per_class_ap_comparison.csv")

    def val(row: dict[str, Any], key: str) -> float:
        return float(row[key]) if row.get(key) is not None else -999.0

    top_gain = sorted(rows, key=lambda r: val(r, "fusion_delta_AP"), reverse=True)[:20]
    top_drop = sorted(rows, key=lambda r: val(r, "fusion_delta_AP"))[:20]
    top_vs_mat = sorted(rows, key=lambda r: val(r, "fusion_delta_vs_material"), reverse=True)[:20]
    mat_gain_fusion_drop = [
        row
        for row in rows
        if row.get("material_delta_AP") is not None
        and row.get("fusion_delta_vs_material") is not None
        and row["material_delta_AP"] > 0
        and row["fusion_delta_vs_material"] < 0
    ]
    fields = [
        "category_id",
        "category_name",
        "gt_count",
        "baseline_AP",
        "material_AP",
        "fusion_AP",
        "fusion_delta_AP",
        "fusion_delta_vs_material",
    ]
    md = [
        "# Per-class AP Changes",
        "",
        "说明：DET-COMPASS full_test 中大多数类别 gt_count 很少，rare 类别 AP 变化不应过度解释。",
        "",
        "## fusion 相对 baseline 提升 top 20",
        "",
        *md_table(top_gain, fields),
        "",
        "## fusion 相对 baseline 下降 top 20",
        "",
        *md_table(top_drop, fields),
        "",
        "## fusion 相对 material-only 提升 top 20",
        "",
        *md_table(top_vs_mat, fields),
        "",
        "## material-only 提升但 fusion 下降的类别",
        "",
        *md_table(mat_gain_fusion_drop[:20], fields),
    ]
    (analysis_dir / "per_class_top_changes.md").write_text("\n".join(md))
    return rows


def bin_value(value: float, bins: list[float], labels: list[str]) -> str:
    for idx, edge in enumerate(bins):
        if value < edge:
            return labels[idx]
    return labels[-1]


def write_stratified_analysis(
    *,
    annotation: dict[str, Any],
    baseline: list[dict[str, Any]],
    fusion: list[dict[str, Any]],
    per_class_rows: list[dict[str, Any]],
    image_rows: list[dict[str, Any]],
    analysis_dir: Path,
) -> None:
    gt_counts = Counter(int(ann["category_id"]) for ann in annotation["annotations"])
    strata = {"rare_gt<=1": [], "medium_gt2-5": [], "frequent_gt>5": []}
    for row in per_class_rows:
        gt = int(row["gt_count"])
        if gt <= 1:
            strata["rare_gt<=1"].append(row)
        elif gt <= 5:
            strata["medium_gt2-5"].append(row)
        else:
            strata["frequent_gt>5"].append(row)

    rows = []
    for name, items in strata.items():
        deltas = [float(r["fusion_delta_AP"]) for r in items if r.get("fusion_delta_AP") is not None]
        rows.append(
            {
                "dimension": "gt_frequency",
                "stratum": name,
                "count": len(items),
                "mean_fusion_delta_AP": sum(deltas) / len(deltas) if deltas else None,
                "positive_count": sum(1 for d in deltas if d > 0),
                "negative_count": sum(1 for d in deltas if d < 0),
            }
        )

    for label, anns in [
        ("small_area<32^2", [a for a in annotation["annotations"] if float(a.get("area", 0)) < 32**2]),
        (
            "medium_area32^2-96^2",
            [a for a in annotation["annotations"] if 32**2 <= float(a.get("area", 0)) < 96**2],
        ),
        ("large_area>=96^2", [a for a in annotation["annotations"] if float(a.get("area", 0)) >= 96**2]),
    ]:
        rows.append({"dimension": "object_area", "stratum": label, "count": len(anns)})

    image_density = []
    baseline_by_image = Counter(int(det["image_id"]) for det in baseline)
    for row in image_rows:
        density = baseline_by_image.get(int(row["image_id"]), 0)
        image_density.append((density, float(row["fusion_delta_vs_baseline"])))
    for label, values in [
        ("low_<100", [d for n, d in image_density if n < 100]),
        ("mid_100-200", [d for n, d in image_density if 100 <= n < 200]),
        ("high_>=200", [d for n, d in image_density if n >= 200]),
    ]:
        rows.append(
            {
                "dimension": "image_detection_density",
                "stratum": label,
                "count": len(values),
                "mean_fusion_delta_simple": sum(values) / len(values) if values else None,
            }
        )

    for field, bins in [
        ("score", [0.02, 0.05, 0.10, 1.01]),
        ("rmdx_material_score", [0.3, 0.5, 0.7, 1.01]),
        ("rmdx_reliability", [0.3, 0.5, 0.7, 1.01]),
    ]:
        source = fusion if field != "score" else baseline
        labels = ["very_low", "low", "mid", "high"]
        counts = Counter(
            bin_value(float(det.get(field, 0.0)), bins, labels)
            for det in source
        )
        for label in labels:
            rows.append(
                {
                    "dimension": f"{field}_bins",
                    "stratum": label,
                    "count": counts.get(label, 0),
                }
            )
    write_csv(rows, analysis_dir / "stratified_analysis.csv")
    md = [
        "# Stratified Analysis",
        "",
        "注意：full_test 中多数类别只有 1 个 GT，rare 类别上的 per-class AP 波动只能作为定位线索。",
        "",
        *md_table(rows, list(rows[0].keys())),
    ]
    (analysis_dir / "stratified_analysis.md").write_text("\n".join(md))


def write_score_summary(fusion: list[dict[str, Any]], analysis_dir: Path) -> dict[str, Any]:
    summary = score_correlation_summary(fusion)
    save_json(summary, analysis_dir / "score_correlation_summary.json")
    lines = [
        "# Score Correlation Summary",
        "",
        f"- count: `{summary['count']}`",
        f"- score range: `{summary['score_range']}`",
        f"- score_rmdx range: `{summary['score_rmdx_range']}`",
        f"- score delta range: `{summary['score_delta_range']}`",
        f"- material score range: `{summary['material_score_range']}`",
        f"- reliability score range: `{summary['reliability_score_range']}`",
        f"- pearson: `{summary['pearson']}`",
        "",
        "解释：material/reliability 与 score_delta 的相关性描述它们对最终重排的方向贡献；不代表因果证明。",
    ]
    (analysis_dir / "score_correlation_summary.md").write_text("\n".join(lines))
    return summary


def select_case_images(image_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    lowest = sorted(image_rows, key=lambda r: float(r["fusion_delta_vs_baseline"]))[:5]
    return {
        "fusion_improved": sorted(image_rows, key=lambda r: float(r["fusion_delta_vs_baseline"]), reverse=True)[:10],
        "fusion_dropped": lowest,
        "fp_suppressed": sorted(image_rows, key=lambda r: int(r["fp_reduction_vs_baseline"]), reverse=True)[:5],
        "tp_wrongly_downweighted": [
            r for r in sorted(image_rows, key=lambda r: int(r["tp_delta_vs_baseline"])) if int(r["tp_delta_vs_baseline"]) < 0
        ][:5]
        or lowest,
        "material_high_wrong": [],
        "reliability_high_wrong": [],
    }


def add_score_based_failure_cases(
    cases: dict[str, list[dict[str, Any]]],
    image_rows: list[dict[str, Any]],
    fusion: list[dict[str, Any]],
) -> None:
    row_by_image = {int(row["image_id"]): row for row in image_rows}
    by_image_scores: dict[int, dict[str, float]] = defaultdict(lambda: {"material": 0.0, "reliability": 0.0})
    for det in fusion:
        image_id = int(det["image_id"])
        by_image_scores[image_id]["material"] = max(
            by_image_scores[image_id]["material"], float(det.get("rmdx_material_score", 0.0))
        )
        by_image_scores[image_id]["reliability"] = max(
            by_image_scores[image_id]["reliability"], float(det.get("rmdx_reliability", 0.0))
        )
    bad_rows = [row for row in image_rows if float(row["fusion_delta_vs_baseline"]) < 0]
    if not bad_rows:
        bad_rows = [row for row in image_rows if float(row["fusion_delta_vs_baseline"]) <= 0]
    cases["material_high_wrong"] = sorted(
        bad_rows,
        key=lambda row: by_image_scores[int(row["image_id"])]["material"],
        reverse=True,
    )[:5]
    cases["reliability_high_wrong"] = sorted(
        bad_rows,
        key=lambda row: by_image_scores[int(row["image_id"])]["reliability"],
        reverse=True,
    )[:5]


def write_case_figures(
    *,
    annotation: dict[str, Any],
    image_dir: Path,
    baseline: list[dict[str, Any]],
    material: list[dict[str, Any]],
    fusion: list[dict[str, Any]],
    image_rows: list[dict[str, Any]],
    figure_dir: Path,
) -> dict[str, list[str]]:
    cat_names, _ = category_maps(annotation)
    images = {int(img["id"]): img for img in annotation["images"]}
    gt_by_image = group_by_image(annotation["annotations"])
    baseline_by_image = group_by_image(baseline)
    material_by_image = group_by_image(material)
    fusion_by_image = group_by_image(fusion)
    cases = select_case_images(image_rows)
    add_score_based_failure_cases(cases, image_rows, fusion)
    case_dir = figure_dir / "case_study"
    outputs: dict[str, list[str]] = {}
    for case_name, rows in cases.items():
        outputs[case_name] = []
        for idx, row in enumerate(rows, start=1):
            image_id = int(row["image_id"])
            info = images[image_id]
            out_path = case_dir / case_name / f"{idx:02d}_image_{image_id}.png"
            title = (
                f"{case_name} image_id={image_id} "
                f"dB={row['fusion_delta_vs_baseline']} "
                f"TP {row['baseline_tp']}->{row['fusion_tp']} "
                f"FP {row['baseline_fp']}->{row['fusion_fp']}"
            )
            render_case_study(
                image_path=image_dir / info["file_name"],
                output_path=out_path,
                image_id=image_id,
                title=title,
                gt_by_image=gt_by_image,
                baseline_by_image=baseline_by_image,
                material_by_image=material_by_image,
                fusion_by_image=fusion_by_image,
                cat_names=cat_names,
            )
            outputs[case_name].append(str(out_path))
    return outputs


def write_main_result_table(analysis_dir: Path) -> None:
    rows = [
        {
            "method": "RAXO baseline",
            "bank_source": "none",
            "AP": 0.480,
            "AP50": 0.544,
            "AP75": 0.488,
            "delta_AP": 0.0,
            "notes": "Recovered baseline known_final_uncertanty",
        },
        {
            "method": "Reliability-only best",
            "bank_source": "none",
            "AP": 0.480,
            "AP50": 0.544,
            "AP75": 0.488,
            "delta_AP": 0.0,
            "notes": "Batch9 margin_filter threshold=0.15",
        },
        {
            "method": "Support material-only best",
            "bank_source": "support_full",
            "AP": 0.490,
            "AP50": 0.554,
            "AP75": 0.499,
            "delta_AP": 0.010,
            "notes": "Batch10.5 material_rescore lambda=0.3",
        },
        {
            "method": "Material + reliability fusion best",
            "bank_source": "support_full",
            "AP": 0.496,
            "AP50": 0.559,
            "AP75": 0.505,
            "delta_AP": 0.016,
            "notes": "Batch11 weighted_additive lm=0.3 lr=0.2",
        },
        {
            "method": "Full-test material exploratory best",
            "bank_source": "full_test GT",
            "AP": 0.496,
            "AP50": 0.564,
            "AP75": 0.506,
            "delta_AP": 0.016,
            "notes": "Exploratory / leakage-risk; not fair main result",
        },
    ]
    fields = ["method", "bank_source", "AP", "AP50", "AP75", "delta_AP", "notes"]
    (analysis_dir / "main_result_table.md").write_text("\n".join(md_table(rows, fields)))


def write_report(
    *,
    args: argparse.Namespace,
    material_best: dict[str, str],
    fusion_best: dict[str, str],
    stats: dict[str, Any],
    per_class_rows: list[dict[str, Any]],
    score_summary: dict[str, Any],
    image_rows: list[dict[str, Any]],
    case_outputs: dict[str, list[str]],
) -> None:
    top_gain = sorted(per_class_rows, key=lambda r: float(r["fusion_delta_AP"] or -999), reverse=True)[:10]
    top_drop = sorted(per_class_rows, key=lambda r: float(r["fusion_delta_AP"] or 999))[:10]
    improved = sorted(image_rows, key=lambda r: float(r["fusion_delta_vs_baseline"]), reverse=True)[:5]
    dropped = sorted(image_rows, key=lambda r: float(r["fusion_delta_vs_baseline"]))[:5]
    fields_pc = ["category_id", "category_name", "gt_count", "baseline_AP", "material_AP", "fusion_AP", "fusion_delta_AP"]
    fields_img = ["image_id", "file_name", "gt_count", "baseline_tp", "fusion_tp", "baseline_fp", "fusion_fp", "fusion_delta_vs_baseline"]
    lines = [
        "# Batch12：RMD-X 分层、per-class 与失败案例分析报告",
        "",
        "## A. 输入文件",
        "",
        f"- annotation: `{args.annotation}`",
        f"- baseline: `{args.baseline_result}`",
        f"- support material best: `{material_best['output_path']}`",
        f"- fusion best: `{fusion_best['output_path']}`",
        f"- image_dir: `{args.image_dir}`",
        "",
        "## B. 主指标",
        "",
        "- RAXO baseline AP/AP50/AP75: `0.480 / 0.544 / 0.488`",
        "- support material-only best AP/AP50/AP75: `0.490 / 0.554 / 0.499`",
        "- material + reliability fusion best AP/AP50/AP75: `0.496 / 0.559 / 0.505`",
        "",
        "## C. Per-class AP Top Gains",
        "",
        *md_table(top_gain, fields_pc),
        "",
        "## D. Per-class AP Drops",
        "",
        *md_table(top_drop, fields_pc),
        "",
        "## E. 分层分析结论",
        "",
        "- per-class 输出：`/workspace/RAXO/outputs_rmdx/analysis/per_class_ap_comparison.csv`",
        "- 分层输出：`/workspace/RAXO/outputs_rmdx/analysis/stratified_analysis.md`",
        "- DET-COMPASS full_test 中多数类别 gt_count <= 1，per-class AP 波动只能作为定位线索，不宜过度解释。",
        "",
        "## F. material_score 分布与作用",
        "",
        f"- material score range: `{score_summary['material_score_range']}`",
        f"- material 与 score_delta Pearson: `{score_summary['pearson']['material_vs_score_delta']}`",
        "",
        "## G. reliability_score 分布与作用",
        "",
        f"- reliability range: `{score_summary['reliability_score_range']}`",
        f"- reliability 与 score_delta Pearson: `{score_summary['pearson']['reliability_vs_score_delta']}`",
        "",
        "## H. material 与 reliability 是否互补",
        "",
        f"- material 与 reliability Pearson: `{score_summary['pearson']['material_vs_reliability']}`",
        "- Batch11 weighted additive fusion 超过 support material-only，说明 reliability 在 post-rescore 排序中提供了补充信号。",
        "",
        "## I. Image-level Case Study 总结",
        "",
        "判定规则：每张图取每个方法 top-100 detections，按 category + IoU>=0.5 做 greedy TP/FP/FN 匹配。该规则用于案例定位，不等价于完整 COCO AP。若没有明确 TP 下降样本，`tp_wrongly_downweighted` / high-score wrong 图会使用无改善或最低改善样本作为 risk/neutral fallback，并在图目录名中保留原标签。",
        "",
        "### Fusion 改善 top images",
        "",
        *md_table(improved, fields_img),
        "",
        "### Fusion 下降 top images",
        "",
        *md_table(dropped, fields_img),
        "",
        "## J. 成功案例图路径",
        "",
        *[f"- `{path}`" for path in case_outputs.get("fusion_improved", [])[:10]],
        "",
        "## K. 失败案例图路径",
        "",
        *[f"- `{path}`" for key in ["fusion_dropped", "fp_suppressed", "tp_wrongly_downweighted", "material_high_wrong", "reliability_high_wrong"] for path in case_outputs.get(key, [])[:5]],
        "",
        "## L. 当前结果是否支持论文叙事",
        "",
        "支持一个谨慎版本的叙事：support-set colour-only material bank 已经带来提升，material+reliability fusion 进一步提升 AP；但目前仍是 post-rescore，不改变 bbox/category，且 colour-only 不是完整物理材料建模。",
        "",
        "## M. 主要风险",
        "",
        "- post-rescore 不改变 bbox/category。",
        "- full_test 类别样本少，per-class 分析容易高方差。",
        "- support material bank 仍需更严格 split 定义。",
        "- colour-only 不等于完整 material-aware X-ray modeling。",
        "",
        "## N. 下一步建议",
        "",
        "- Batch13：写论文实验表格和方法小节。",
        "- 或 Batch13：复制 `main_with_masks_prop.py`，在 similarity 层做更强 fusion。",
        "- 或 Batch13：导出多通道 DET-COMPASS，做 Grey/High/Low/Density material branch。",
        "",
        "## O. 是否修改 RAXO 原始代码",
        "",
        "否。未修改 `raxo/*.py`、官方 run 脚本、annotation、baseline result 或 RMD-X result。",
        "",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines))


def main() -> int:
    args = parse_args()
    analysis_dir = Path(args.analysis_dir)
    figure_dir = Path(args.figure_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    annotation_path = Path(args.annotation)
    annotation = load_json(annotation_path)
    material_best = read_best_from_grid(args.material_grid)
    fusion_best = read_best_from_grid(args.fusion_grid)
    baseline_path = Path(args.baseline_result)
    material_path = Path(material_best["output_path"])
    fusion_path = Path(fusion_best["output_path"])

    baseline = load_result_list(baseline_path)
    material = load_result_list(material_path)
    fusion = load_result_list(fusion_path)

    stats = {
        "baseline": result_basic_stats(baseline),
        "material": result_basic_stats(material),
        "fusion": result_basic_stats(fusion),
        "alignment": {
            "baseline": validate_result_alignment(annotation, baseline),
            "material": validate_result_alignment(annotation, material),
            "fusion": validate_result_alignment(annotation, fusion),
        },
        "material_best_row": material_best,
        "fusion_best_row": fusion_best,
    }
    save_json(stats, analysis_dir / "basic_stats.json")

    per_class_rows = write_per_class_outputs(
        annotation_path=annotation_path,
        annotation=annotation,
        baseline_path=baseline_path,
        material_path=material_path,
        fusion_path=fusion_path,
        analysis_dir=analysis_dir,
    )
    image_rows = image_level_comparison(annotation, baseline, material, fusion)
    write_csv(image_rows, analysis_dir / "image_level_comparison.csv")
    write_stratified_analysis(
        annotation=annotation,
        baseline=baseline,
        fusion=fusion,
        per_class_rows=per_class_rows,
        image_rows=image_rows,
        analysis_dir=analysis_dir,
    )
    score_summary = write_score_summary(fusion, analysis_dir)
    case_outputs = write_case_figures(
        annotation=annotation,
        image_dir=Path(args.image_dir),
        baseline=baseline,
        material=material,
        fusion=fusion,
        image_rows=image_rows,
        figure_dir=figure_dir,
    )
    save_json(case_outputs, analysis_dir / "case_study_outputs.json")
    write_main_result_table(analysis_dir)
    write_report(
        args=args,
        material_best=material_best,
        fusion_best=fusion_best,
        stats=stats,
        per_class_rows=per_class_rows,
        score_summary=score_summary,
        image_rows=image_rows,
        case_outputs=case_outputs,
    )
    print(json.dumps({"material_best": material_best, "fusion_best": fusion_best, "report": str(REPORT_PATH)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
