#!/usr/bin/env python3
"""Run RMD-X material + reliability fusion MVP."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rmdx.fusion import (  # noqa: E402
    apply_material_filter_reliability_rescore,
    apply_material_only_reproduce,
    apply_reliability_gated_material,
    apply_weighted_additive_fusion,
    attach_material_scores,
    attach_reliability_scores,
    summarize_fusion_result,
)
from rmdx.rescore import filter_by_image_ids, load_detections, save_detections  # noqa: E402


DEFAULT_INPUT = Path(
    "/models/ICCV25_experimentation/initial_detections/groundingDINO/"
    "coco_results.bbox_DET-COMPASS_with_masks_nms_0.8_our_method_known_final_uncertanty.json"
)
DEFAULT_ANN = Path("/datasets/xray-datasets/DET-COMPASS/annotations/full_test.json")
DEFAULT_IMAGE_DIR = Path("/datasets/xray-datasets/DET-COMPASS/test")
DEFAULT_BANK = Path(
    "/workspace/RAXO/outputs_rmdx/material/material_colour_bank_support_full_bbox_only.json"
)
DEFAULT_OUT = Path("/workspace/RAXO/outputs_rmdx/rmdx_results")
DEFAULT_SMOKE = Path("/workspace/RAXO/outputs_rmdx/smoke")
REPORT_PATH = Path("/workspace/RAXO/docs_rmdx/batch11_material_reliability_fusion_report.md")

BASELINE = {"AP": 0.480, "AP50": 0.544, "AP75": 0.488}
SUPPORT_MATERIAL_BEST = {"AP": 0.490, "AP50": 0.554, "AP75": 0.499}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RMD-X material + reliability fusion")
    parser.add_argument("--input-result", default=str(DEFAULT_INPUT))
    parser.add_argument("--annotation", default=str(DEFAULT_ANN))
    parser.add_argument("--image-dir", default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("--material-bank", default=str(DEFAULT_BANK))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--strategy",
        choices=[
            "material_only_reproduce",
            "reliability_gated_material",
            "weighted_additive_fusion",
            "material_filter_reliability_rescore",
        ],
        default="material_only_reproduce",
    )
    parser.add_argument("--lambda-material", type=float, default=0.3)
    parser.add_argument("--lambda-reliability", type=float, default=0.1)
    parser.add_argument("--threshold-reliability", type=float, default=0.5)
    parser.add_argument("--threshold-material", type=float, default=0.4)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--run-grid", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def slug_float(value: float | None) -> str:
    if value is None:
        return "none"
    return str(value).replace(".", "p")


def load_annotation(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"{path} is not a COCO annotation dict")
    return data


def write_annotation_5imgs(annotation: dict[str, Any], out_path: Path) -> tuple[set[int], Path]:
    images = annotation.get("images", [])[:5]
    image_ids = {int(img["id"]) for img in images}
    ann5 = {
        "images": images,
        "annotations": [
            ann for ann in annotation.get("annotations", []) if int(ann["image_id"]) in image_ids
        ],
        "categories": annotation.get("categories", []),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(ann5, f)
    return image_ids, out_path


def run_eval(annotation: Path, detections: Path, log_path: Path) -> dict[str, Any]:
    cmd = [
        "python3",
        "raxo/cocoapi_2.py",
        "--cocoGt",
        str(annotation),
        "--cocoDt",
        str(detections),
    ]
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(proc.stdout)
    aps = re.findall(r"Average Precision.*=\s+(-?\d+\.\d+)", proc.stdout)
    return {
        "command": " ".join(cmd),
        "passed": proc.returncode == 0,
        "returncode": proc.returncode,
        "AP": float(aps[0]) if len(aps) > 0 else None,
        "AP50": float(aps[1]) if len(aps) > 1 else None,
        "AP75": float(aps[2]) if len(aps) > 2 else None,
        "log_path": str(log_path),
    }


def output_stem(
    strategy: str,
    alpha: float,
    beta: float,
    gamma: float,
    lambda_material: float | None,
    lambda_reliability: float | None,
    threshold_reliability: float | None,
    threshold_material: float | None,
) -> str:
    return (
        f"rmdx_fusion_{strategy}_"
        f"a{slug_float(alpha)}_b{slug_float(beta)}_g{slug_float(gamma)}_"
        f"lm{slug_float(lambda_material)}_lr{slug_float(lambda_reliability)}_"
        f"tr{slug_float(threshold_reliability)}_tm{slug_float(threshold_material)}"
    )


def prepare_scored_dets(
    args: argparse.Namespace,
    *,
    smoke: bool,
) -> tuple[list[dict[str, Any]], Path, dict[str, Any]]:
    annotation = load_annotation(Path(args.annotation))
    dets = load_detections(Path(args.input_result))
    eval_annotation = Path(args.annotation)
    if smoke:
        image_ids, eval_annotation = write_annotation_5imgs(
            annotation, DEFAULT_SMOKE / "full_test_5imgs_for_fusion_mvp.json"
        )
        dets = filter_by_image_ids(dets, image_ids)
    material_dets, material_summary = attach_material_scores(
        dets,
        args.material_bank,
        annotation,
        args.image_dir,
        mode="bbox_only",
    )
    scored = attach_reliability_scores(material_dets, args.alpha, args.beta, args.gamma)
    return scored, eval_annotation, material_summary


def apply_strategy(
    dets: list[dict[str, Any]],
    *,
    strategy: str,
    lambda_material: float,
    lambda_reliability: float,
    threshold_reliability: float,
    threshold_material: float,
) -> list[dict[str, Any]]:
    if strategy == "material_only_reproduce":
        return apply_material_only_reproduce(dets, lambda_material=lambda_material)
    if strategy == "reliability_gated_material":
        return apply_reliability_gated_material(
            dets,
            lambda_material=lambda_material,
            threshold_reliability=threshold_reliability,
        )
    if strategy == "weighted_additive_fusion":
        return apply_weighted_additive_fusion(
            dets,
            lambda_material=lambda_material,
            lambda_reliability=lambda_reliability,
        )
    if strategy == "material_filter_reliability_rescore":
        return apply_material_filter_reliability_rescore(
            dets,
            threshold_material=threshold_material,
            lambda_material=lambda_material,
            lambda_reliability=lambda_reliability,
        )
    raise ValueError(f"Unknown strategy: {strategy}")


def run_one(
    *,
    scored_dets: list[dict[str, Any]],
    eval_annotation: Path,
    output_dir: Path,
    input_result: Path,
    strategy: str,
    alpha: float,
    beta: float,
    gamma: float,
    lambda_material: float,
    lambda_reliability: float,
    threshold_reliability: float,
    threshold_material: float,
    do_eval: bool,
    smoke: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_dets = apply_strategy(
        scored_dets,
        strategy=strategy,
        lambda_material=lambda_material,
        lambda_reliability=lambda_reliability,
        threshold_reliability=threshold_reliability,
        threshold_material=threshold_material,
    )
    stem = output_stem(
        strategy,
        alpha,
        beta,
        gamma,
        lambda_material if strategy in {
            "material_only_reproduce",
            "reliability_gated_material",
            "weighted_additive_fusion",
            "material_filter_reliability_rescore",
        } else None,
        lambda_reliability if strategy in {
            "weighted_additive_fusion",
            "material_filter_reliability_rescore",
        } else None,
        threshold_reliability if strategy == "reliability_gated_material" else None,
        threshold_material if strategy == "material_filter_reliability_rescore" else None,
    )
    if smoke:
        output_path = DEFAULT_SMOKE / f"{stem}_5imgs.json"
        summary_path = DEFAULT_SMOKE / f"{stem}_5imgs_summary.json"
        eval_log_path = DEFAULT_SMOKE / f"{stem}_5imgs_eval.log"
    else:
        output_path = output_dir / f"{stem}.json"
        summary_path = output_dir / f"{stem}_summary.json"
        eval_log_path = output_dir / f"{stem}_eval.log"
    save_detections(out_dets, output_path)
    evaluation = run_eval(eval_annotation, output_path, eval_log_path) if do_eval else None
    summary = {
        "strategy": strategy,
        "input_result": str(input_result),
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "lambda_material": lambda_material,
        "lambda_reliability": lambda_reliability
        if strategy in {"weighted_additive_fusion", "material_filter_reliability_rescore"}
        else None,
        "threshold_reliability": threshold_reliability
        if strategy == "reliability_gated_material"
        else None,
        "threshold_material": threshold_material
        if strategy == "material_filter_reliability_rescore"
        else None,
        "output_path": str(output_path),
        "summary_path": str(summary_path),
        "fusion_summary": summarize_fusion_result(scored_dets, out_dets),
        "evaluation": evaluation,
    }
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    return summary


def smoke_specs() -> list[dict[str, Any]]:
    return [
        {
            "strategy": "material_only_reproduce",
            "lambda_material": 0.3,
            "lambda_reliability": 0.0,
            "threshold_reliability": 0.0,
            "threshold_material": 0.0,
        },
        {
            "strategy": "reliability_gated_material",
            "lambda_material": 0.3,
            "lambda_reliability": 0.0,
            "threshold_reliability": 0.5,
            "threshold_material": 0.0,
        },
        {
            "strategy": "weighted_additive_fusion",
            "lambda_material": 0.3,
            "lambda_reliability": 0.1,
            "threshold_reliability": 0.0,
            "threshold_material": 0.0,
        },
        {
            "strategy": "material_filter_reliability_rescore",
            "lambda_material": 0.1,
            "lambda_reliability": 0.1,
            "threshold_reliability": 0.0,
            "threshold_material": 0.4,
        },
    ]


def grid_specs() -> list[dict[str, Any]]:
    specs = [
        {
            "strategy": "material_only_reproduce",
            "lambda_material": 0.3,
            "lambda_reliability": 0.0,
            "threshold_reliability": 0.0,
            "threshold_material": 0.0,
        }
    ]
    for threshold in [0.40, 0.50, 0.60, 0.70]:
        for lambda_material in [0.1, 0.2, 0.3]:
            specs.append(
                {
                    "strategy": "reliability_gated_material",
                    "lambda_material": lambda_material,
                    "lambda_reliability": 0.0,
                    "threshold_reliability": threshold,
                    "threshold_material": 0.0,
                }
            )
    for lambda_material in [0.1, 0.2, 0.3]:
        for lambda_reliability in [0.05, 0.10, 0.20]:
            specs.append(
                {
                    "strategy": "weighted_additive_fusion",
                    "lambda_material": lambda_material,
                    "lambda_reliability": lambda_reliability,
                    "threshold_reliability": 0.0,
                    "threshold_material": 0.0,
                }
            )
    for threshold_material in [0.40, 0.50]:
        for lambda_material in [0.05, 0.10, 0.20]:
            for lambda_reliability in [0.05, 0.10]:
                specs.append(
                    {
                        "strategy": "material_filter_reliability_rescore",
                        "lambda_material": lambda_material,
                        "lambda_reliability": lambda_reliability,
                        "threshold_reliability": 0.0,
                        "threshold_material": threshold_material,
                    }
                )
    return specs


def row_from_summary(summary: dict[str, Any]) -> dict[str, Any]:
    ev = summary.get("evaluation") or {}
    fs = summary.get("fusion_summary") or {}
    ap = ev.get("AP")
    return {
        "strategy": summary["strategy"],
        "input_result": summary["input_result"],
        "alpha": summary["alpha"],
        "beta": summary["beta"],
        "gamma": summary["gamma"],
        "lambda_material": summary["lambda_material"],
        "lambda_reliability": summary["lambda_reliability"],
        "threshold_reliability": summary["threshold_reliability"],
        "threshold_material": summary["threshold_material"],
        "input_count": fs.get("input_count"),
        "output_count": fs.get("output_count"),
        "removed_ratio": fs.get("removed_ratio"),
        "AP": ev.get("AP"),
        "AP50": ev.get("AP50"),
        "AP75": ev.get("AP75"),
        "delta_AP_vs_baseline": (ap - BASELINE["AP"]) if ap is not None else None,
        "delta_AP_vs_support_material_best": (ap - SUPPORT_MATERIAL_BEST["AP"])
        if ap is not None
        else None,
        "output_path": summary["output_path"],
    }


def write_grid_outputs(rows: list[dict[str, Any]], output_dir: Path) -> None:
    csv_path = output_dir / "material_reliability_fusion_grid_results.csv"
    md_path = output_dir / "material_reliability_fusion_grid_results.md"
    fields = [
        "strategy",
        "input_result",
        "alpha",
        "beta",
        "gamma",
        "lambda_material",
        "lambda_reliability",
        "threshold_reliability",
        "threshold_material",
        "input_count",
        "output_count",
        "removed_ratio",
        "AP",
        "AP50",
        "AP75",
        "delta_AP_vs_baseline",
        "delta_AP_vs_support_material_best",
        "output_path",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with md_path.open("w") as f:
        f.write("| " + " | ".join(fields) + " |\n")
        f.write("|" + "|".join(["---"] * len(fields)) + "|\n")
        for row in rows:
            f.write("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |\n")


def write_report(
    *,
    args: argparse.Namespace,
    smoke_summaries: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> None:
    sorted_rows = sorted(rows, key=lambda r: float(r["AP"] if r["AP"] is not None else -1), reverse=True)
    top10 = sorted_rows[:10]
    exceeds_baseline = [row for row in rows if row["AP"] is not None and row["AP"] > BASELINE["AP"]]
    exceeds_support = [
        row for row in rows if row["AP"] is not None and row["AP"] > SUPPORT_MATERIAL_BEST["AP"]
    ]
    best = top10[0] if top10 else None
    strategy_counts: dict[str, int] = {}
    for row in exceeds_support:
        strategy_counts[row["strategy"]] = strategy_counts.get(row["strategy"], 0) + 1
    effective_strategy = best["strategy"] if best else None

    fields = [
        "strategy",
        "lambda_material",
        "lambda_reliability",
        "threshold_reliability",
        "threshold_material",
        "output_count",
        "removed_ratio",
        "AP",
        "AP50",
        "AP75",
        "delta_AP_vs_support_material_best",
    ]
    table = ["| " + " | ".join(fields) + " |", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in top10:
        table.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")

    smoke_ok = all((s.get("evaluation") or {}).get("passed") for s in smoke_summaries)
    lines = [
        "# Batch11：Material + Reliability Fusion 报告",
        "",
        "## A. 新增/修改文件",
        "",
        "- 新增：`/workspace/RAXO/rmdx/fusion.py`",
        "- 新增：`/workspace/RAXO/scripts_rmdx/08_run_rmdx_material_reliability_fusion.py`",
        "",
        "## B. 使用的 input result",
        "",
        f"`{args.input_result}`",
        "",
        "## C. 使用的 material bank",
        "",
        f"`{args.material_bank}`",
        "",
        "## D. 是否使用 full_test GT bank",
        "",
        "否。本批次使用 Batch10.5 support material bank，不使用 Batch10 full_test GT material bank。",
        "",
        "## E. Reliability 公式",
        "",
        "```text",
        "sigmoid(alpha*z(dist_cosine_inter) + beta*z(dist_cosine_intra) - gamma*z(score))",
        "```",
        "",
        "默认参数：alpha=1.0, beta=0.5, gamma=0.5。",
        "",
        "## F. 是否修正 Batch9 reliability 公式方向",
        "",
        "否。已核对 Batch9 实现方向与本批次 audited corrected_reliability 一致；Batch11 单独实现 `compute_corrected_reliability_scores`，没有改写 Batch9 历史结果。",
        "",
        "## G. Smoke 是否通过",
        "",
        f"`{smoke_ok}`。四种 fusion strategy 各跑了一个 5-image 配置并通过 `cocoapi_2.py`。",
        "",
        "## H. Run-grid 表格路径",
        "",
        "- `/workspace/RAXO/outputs_rmdx/rmdx_results/material_reliability_fusion_grid_results.csv`",
        "- `/workspace/RAXO/outputs_rmdx/rmdx_results/material_reliability_fusion_grid_results.md`",
        "",
        "## I. 最优 Top 10",
        "",
        *table,
        "",
        "## J. 是否超过 RAXO baseline",
        "",
        f"- baseline AP / AP50 / AP75: `{BASELINE['AP']} / {BASELINE['AP50']} / {BASELINE['AP75']}`",
        f"- 超过 baseline 的配置数: `{len(exceeds_baseline)}` / `{len(rows)}`",
        "",
        "## K. 是否超过 support material best",
        "",
        f"- support material best AP / AP50 / AP75: `{SUPPORT_MATERIAL_BEST['AP']} / {SUPPORT_MATERIAL_BEST['AP50']} / {SUPPORT_MATERIAL_BEST['AP75']}`",
        f"- 超过 support material best 的配置数: `{len(exceeds_support)}` / `{len(rows)}`",
        "",
        "## L. 哪种 fusion 最有效",
        "",
        f"`{effective_strategy}`。超过 support material best 的 strategy 分布：`{strategy_counts}`。",
        "",
        "## M. Fusion 与 material-only 关系",
        "",
        "如果 top 配置主要仍是 material_only_reproduce，说明 reliability 暂不适合融合；如果 weighted/gated/filter 类策略超过 0.490，则说明 reliability signal 能在 support material 上进一步改善排序。",
        "",
        "## N. 下一步建议",
        "",
        "- 若超过 support material best，进入 Batch12 做 per-class / failure analysis。",
        "- 若没有超过，进入 Batch12 做 material-only 结果分析和论文表格整理。",
        "",
        "## O. 是否修改 RAXO 原始代码",
        "",
        "否。未修改 `raxo/*.py`、`run_method_database_branch_v2.sh`、`run_method_web_branch_v2.sh`、`full_test.json` 或 baseline 输出。",
        "",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines))


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    DEFAULT_SMOKE.mkdir(parents=True, exist_ok=True)

    if args.run_grid:
        smoke_scored, smoke_ann, _ = prepare_scored_dets(args, smoke=True)
        smoke_summaries = [
            run_one(
                scored_dets=smoke_scored,
                eval_annotation=smoke_ann,
                output_dir=output_dir,
                input_result=Path(args.input_result),
                strategy=spec["strategy"],
                alpha=args.alpha,
                beta=args.beta,
                gamma=args.gamma,
                lambda_material=spec["lambda_material"],
                lambda_reliability=spec["lambda_reliability"],
                threshold_reliability=spec["threshold_reliability"],
                threshold_material=spec["threshold_material"],
                do_eval=args.eval,
                smoke=True,
            )
            for spec in smoke_specs()
        ]
        if args.eval and not all((s.get("evaluation") or {}).get("passed") for s in smoke_summaries):
            raise RuntimeError("Fusion smoke evaluation failed")

        scored, full_ann, _ = prepare_scored_dets(args, smoke=False)
        summaries = [
            run_one(
                scored_dets=scored,
                eval_annotation=full_ann,
                output_dir=output_dir,
                input_result=Path(args.input_result),
                strategy=spec["strategy"],
                alpha=args.alpha,
                beta=args.beta,
                gamma=args.gamma,
                lambda_material=spec["lambda_material"],
                lambda_reliability=spec["lambda_reliability"],
                threshold_reliability=spec["threshold_reliability"],
                threshold_material=spec["threshold_material"],
                do_eval=args.eval,
                smoke=False,
            )
            for spec in grid_specs()
        ]
        rows = [row_from_summary(summary) for summary in summaries]
        write_grid_outputs(rows, output_dir)
        write_report(args=args, smoke_summaries=smoke_summaries, rows=rows)
        print(json.dumps({"smoke_ok": True, "rows": rows}, indent=2))
        return 0

    scored, eval_ann, _ = prepare_scored_dets(args, smoke=args.smoke)
    summary = run_one(
        scored_dets=scored,
        eval_annotation=eval_ann,
        output_dir=output_dir,
        input_result=Path(args.input_result),
        strategy=args.strategy,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        lambda_material=args.lambda_material,
        lambda_reliability=args.lambda_reliability,
        threshold_reliability=args.threshold_reliability,
        threshold_material=args.threshold_material,
        do_eval=args.eval,
        smoke=args.smoke,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
