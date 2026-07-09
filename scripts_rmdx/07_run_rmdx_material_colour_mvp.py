#!/usr/bin/env python3
"""Run RMD-X colour-only material MVP experiments."""

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

from rmdx.material import (  # noqa: E402
    FEATURE_NAMES,
    apply_material_append_only,
    apply_material_filter_and_rescore,
    apply_material_rescore,
    load_material_bank,
    score_detections_with_material,
)
from rmdx.rescore import filter_by_image_ids, load_detections, save_detections  # noqa: E402


DEFAULT_INPUT = Path(
    "/models/ICCV25_experimentation/initial_detections/groundingDINO/"
    "coco_results.bbox_DET-COMPASS_with_masks_nms_0.8_our_method_known_final_uncertanty.json"
)
DEFAULT_ANN = Path("/datasets/xray-datasets/DET-COMPASS/annotations/full_test.json")
DEFAULT_IMAGE_DIR = Path("/datasets/xray-datasets/DET-COMPASS/test")
DEFAULT_BANK = Path("/workspace/RAXO/outputs_rmdx/material/material_colour_bank_bbox_only.json")
DEFAULT_OUT = Path("/workspace/RAXO/outputs_rmdx/rmdx_results")
DEFAULT_SMOKE = Path("/workspace/RAXO/outputs_rmdx/smoke")
REPORT_PATH = Path("/workspace/RAXO/docs_rmdx/batch10_material_colour_mvp_report.md")

BASELINE = {"AP": 0.480, "AP50": 0.544, "AP75": 0.488}
RESCORE_LAMBDAS = [0.05, 0.10, 0.20, 0.30]
FILTER_THRESHOLDS = [0.20, 0.30, 0.40, 0.50]
FILTER_LAMBDAS = [0.05, 0.10]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RMD-X material colour-only MVP")
    parser.add_argument("--input-result", default=str(DEFAULT_INPUT))
    parser.add_argument("--annotation", default=str(DEFAULT_ANN))
    parser.add_argument("--image-dir", default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("--material-bank", default=str(DEFAULT_BANK))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--strategy",
        choices=[
            "material_score_append_only",
            "material_rescore",
            "material_filter_and_rescore",
        ],
        default="material_score_append_only",
    )
    parser.add_argument("--lambda-material", type=float, default=0.10)
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument("--mode", choices=["bbox_only", "mask_weighted"], default="bbox_only")
    parser.add_argument("--output-prefix", default="")
    parser.add_argument("--report-output", default=None)
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
    mode: str,
    lambda_material: float | None,
    threshold: float | None,
    output_prefix: str = "",
) -> str:
    prefix = f"{output_prefix}_" if output_prefix else ""
    return (
        f"rmdx_material_colour_{prefix}{strategy}_{mode}_"
        f"lambda{slug_float(lambda_material)}_thr{slug_float(threshold)}"
    )


def apply_strategy(
    scored_dets: list[dict[str, Any]],
    *,
    strategy: str,
    lambda_material: float,
    threshold: float,
) -> list[dict[str, Any]]:
    if strategy == "material_score_append_only":
        return apply_material_append_only(scored_dets)
    if strategy == "material_rescore":
        return apply_material_rescore(scored_dets, lambda_material, replace_score=True)
    if strategy == "material_filter_and_rescore":
        return apply_material_filter_and_rescore(scored_dets, threshold, lambda_material)
    raise ValueError(f"Unknown strategy: {strategy}")


def material_score_range(dets: list[dict[str, Any]]) -> dict[str, float | None]:
    vals = [float(det["rmdx_material_score"]) for det in dets if "rmdx_material_score" in det]
    if not vals:
        return {"min": None, "max": None, "mean": None}
    return {"min": min(vals), "max": max(vals), "mean": sum(vals) / len(vals)}


def run_one(
    *,
    scored_dets: list[dict[str, Any]],
    annotation_path: Path,
    output_dir: Path,
    strategy: str,
    mode: str,
    lambda_material: float,
    threshold: float,
    output_prefix: str,
    do_eval: bool,
    smoke: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_dets = apply_strategy(
        scored_dets,
        strategy=strategy,
        lambda_material=lambda_material,
        threshold=threshold,
    )
    stem = output_stem(
        strategy,
        mode,
        lambda_material if strategy != "material_score_append_only" else None,
        threshold if strategy == "material_filter_and_rescore" else None,
        output_prefix,
    )
    if smoke:
        smoke_name = f"rmdx_material_colour_{output_prefix + '_' if output_prefix else ''}5imgs"
        output_path = DEFAULT_SMOKE / f"{smoke_name}.json"
        summary_path = DEFAULT_SMOKE / f"{smoke_name}_summary.json"
        eval_log_path = DEFAULT_SMOKE / f"{smoke_name}_eval.log"
    else:
        output_path = output_dir / f"{stem}.json"
        summary_path = output_dir / f"{stem}_summary.json"
        eval_log_path = output_dir / f"{stem}_eval.log"
    save_detections(out_dets, output_path)
    evaluation = run_eval(annotation_path, output_path, eval_log_path) if do_eval else None
    summary = {
        "strategy": strategy,
        "mode": mode,
        "lambda_material": lambda_material if strategy != "material_score_append_only" else None,
        "threshold": threshold if strategy == "material_filter_and_rescore" else None,
        "input_detection_count": len(scored_dets),
        "output_detection_count": len(out_dets),
        "removed_count": len(scored_dets) - len(out_dets),
        "removed_ratio": ((len(scored_dets) - len(out_dets)) / len(scored_dets)) if scored_dets else None,
        "material_score_range": material_score_range(out_dets),
        "output_path": str(output_path),
        "summary_path": str(summary_path),
        "evaluation": evaluation,
    }
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    return summary


def grid_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        {
            "strategy": "material_score_append_only",
            "lambda_material": 0.0,
            "threshold": 0.0,
        }
    ]
    for value in RESCORE_LAMBDAS:
        specs.append(
            {
                "strategy": "material_rescore",
                "lambda_material": value,
                "threshold": 0.0,
            }
        )
    for threshold in FILTER_THRESHOLDS:
        for value in FILTER_LAMBDAS:
            specs.append(
                {
                    "strategy": "material_filter_and_rescore",
                    "lambda_material": value,
                    "threshold": threshold,
                }
            )
    return specs


def row_from_summary(summary: dict[str, Any]) -> dict[str, Any]:
    ev = summary.get("evaluation") or {}
    ap = ev.get("AP")
    ap50 = ev.get("AP50")
    ap75 = ev.get("AP75")
    return {
        "strategy": summary["strategy"],
        "mode": summary["mode"],
        "lambda_material": summary["lambda_material"],
        "threshold": summary["threshold"],
        "input_count": summary["input_detection_count"],
        "output_count": summary["output_detection_count"],
        "removed_ratio": summary["removed_ratio"],
        "AP": ap,
        "AP50": ap50,
        "AP75": ap75,
        "delta_AP_vs_baseline": (ap - BASELINE["AP"]) if ap is not None else None,
        "delta_AP50_vs_baseline": (ap50 - BASELINE["AP50"]) if ap50 is not None else None,
        "delta_AP75_vs_baseline": (ap75 - BASELINE["AP75"]) if ap75 is not None else None,
        "output_path": summary["output_path"],
    }


def write_grid_outputs(rows: list[dict[str, Any]], output_dir: Path, output_prefix: str = "") -> None:
    if output_prefix:
        csv_path = output_dir / f"material_colour_{output_prefix}_grid_results.csv"
        md_path = output_dir / f"material_colour_{output_prefix}_grid_results.md"
    else:
        csv_path = output_dir / "material_colour_mvp_grid_results.csv"
        md_path = output_dir / "material_colour_mvp_grid_results.md"
    fieldnames = [
        "strategy",
        "mode",
        "lambda_material",
        "threshold",
        "input_count",
        "output_count",
        "removed_ratio",
        "AP",
        "AP50",
        "AP75",
        "delta_AP_vs_baseline",
        "delta_AP50_vs_baseline",
        "delta_AP75_vs_baseline",
        "output_path",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with md_path.open("w") as f:
        f.write("| " + " | ".join(fieldnames) + " |\n")
        f.write("|" + "|".join(["---"] * len(fieldnames)) + "|\n")
        for row in rows:
            f.write("| " + " | ".join(str(row.get(field, "")) for field in fieldnames) + " |\n")


def write_report(
    *,
    args: argparse.Namespace,
    material_bank: dict[str, Any],
    smoke_summary: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    sorted_rows = sorted(rows, key=lambda r: float(r["AP"] if r["AP"] is not None else -1), reverse=True)
    top5 = sorted_rows[:5]
    exceeds = [row for row in rows if row["AP"] is not None and row["AP"] > BASELINE["AP"]]
    ap50_up = [row for row in rows if row["AP50"] is not None and row["AP50"] > BASELINE["AP50"]]
    ap75_up = [row for row in rows if row["AP75"] is not None and row["AP75"] > BASELINE["AP75"]]
    proto_counts = [
        int(proto["sample_count"]) for proto in material_bank.get("prototypes", {}).values()
    ]
    hist = {}
    for count in proto_counts:
        hist[count] = hist.get(count, 0) + 1

    def table(rows_: list[dict[str, Any]]) -> list[str]:
        fields = ["strategy", "mode", "lambda_material", "threshold", "output_count", "AP", "AP50", "AP75"]
        lines = ["| " + " | ".join(fields) + " |", "|" + "|".join(["---"] * len(fields)) + "|"]
        for row in rows_:
            lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
        return lines

    if args.output_prefix:
        csv_name = f"material_colour_{args.output_prefix}_grid_results.csv"
        md_name = f"material_colour_{args.output_prefix}_grid_results.md"
    else:
        csv_name = "material_colour_mvp_grid_results.csv"
        md_name = "material_colour_mvp_grid_results.md"
    if args.report_output:
        report_path = Path(args.report_output)
    elif args.output_prefix:
        report_path = Path(f"/workspace/RAXO/docs_rmdx/material_colour_{args.output_prefix}_run_report.md")
    else:
        report_path = REPORT_PATH

    lines = [
        "# Batch10：Material Colour-only MVP 报告",
        "",
        "## A. 新增/修改文件",
        "",
        "- 新增：`/workspace/RAXO/rmdx/material.py`",
        "- 修改：`/workspace/RAXO/rmdx/rescore.py`（仅保留兼容，material 主逻辑在 `rmdx/material.py`）",
        "- 新增：`/workspace/RAXO/scripts_rmdx/06_build_material_colour_bank.py`",
        "- 新增：`/workspace/RAXO/scripts_rmdx/07_run_rmdx_material_colour_mvp.py`",
        "",
        "## B. Material Bank 构建方式",
        "",
        f"- annotation: `{args.annotation}`",
        f"- image_dir: `{args.image_dir}`",
        f"- bank: `{args.material_bank}`",
        "- 使用 `full_test.json` 的 GT bbox 从 Colour X-ray 图像提取 bbox crop feature，按 category_id 求均值 prototype。",
        "",
        "## C. Material Feature Names",
        "",
        "```text",
        "\n".join(FEATURE_NAMES),
        "```",
        "",
        "## D. Material Bank 类别数量与 sample_count 分布",
        "",
        f"- prototype_count: `{material_bank.get('prototype_count')}`",
        f"- sample_count min/max/mean: `{min(proto_counts) if proto_counts else None} / {max(proto_counts) if proto_counts else None} / {(sum(proto_counts) / len(proto_counts)) if proto_counts else None}`",
        f"- sample_count histogram: `{hist}`",
        "",
        "## E. Exploratory MVP 说明",
        "",
        "当前 bank 使用 full_test GT 构建，只是 colour-only exploratory MVP，不适合作为最终公平泛化实验结论。后续应改用 support/prototype split 或 external material bank。",
        "",
        "## F. Smoke 是否通过",
        "",
        f"`{bool(smoke_summary.get('evaluation', {}).get('passed'))}`",
        "",
        "## G. Full Run-grid 表格路径",
        "",
        f"- `/workspace/RAXO/outputs_rmdx/rmdx_results/{csv_name}`",
        f"- `/workspace/RAXO/outputs_rmdx/rmdx_results/{md_name}`",
        "",
        "## H. 最优策略 Top 5",
        "",
        *table(top5),
        "",
        "## I. 是否超过 baseline",
        "",
        f"- baseline AP / AP50 / AP75: `{BASELINE['AP']} / {BASELINE['AP50']} / {BASELINE['AP75']}`",
        f"- AP 超过 baseline 的策略数: `{len(exceeds)}`",
        "",
        "## J. AP50/AP75 局部提升",
        "",
        f"- AP50 超过 baseline 的策略数: `{len(ap50_up)}`",
        f"- AP75 超过 baseline 的策略数: `{len(ap75_up)}`",
        "",
        "## K. 失败或下降原因分析",
        "",
        "该 MVP 只使用 Colour X-ray bbox crop 统计特征；多数 DET-COMPASS 类别只有 1 个 GT prototype，bank 方差估计很弱。post-rescore 只改变 COCO 排序，不改变 bbox/category，因此容易因 score 校准不足导致 AP 下降。",
        "",
        "## L. 下一步建议",
        "",
        "- 改为 support/prototype split 构建 material bank。",
        "- 引入 Grey/High/Low/Density 多通道后再做 material descriptor。",
        "- 在 `main_with_masks_prop.py` similarity 层融合 material，而不是只做末端 score 后处理。",
        "- 可以进入 Batch11 risk sidecar。",
        "",
        "## M. 是否修改 RAXO 原始代码",
        "",
        "否。未修改 `raxo/*.py`、`run_method_database_branch_v2.sh`、`run_method_web_branch_v2.sh`、`full_test.json`、known_final 或 known_final_uncertanty。",
        "",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))


def prepare_scored(
    *,
    args: argparse.Namespace,
    smoke: bool,
) -> tuple[list[dict[str, Any]], Path, dict[str, Any], dict[str, Any]]:
    annotation = load_annotation(Path(args.annotation))
    input_dets = load_detections(Path(args.input_result))
    eval_annotation = Path(args.annotation)
    if smoke:
        image_ids, eval_annotation = write_annotation_5imgs(
            annotation, DEFAULT_SMOKE / "full_test_5imgs_for_material_mvp.json"
        )
        input_dets = filter_by_image_ids(input_dets, image_ids)
    material_bank = load_material_bank(args.material_bank)
    scored, score_summary = score_detections_with_material(
        input_dets,
        annotation,
        args.image_dir,
        material_bank,
        mode=args.mode,
    )
    return scored, eval_annotation, material_bank, score_summary


def main() -> int:
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    DEFAULT_SMOKE.mkdir(parents=True, exist_ok=True)

    if args.run_grid:
        smoke_scored, smoke_ann, material_bank, _ = prepare_scored(args=args, smoke=True)
        smoke_summary = run_one(
            scored_dets=smoke_scored,
            annotation_path=smoke_ann,
            output_dir=Path(args.output_dir),
            strategy="material_score_append_only",
        mode=args.mode,
        lambda_material=0.0,
        threshold=0.0,
        output_prefix=args.output_prefix,
        do_eval=args.eval,
        smoke=True,
        )
        if args.eval and not smoke_summary.get("evaluation", {}).get("passed"):
            raise RuntimeError(f"Smoke evaluation failed: {smoke_summary['evaluation']}")

        scored, full_ann, material_bank, _ = prepare_scored(args=args, smoke=False)
        summaries = [
            run_one(
                scored_dets=scored,
                annotation_path=full_ann,
                output_dir=Path(args.output_dir),
                strategy=spec["strategy"],
                mode=args.mode,
                lambda_material=spec["lambda_material"],
                threshold=spec["threshold"],
                output_prefix=args.output_prefix,
                do_eval=args.eval,
                smoke=False,
            )
            for spec in grid_specs()
        ]
        rows = [row_from_summary(summary) for summary in summaries]
        write_grid_outputs(rows, Path(args.output_dir), args.output_prefix)
        write_report(
            args=args,
            material_bank=material_bank,
            smoke_summary=smoke_summary,
            rows=rows,
        )
        print(json.dumps({"smoke_ok": smoke_summary["evaluation"]["passed"] if args.eval else None, "rows": rows}, indent=2))
        return 0

    scored, eval_ann, _, _ = prepare_scored(args=args, smoke=args.smoke)
    summary = run_one(
        scored_dets=scored,
        annotation_path=eval_ann,
        output_dir=Path(args.output_dir),
        strategy=args.strategy,
        mode=args.mode,
        lambda_material=args.lambda_material,
        threshold=args.threshold,
        output_prefix=args.output_prefix,
        do_eval=args.eval,
        smoke=args.smoke,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
