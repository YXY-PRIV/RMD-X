#!/usr/bin/env python3
"""Run Batch14.1 RMD-X pre-argmax fusion smoke tests."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


FULL_WITH_MASKS = Path(
    "/models/ICCV25_experimentation/initial_detections/groundingDINO/"
    "coco_results.bbox_DET-COMPASS_with_masks.json"
)
PROTOTYPES = Path("/models/ICCV25_experimentation/known_prototypes_sam2.pt")
MATERIAL_BANK = Path("/workspace/RAXO/outputs_rmdx/material/material_colour_bank_support_full_bbox_only.json")
ANNOTATION = Path("/datasets/xray-datasets/DET-COMPASS/annotations/full_test.json")
IMAGE_DIR = Path("/datasets/xray-datasets/DET-COMPASS/test")
SMOKE_DIR = Path("/workspace/RAXO/outputs_rmdx/smoke/preargmax")
REPORT_PATH = Path("/workspace/RAXO/docs_rmdx/batch14_1_preargmax_smoke_report.md")
REFERENCE_KNOWN_FINAL = Path(
    "/workspace/RAXO/outputs_rmdx/smoke/"
    "coco_results.bbox_DET-COMPASS_with_masks_5imgs_nms_0.8_our_method_known_final.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch14.1 pre-argmax smoke")
    parser.add_argument("--debug-limit-proposals", type=int, default=None)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f)


def run_cmd(cmd: list[str], log_path: Path) -> dict[str, Any]:
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
    return {
        "cmd": " ".join(cmd),
        "returncode": proc.returncode,
        "passed": proc.returncode == 0,
        "log_path": str(log_path),
        "stdout_tail": "\n".join(proc.stdout.splitlines()[-80:]),
    }


def run_eval(annotation: Path, detections: Path, log_path: Path) -> dict[str, Any]:
    result = run_cmd(
        [
            "python3",
            "raxo/cocoapi_2.py",
            "--cocoGt",
            str(annotation),
            "--cocoDt",
            str(detections),
        ],
        log_path,
    )
    text = Path(result["log_path"]).read_text()
    aps = re.findall(r"Average Precision.*=\s+(-?\d+\.\d+)", text)
    result.update(
        {
            "AP": float(aps[0]) if len(aps) > 0 else None,
            "AP50": float(aps[1]) if len(aps) > 1 else None,
            "AP75": float(aps[2]) if len(aps) > 2 else None,
        }
    )
    return result


def build_smoke_inputs() -> dict[str, Any]:
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    ann = load_json(ANNOTATION)
    images = ann["images"][:5]
    image_ids = {int(img["id"]) for img in images}
    ann5 = {
        "images": images,
        "annotations": [a for a in ann["annotations"] if int(a["image_id"]) in image_ids],
        "categories": ann["categories"],
    }
    ann5_path = SMOKE_DIR / "full_test_5imgs_preargmax.json"
    save_json(ann5, ann5_path)

    wm = load_json(FULL_WITH_MASKS)
    wm5 = {
        "images": [img for img in wm["images"] if int(img["id"]) in image_ids],
        "annotations": [a for a in wm["annotations"] if int(a["image_id"]) in image_ids],
        "categories": wm["categories"],
    }
    wm5_path = SMOKE_DIR / "coco_results.bbox_DET-COMPASS_with_masks_5imgs_preargmax.json"
    save_json(wm5, wm5_path)

    for path in [ann5_path, wm5_path]:
        result = subprocess.run(
            ["python3", "-m", "json.tool", str(path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"{path} invalid JSON: {result.stderr}")

    return {
        "image_ids": sorted(image_ids),
        "annotation_path": ann5_path,
        "with_masks_path": wm5_path,
        "image_count": len(images),
        "gt_annotation_count": len(ann5["annotations"]),
        "with_masks_annotation_count": len(wm5["annotations"]),
        "category_count": len(ann5["categories"]),
    }


def summarize_dets(path: Path) -> dict[str, Any]:
    dets = load_json(path)
    scores = [float(d["score"]) for d in dets if "score" in d]
    return {
        "path": str(path),
        "count": len(dets),
        "category_distribution_top20": Counter(int(d["category_id"]) for d in dets).most_common(20),
        "score_range": {
            "min": min(scores) if scores else None,
            "max": max(scores) if scores else None,
            "mean": sum(scores) / len(scores) if scores else None,
        },
        "changed_category_count": sum(
            1
            for d in dets
            if d.get("rmdx_original_category_id") is not None
            and int(d["category_id"]) != int(d["rmdx_original_category_id"])
        ),
    }


def run_config(config: dict[str, Any], smoke_inputs: dict[str, Any], debug_limit: int | None) -> dict[str, Any]:
    output_path = SMOKE_DIR / f"{config['name']}.json"
    log_path = SMOKE_DIR / f"{config['name']}_main.log"
    cmd = [
        "python3",
        "/workspace/RAXO/rmdx/rmdx_main_with_masks_prop.py",
        "--json_res",
        str(smoke_inputs["with_masks_path"]),
        "--image_path",
        str(IMAGE_DIR),
        "--prototypes",
        str(PROTOTYPES),
        "--nms",
        "0.8",
        "--name",
        config["name"],
        "--branch",
        "known",
        "--rmdx-mode",
        config["mode"],
        "--material-bank",
        str(MATERIAL_BANK),
        "--lambda-material",
        str(config["lambda_material"]),
        "--lambda-reliability",
        str(config["lambda_reliability"]),
        "--output-path",
        str(output_path),
    ]
    if debug_limit is not None:
        cmd += ["--debug-limit-proposals", str(debug_limit)]
    main_result = run_cmd(cmd, log_path)
    if not main_result["passed"]:
        return {"config": config, "main": main_result, "passed": False}

    summary = summarize_dets(output_path)
    uncert_result = run_cmd(
        ["python3", "raxo/uncertanty_estimation_fixed.py", "--dets", str(output_path)],
        SMOKE_DIR / f"{config['name']}_uncertanty.log",
    )
    uncert_path = Path(str(output_path).replace(".json", "_uncertanty.json"))
    eval_result = None
    uncert_summary = None
    if uncert_result["passed"] and uncert_path.exists():
        uncert_summary = summarize_dets(uncert_path)
        eval_result = run_eval(
            smoke_inputs["annotation_path"],
            uncert_path,
            SMOKE_DIR / f"{config['name']}_eval.log",
        )

    return {
        "config": config,
        "main": main_result,
        "output_path": str(output_path),
        "summary": summary,
        "uncertanty": uncert_result,
        "uncertanty_path": str(uncert_path),
        "uncertanty_summary": uncert_summary,
        "evaluation": eval_result,
        "passed": bool(
            main_result["passed"]
            and uncert_result["passed"]
            and eval_result
            and eval_result["passed"]
        ),
    }


def compare_visual_reference(visual_result: dict[str, Any]) -> dict[str, Any]:
    if not REFERENCE_KNOWN_FINAL.exists():
        return {"reference_exists": False}
    ref = summarize_dets(REFERENCE_KNOWN_FINAL)
    vis = visual_result.get("summary") or {}
    return {
        "reference_exists": True,
        "reference_path": str(REFERENCE_KNOWN_FINAL),
        "reference_count": ref["count"],
        "visual_count": vis.get("count"),
        "count_delta": (vis.get("count") - ref["count"]) if vis.get("count") is not None else None,
        "reference_score_range": ref["score_range"],
        "visual_score_range": vis.get("score_range"),
        "reference_category_top20": ref["category_distribution_top20"],
        "visual_category_top20": vis.get("category_distribution_top20"),
    }


def write_report(smoke_inputs: dict[str, Any], results: list[dict[str, Any]], visual_compare: dict[str, Any]) -> None:
    lines = [
        "# Batch14.1：RMD-X Pre-argmax Fusion Smoke 报告",
        "",
        "## A. 新增文件",
        "",
        "- `/workspace/RAXO/rmdx/rmdx_main_with_masks_prop.py`",
        "- `/workspace/RAXO/rmdx/preargmax_fusion.py`",
        "- `/workspace/RAXO/scripts_rmdx/10_run_rmdx_preargmax_smoke.py`",
        "",
        "## B. 是否修改 RAXO 原始代码",
        "",
        "否。未修改 `/workspace/RAXO/raxo/main_with_masks_prop.py`、`raxo/*.py` 或官方 run 脚本。",
        "",
        "## C. Material Bank",
        "",
        f"- 使用 support material bank: `{MATERIAL_BANK}`",
        "- 是否使用 full_test GT bank: `False`",
        "",
        "## D. Smoke 输入",
        "",
        f"- image_id 列表: `{smoke_inputs['image_ids']}`",
        f"- image 数: `{smoke_inputs['image_count']}`",
        f"- GT annotation 数: `{smoke_inputs['gt_annotation_count']}`",
        f"- with_masks proposal 数: `{smoke_inputs['with_masks_annotation_count']}`",
        f"- category 数: `{smoke_inputs['category_count']}`",
        f"- annotation: `{smoke_inputs['annotation_path']}`",
        f"- with_masks: `{smoke_inputs['with_masks_path']}`",
        "",
        "## E. 配置结果",
        "",
        "| config | passed | output | uncertanty_output | det_count | uncert_count | changed_category_count | AP | AP50 | AP75 |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        cfg = result["config"]
        ev = result.get("evaluation") or {}
        summary = result.get("summary") or {}
        us = result.get("uncertanty_summary") or {}
        lines.append(
            "| "
            + " | ".join(
                [
                    cfg["name"],
                    str(result.get("passed")),
                    f"`{result.get('output_path')}`",
                    f"`{result.get('uncertanty_path')}`",
                    str(summary.get("count")),
                    str(us.get("count")),
                    str(summary.get("changed_category_count")),
                    str(ev.get("AP")),
                    str(ev.get("AP50")),
                    str(ev.get("AP75")),
                ]
            )
            + " |"
        )

    failures = [r for r in results if not r.get("passed")]
    lines += [
        "",
        "## F. visual_only 与 Batch5.2-smoke 是否接近",
        "",
        f"`{visual_compare}`",
        "",
        "## G. Category 变化",
        "",
    ]
    for result in results:
        summary = result.get("summary") or {}
        lines.append(
            f"- `{result['config']['name']}` changed_category_count: `{summary.get('changed_category_count')}`"
        )
    lines += [
        "",
        "## H. 是否可以进入 Batch14.2 full grid",
        "",
        "`{}`".format(not failures),
        "",
        "## I. 失败信息",
        "",
    ]
    if failures:
        for result in failures:
            lines.append(f"### {result['config']['name']}")
            main = result.get("main") or {}
            lines.append(f"- main passed: `{main.get('passed')}`")
            lines.append(f"- log: `{main.get('log_path')}`")
            lines.append("```text")
            lines.append(str(main.get("stdout_tail", "")))
            lines.append("```")
    else:
        lines.append("无。")
    lines += [
        "",
        "## J. 说明",
        "",
        "- 本批只跑 5-image smoke，AP 仅用于链路验证，不代表真实性能。",
        "- 当前仍是 colour-only material，不包含 Grey/High/Low/Density 多通道。",
        "- pre-argmax fusion 不使用 GT matching 信息参与 scoring。",
        "",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines))


def main() -> int:
    args = parse_args()
    try:
        smoke_inputs = build_smoke_inputs()
        configs = [
            {
                "name": "visual_only",
                "mode": "visual_only",
                "lambda_material": 0.0,
                "lambda_reliability": 0.0,
            },
            {
                "name": "material_preargmax_lm0p05",
                "mode": "material_preargmax",
                "lambda_material": 0.05,
                "lambda_reliability": 0.0,
            },
            {
                "name": "material_preargmax_lm0p10",
                "mode": "material_preargmax",
                "lambda_material": 0.10,
                "lambda_reliability": 0.0,
            },
            {
                "name": "material_reliability_preargmax_lm0p10_lr0p5",
                "mode": "material_reliability_preargmax",
                "lambda_material": 0.10,
                "lambda_reliability": 0.50,
            },
        ]
        results = [run_config(cfg, smoke_inputs, args.debug_limit_proposals) for cfg in configs]
        visual_compare = compare_visual_reference(results[0])
        write_report(smoke_inputs, results, visual_compare)
        print(
            json.dumps(
                {"smoke_inputs": smoke_inputs, "results": results, "report": str(REPORT_PATH)},
                indent=2,
                default=str,
            )
        )
        return 0 if all(r.get("passed") for r in results) else 1
    except Exception:
        SMOKE_DIR.mkdir(parents=True, exist_ok=True)
        error = traceback.format_exc()
        (SMOKE_DIR / "preargmax_smoke_exception.log").write_text(error)
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(
            "# Batch14.1：RMD-X Pre-argmax Fusion Smoke 报告\n\n"
            "运行脚本发生异常：\n\n```text\n" + error + "\n```\n"
        )
        print(error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
