#!/usr/bin/env python3
"""Run Batch17 cache-fusion smoke evaluation."""

from __future__ import annotations

import csv
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path("/workspace/RAXO")
CACHE_DIR = Path("/workspace/RAXO/outputs_rmdx/cache/det_compass_groundingDINO_100_0_smoke")
OUT_DIR = Path("/workspace/RAXO/outputs_rmdx/smoke/cache_fusion")
REPORT = Path("/workspace/RAXO/docs_rmdx/batch17_cache_fusion_smoke_report.md")
FULL_ANNOTATION = Path("/datasets/xray-datasets/DET-COMPASS/annotations/full_test.json")
SMOKE_ANNOTATION = OUT_DIR / "full_test_5imgs_cache_fusion.json"
B14_VISUAL = Path("/workspace/RAXO/outputs_rmdx/smoke/preargmax/visual_only_uncertanty.json")


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


def load_dets(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(path)
    return data


def summarize(path: Path) -> dict[str, Any]:
    dets = load_dets(path)
    changed = sum(
        1
        for det in dets
        if det.get("rmdx_original_category_id") is not None
        and int(det["category_id"]) != int(det["rmdx_original_category_id"])
    )
    return {
        "count": len(dets),
        "changed_category_count": changed,
        "category_top20": Counter(int(det["category_id"]) for det in dets).most_common(20),
    }


def write_smoke_annotation() -> list[int]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with FULL_ANNOTATION.open() as f:
        data = json.load(f)
    images = data["images"][:5]
    image_ids = {int(img["id"]) for img in images}
    out = {
        "images": images,
        "annotations": [ann for ann in data["annotations"] if int(ann["image_id"]) in image_ids],
        "categories": data["categories"],
    }
    with SMOKE_ANNOTATION.open("w") as f:
        json.dump(out, f)
    return [int(img["id"]) for img in images]


def parse_eval(log_path: Path) -> dict[str, Any]:
    text = log_path.read_text() if log_path.exists() else ""
    aps = re.findall(r"Average Precision.*=\s+(-?\d+\.\d+)", text)
    return {
        "AP": float(aps[0]) if len(aps) > 0 else None,
        "AP50": float(aps[1]) if len(aps) > 1 else None,
        "AP75": float(aps[2]) if len(aps) > 2 else None,
    }


def configs() -> list[dict[str, Any]]:
    return [
        {"name": "visual_only", "mode": "visual_only", "lm": 0.0, "lr": 0.0},
        {"name": "material_preargmax_lm0p05", "mode": "material_preargmax", "lm": 0.05, "lr": 0.0},
        {"name": "material_preargmax_lm0p10", "mode": "material_preargmax", "lm": 0.10, "lr": 0.0},
        {
            "name": "material_reliability_preargmax_lm0p05_lr0p50",
            "mode": "material_reliability_preargmax",
            "lm": 0.05,
            "lr": 0.50,
        },
    ]


def run_one(config: dict[str, Any]) -> dict[str, Any]:
    output = OUT_DIR / f"{config['name']}.json"
    uncert = OUT_DIR / f"{config['name']}_uncertanty.json"
    fusion = run_cmd(
        [
            "python3",
            "/workspace/RAXO/rmdx/cache_fusion.py",
            "--cache-dir",
            str(CACHE_DIR),
            "--mode",
            config["mode"],
            "--lambda-material",
            str(config["lm"]),
            "--lambda-reliability",
            str(config["lr"]),
            "--output-path",
            str(output),
        ],
        OUT_DIR / f"{config['name']}_fusion.log",
    )
    if not fusion["passed"]:
        return {"config": config, "passed": False, "stage": "fusion", "log": fusion}
    raw = summarize(output)
    uncertainty = run_cmd(
        ["python3", "raxo/uncertanty_estimation_fixed.py", "--dets", str(output)],
        OUT_DIR / f"{config['name']}_uncertanty.log",
    )
    if not uncertainty["passed"]:
        return {"config": config, "passed": False, "stage": "uncertanty", "log": uncertainty}
    after = summarize(uncert)
    eval_result = run_cmd(
        [
            "python3",
            "raxo/cocoapi_2.py",
            "--cocoGt",
            str(SMOKE_ANNOTATION),
            "--cocoDt",
            str(uncert),
        ],
        OUT_DIR / f"{config['name']}_eval.log",
    )
    metrics = parse_eval(Path(eval_result["log_path"]))
    return {
        "config": config,
        "passed": bool(eval_result["passed"] and metrics["AP"] is not None),
        "output_path": str(output),
        "uncertanty_output_path": str(uncert),
        "detection_count_before_uncertanty": raw["count"],
        "detection_count_after_uncertanty": after["count"],
        "changed_category_count": raw["changed_category_count"],
        "AP": metrics["AP"],
        "AP50": metrics["AP50"],
        "AP75": metrics["AP75"],
        "fusion_log": fusion["log_path"],
        "eval_log": eval_result["log_path"],
    }


def write_report(image_ids: list[int], rows: list[dict[str, Any]]) -> None:
    b14_count = summarize(B14_VISUAL)["count"] if B14_VISUAL.exists() else None
    visual = next((row for row in rows if row["config"]["name"] == "visual_only"), None)
    count_close = (
        visual is not None
        and visual.get("passed")
        and b14_count is not None
        and abs(int(visual["detection_count_after_uncertanty"]) - int(b14_count)) <= 5
    )
    ap_close = visual is not None and visual.get("AP") is not None and abs(float(visual["AP"]) - 0.940) <= 0.002
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Batch17 Cache Fusion Smoke Report",
        "",
        f"- image_ids: `{image_ids}`",
        f"- cache dir: `{CACHE_DIR}`",
        f"- smoke annotation: `{SMOKE_ANNOTATION}`",
        f"- Batch14.1 visual_only uncert count: `{b14_count}`",
        f"- visual count close: `{count_close}`",
        f"- visual AP close to Batch14.1 smoke AP=0.940: `{ap_close}`",
        f"- success: `{all(row.get('passed') for row in rows) and count_close and ap_close}`",
        "",
        "| config | passed | det_count | uncert_count | changed_category_count | AP | AP50 | AP75 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        cfg = row["config"]
        lines.append(
            f"| {cfg['name']} | {row.get('passed')} | {row.get('detection_count_before_uncertanty')} | "
            f"{row.get('detection_count_after_uncertanty')} | {row.get('changed_category_count')} | "
            f"{row.get('AP')} | {row.get('AP50')} | {row.get('AP75')} |"
        )
    REPORT.write_text("\n".join(lines) + "\n")
    if not (all(row.get("passed") for row in rows) and count_close and ap_close):
        raise SystemExit("cache fusion smoke did not match Batch14.1 closely enough")


def main() -> int:
    image_ids = write_smoke_annotation()
    rows = [run_one(config) for config in configs()]
    write_report(image_ids, rows)
    print(json.dumps(rows, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
