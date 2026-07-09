#!/usr/bin/env python3
"""Run Batch21 multi-detector cache-fusion grid and COCO evaluation."""

from __future__ import annotations

import csv
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path("/workspace/RAXO")
ANNOTATION = Path("/datasets/xray-datasets/DET-COMPASS/annotations/full_test.json")
BASELINE_TABLE = ROOT / "outputs_rmdx" / "rmdx_results" / "multidetector_raxo_baseline_results.csv"
CACHE_ROOT = ROOT / "outputs_rmdx" / "cache"
OUT_ROOT = ROOT / "outputs_rmdx" / "preargmax_cache_multidetector"
LOG_DIR = ROOT / "outputs_rmdx" / "logs"
RESULT_DIR = ROOT / "outputs_rmdx" / "rmdx_results"
STATE_JSON = RESULT_DIR / "multidetector_rmdx_preargmax_run_state.json"

DETECTOR_CACHE = {
    "groundingDINO": CACHE_ROOT / "det_compass_groundingDINO_100_0_full",
    "VLDet": CACHE_ROOT / "det_compass_VLDet_100_0_full",
    "CoDet": CACHE_ROOT / "det_compass_CoDet_100_0_full",
}


def slug(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def configs() -> list[dict[str, Any]]:
    rows = [{"mode": "visual_only", "lm": 0.0, "lr": 0.0}]
    for lm in [0.03, 0.05, 0.07, 0.10]:
        rows.append({"mode": "material_preargmax", "lm": lm, "lr": 0.0})
    for lm in [0.05, 0.10]:
        for lr in [0.25, 0.50]:
            rows.append({"mode": "material_reliability_preargmax", "lm": lm, "lr": lr})
    for row in rows:
        row["name"] = f"{row['mode']}_lm{slug(row['lm'])}_lr{slug(row['lr'])}"
    return rows


def load_baselines() -> dict[str, dict[str, float]]:
    baselines: dict[str, dict[str, float]] = {}
    with BASELINE_TABLE.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["included_in_main_multidetector_experiment"] != "True":
                continue
            baselines[row["detector"]] = {
                "AP": float(row["raxo_AP"]),
                "AP50": float(row["raxo_AP50"]),
                "AP75": float(row["raxo_AP75"]),
            }
    return baselines


def run_cmd(cmd: list[str], log_path: Path) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text("$ " + " ".join(cmd) + "\n" + proc.stdout)
    return {
        "passed": proc.returncode == 0,
        "returncode": proc.returncode,
        "log_path": str(log_path),
        "elapsed_seconds": round(time.time() - start, 3),
        "output": proc.stdout,
    }


def parse_eval(log_path: Path) -> dict[str, float | None]:
    text = log_path.read_text() if log_path.exists() else ""
    metrics = {"AP": None, "AP50": None, "AP75": None}
    for line in text.splitlines():
        if "Average Precision" not in line or "maxDets=100" not in line:
            continue
        match = re.search(r"=\s*([0-9.]+)\s*$", line)
        if not match:
            continue
        value = float(match.group(1))
        if "IoU=0.50:0.95" in line and metrics["AP"] is None:
            metrics["AP"] = value
        elif "IoU=0.50      " in line and metrics["AP50"] is None:
            metrics["AP50"] = value
        elif "IoU=0.75      " in line and metrics["AP75"] is None:
            metrics["AP75"] = value
    return metrics


def load_dets(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return json.load(f)


def summarize_detections(path: Path) -> dict[str, Any]:
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
        "changed_category_ratio": (changed / len(dets)) if dets else 0.0,
    }


def cache_summary(cache_dir: Path) -> dict[str, Any]:
    with (cache_dir / "cache_summary.json").open() as f:
        return json.load(f)


def visual_repro_status(ap: float | None, baseline_ap: float) -> str:
    if ap is None:
        return "FAIL_FOR_MAIN_TABLE"
    diff = abs(ap - baseline_ap)
    if diff <= 0.002:
        return "PASS"
    if diff <= 0.005:
        return "WARNING"
    return "FAIL_FOR_MAIN_TABLE"


def run_one(detector: str, config: dict[str, Any], baseline: dict[str, float]) -> dict[str, Any]:
    out_dir = OUT_ROOT / detector
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = DETECTOR_CACHE[detector]
    name = config["name"]
    output = out_dir / f"{detector}_{name}.json"
    uncert = out_dir / f"{detector}_{name}_uncertanty.json"
    row: dict[str, Any] = {
        "detector": detector,
        "mode": config["mode"],
        "lambda_material": config["lm"],
        "lambda_reliability": config["lr"],
        "raxo_baseline_AP": baseline["AP"],
        "raxo_baseline_AP50": baseline["AP50"],
        "raxo_baseline_AP75": baseline["AP75"],
        "cache_dir": str(cache_dir),
        "output_path": str(output),
        "uncertanty_output_path": str(uncert),
        "status": "PENDING",
        "failure_reason": "",
    }

    fusion = run_cmd(
        [
            "python3",
            "/workspace/RAXO/rmdx/cache_fusion.py",
            "--cache-dir",
            str(cache_dir),
            "--mode",
            config["mode"],
            "--lambda-material",
            str(config["lm"]),
            "--lambda-reliability",
            str(config["lr"]),
            "--output-path",
            str(output),
        ],
        LOG_DIR / f"batch21_{detector}_{name}_fusion.log",
    )
    row["fusion_log"] = fusion["log_path"]
    if not fusion["passed"]:
        row["status"] = "FAILED"
        row["failure_reason"] = "cache_fusion failed"
        return row
    raw = summarize_detections(output)
    row["detection_count_before_uncertanty"] = raw["count"]
    row["changed_category_count"] = raw["changed_category_count"]
    row["changed_category_ratio"] = raw["changed_category_ratio"]

    unc = run_cmd(
        ["python3", "raxo/uncertanty_estimation_fixed.py", "--dets", str(output)],
        LOG_DIR / f"batch21_{detector}_{name}_uncertanty.log",
    )
    row["uncertanty_log"] = unc["log_path"]
    if not unc["passed"]:
        row["status"] = "FAILED"
        row["failure_reason"] = "uncertanty failed"
        return row
    after = summarize_detections(uncert)
    row["detection_count_after_uncertanty"] = after["count"]

    eval_log = LOG_DIR / f"batch21_{detector}_{name}_eval.log"
    ev = run_cmd(
        ["python3", "raxo/cocoapi_2.py", "--cocoGt", str(ANNOTATION), "--cocoDt", str(uncert)],
        eval_log,
    )
    row["eval_log"] = ev["log_path"]
    if not ev["passed"]:
        row["status"] = "FAILED"
        row["failure_reason"] = "COCO eval failed"
        return row
    metrics = parse_eval(eval_log)
    row.update(metrics)
    row["delta_AP_vs_detector_raxo"] = (
        row["AP"] - baseline["AP"] if row.get("AP") is not None else None
    )
    row["delta_AP50_vs_detector_raxo"] = (
        row["AP50"] - baseline["AP50"] if row.get("AP50") is not None else None
    )
    row["delta_AP75_vs_detector_raxo"] = (
        row["AP75"] - baseline["AP75"] if row.get("AP75") is not None else None
    )
    row["status"] = "SUCCESS"
    return row


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    baselines = load_baselines()
    state: dict[str, Any] = {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "configs": configs(),
        "baselines": baselines,
        "cache_summaries": {},
        "rows": [],
    }
    STATE_JSON.write_text(json.dumps(state, indent=2) + "\n")

    visual_status_by_detector: dict[str, str] = {}
    for detector in ["groundingDINO", "VLDet", "CoDet"]:
        print(f"\n=== Batch21 detector {detector} ===", flush=True)
        state["cache_summaries"][detector] = cache_summary(DETECTOR_CACHE[detector])
        detector_rows = []
        for config in configs():
            print(f"[{detector}] {config['name']}", flush=True)
            row = run_one(detector, config, baselines[detector])
            detector_rows.append(row)
            if config["mode"] == "visual_only":
                visual_status_by_detector[detector] = visual_repro_status(
                    row.get("AP"), baselines[detector]["AP"]
                )
            row["visual_only_reproduction_status"] = visual_status_by_detector.get(detector, "")
            state["rows"].append(row)
            STATE_JSON.write_text(json.dumps(state, indent=2) + "\n")
        for row in detector_rows:
            row["visual_only_reproduction_status"] = visual_status_by_detector.get(detector, "")
        STATE_JSON.write_text(json.dumps(state, indent=2) + "\n")

    state["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    STATE_JSON.write_text(json.dumps(state, indent=2) + "\n")
    print(f"state={STATE_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
