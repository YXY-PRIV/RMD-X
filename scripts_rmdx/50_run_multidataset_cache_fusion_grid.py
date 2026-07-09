#!/usr/bin/env python3
"""Batch25: run multi-dataset RMD-X cache fusion grid and evaluation."""

from __future__ import annotations

import csv
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path("/workspace/RAXO")
if not ROOT.exists():
    ROOT = Path("/mnt/data/yxy/program/RAXO")

DATASET_ROOT = Path("/datasets/xray-datasets")
OUT_ROOT = ROOT / "outputs_rmdx" / "preargmax_cache_multidataset_groundingDINO"
LOG_DIR = ROOT / "outputs_rmdx" / "logs"
RESULT_DIR = ROOT / "outputs_rmdx" / "rmdx_results"
CACHE_ROOT = ROOT / "outputs_rmdx" / "cache_multidataset"
DET_CACHE = ROOT / "outputs_rmdx" / "cache" / "det_compass_groundingDINO_100_0_full"
BASELINE_TABLE = RESULT_DIR / "multidataset_raxo_baseline_results_v2.csv"
READINESS_TABLE = RESULT_DIR / "multidataset_material_bank_readiness_v2.csv"
CACHE_STATUS = RESULT_DIR / "multidataset_cache_build_status.csv"
STATE_JSON = RESULT_DIR / "multidataset_rmdx_preargmax_run_state.json"

DATASETS = ["DET-COMPASS", "PIXray", "CLCXray", "HiXray", "DvXray", "pidray"]


def slug(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def configs() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [{"mode": "visual_only", "lm": 0.0, "lr": 0.0}]
    for lm in [0.03, 0.05, 0.07, 0.10]:
        rows.append({"mode": "material_preargmax", "lm": lm, "lr": 0.0})
    for lm in [0.05, 0.10]:
        for lr in [0.25, 0.50]:
            rows.append({"mode": "material_reliability_preargmax", "lm": lm, "lr": lr})
    for row in rows:
        row["name"] = f"{row['mode']}_lm{slug(row['lm'])}_lr{slug(row['lr'])}"
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def annotation(dataset: str) -> Path:
    return DATASET_ROOT / dataset / "annotations" / "full_test.json"


def cache_dir(dataset: str) -> Path:
    if dataset == "DET-COMPASS":
        return DET_CACHE
    return CACHE_ROOT / f"{dataset}_groundingDINO_100_0_full"


def output_path(dataset: str, cfg: dict[str, Any]) -> Path:
    return OUT_ROOT / dataset / f"{dataset}_groundingDINO_{cfg['name']}.json"


def uncert_path(path: Path) -> Path:
    return path.with_name(path.stem + "_uncertanty.json")


def load_baselines() -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for row in read_csv(BASELINE_TABLE):
        dataset = row["dataset"]
        out[dataset] = {
            "AP": float(row["raxo_AP"]),
            "AP50": float(row["raxo_AP50"]),
            "AP75": float(row["raxo_AP75"]),
        }
    return out


def readiness_by_dataset() -> dict[str, dict[str, str]]:
    return {row["dataset"]: row for row in read_csv(READINESS_TABLE)}


def cache_ready() -> dict[str, dict[str, str]]:
    ready: dict[str, dict[str, str]] = {}
    for row in read_csv(CACHE_STATUS):
        if row.get("status") in {"SUCCESS", "SKIPPED_REUSE"}:
            ready[row["dataset"]] = row
    return ready


def run_cmd(cmd: list[str], log_path: Path) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    log_path.write_text("$ " + " ".join(cmd) + "\n" + proc.stdout, encoding="utf-8")
    return {
        "passed": proc.returncode == 0,
        "returncode": proc.returncode,
        "log_path": str(log_path),
        "elapsed_seconds": round(time.time() - start, 3),
        "output": proc.stdout,
    }


def parse_eval(log_path: Path) -> dict[str, float | None]:
    text = log_path.read_text(encoding="utf-8", errors="ignore") if log_path.exists() else ""
    metrics: dict[str, float | None] = {"AP": None, "AP50": None, "AP75": None}
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
    with path.open("r", encoding="utf-8") as f:
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
        "changed_category_ratio": round(changed / len(dets), 6) if dets else 0.0,
    }


def visual_repro_status(ap: float | None, baseline_ap: float) -> str:
    if ap is None:
        return "FAIL_FOR_MAIN_TABLE"
    diff = abs(ap - baseline_ap)
    if diff <= 0.002:
        return "PASS"
    if diff <= 0.005:
        return "WARNING"
    return "FAIL_FOR_MAIN_TABLE"


def run_one(dataset: str, cfg: dict[str, Any], baseline: dict[str, float], readiness: dict[str, str]) -> dict[str, Any]:
    OUT_ROOT.joinpath(dataset).mkdir(parents=True, exist_ok=True)
    out = output_path(dataset, cfg)
    unc = uncert_path(out)
    eval_log = LOG_DIR / f"batch25_{dataset}_{cfg['name']}_eval.log"
    row: dict[str, Any] = {
        "dataset": dataset,
        "result_group": "exploratory" if dataset == "pidray" else "main",
        "material_bank_status": readiness.get("final_status", ""),
        "mode": cfg["mode"],
        "lambda_material": cfg["lm"],
        "lambda_reliability": cfg["lr"],
        "raxo_baseline_AP": baseline["AP"],
        "raxo_baseline_AP50": baseline["AP50"],
        "raxo_baseline_AP75": baseline["AP75"],
        "cache_dir": str(cache_dir(dataset)),
        "material_bank_path": readiness.get("bank_path", ""),
        "output_path": str(out),
        "uncertanty_output_path": str(unc),
        "status": "PENDING",
        "failure_reason": "",
    }

    if out.exists() and unc.exists() and eval_log.exists():
        raw = summarize_detections(out)
        after = summarize_detections(unc)
        metrics = parse_eval(eval_log)
        row.update(metrics)
        row["detection_count_before_uncertanty"] = raw["count"]
        row["detection_count_after_uncertanty"] = after["count"]
        row["changed_category_count"] = raw["changed_category_count"]
        row["changed_category_ratio"] = raw["changed_category_ratio"]
        row["fusion_log"] = str(LOG_DIR / f"batch25_{dataset}_{cfg['name']}_fusion.log")
        row["uncertanty_log"] = str(LOG_DIR / f"batch25_{dataset}_{cfg['name']}_uncertanty.log")
        row["eval_log"] = str(eval_log)
        row["status"] = "SUCCESS_REUSED"
        return finish_metrics(row, baseline)

    fusion = run_cmd(
        [
            "python3",
            str(ROOT / "rmdx" / "cache_fusion.py"),
            "--cache-dir",
            str(cache_dir(dataset)),
            "--mode",
            cfg["mode"],
            "--lambda-material",
            str(cfg["lm"]),
            "--lambda-reliability",
            str(cfg["lr"]),
            "--output-path",
            str(out),
        ],
        LOG_DIR / f"batch25_{dataset}_{cfg['name']}_fusion.log",
    )
    row["fusion_log"] = fusion["log_path"]
    if not fusion["passed"]:
        row["status"] = "FAILED"
        row["failure_reason"] = "cache_fusion failed"
        return row
    raw = summarize_detections(out)
    row["detection_count_before_uncertanty"] = raw["count"]
    row["changed_category_count"] = raw["changed_category_count"]
    row["changed_category_ratio"] = raw["changed_category_ratio"]

    unc_res = run_cmd(
        ["python3", str(ROOT / "raxo" / "uncertanty_estimation_fixed.py"), "--dets", str(out)],
        LOG_DIR / f"batch25_{dataset}_{cfg['name']}_uncertanty.log",
    )
    row["uncertanty_log"] = unc_res["log_path"]
    if not unc_res["passed"] or not unc.exists():
        row["status"] = "FAILED"
        row["failure_reason"] = "uncertanty failed"
        return row
    after = summarize_detections(unc)
    row["detection_count_after_uncertanty"] = after["count"]

    ev = run_cmd(
        ["python3", str(ROOT / "raxo" / "cocoapi_2.py"), "--cocoGt", str(annotation(dataset)), "--cocoDt", str(unc)],
        eval_log,
    )
    row["eval_log"] = ev["log_path"]
    if not ev["passed"]:
        row["status"] = "FAILED"
        row["failure_reason"] = "COCO eval failed"
        return row
    row.update(parse_eval(eval_log))
    row["status"] = "SUCCESS"
    return finish_metrics(row, baseline)


def finish_metrics(row: dict[str, Any], baseline: dict[str, float]) -> dict[str, Any]:
    for key in ["AP", "AP50", "AP75"]:
        row.setdefault(key, None)
    row["delta_AP_vs_raxo"] = round(row["AP"] - baseline["AP"], 6) if row.get("AP") is not None else None
    row["delta_AP50_vs_raxo"] = round(row["AP50"] - baseline["AP50"], 6) if row.get("AP50") is not None else None
    row["delta_AP75_vs_raxo"] = round(row["AP75"] - baseline["AP75"], 6) if row.get("AP75") is not None else None
    return row


def main() -> int:
    for path in [OUT_ROOT, LOG_DIR, RESULT_DIR]:
        path.mkdir(parents=True, exist_ok=True)
    baselines = load_baselines()
    readiness = readiness_by_dataset()
    ready_cache = cache_ready()
    state: dict[str, Any] = {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "configs": configs(),
        "rows": [],
        "skipped": [],
    }
    write_json(STATE_JSON, state)
    visual_status_by_dataset: dict[str, str] = {}
    for dataset in DATASETS:
        if dataset not in ready_cache:
            state["skipped"].append({"dataset": dataset, "reason": "cache not ready"})
            write_json(STATE_JSON, state)
            continue
        if dataset not in baselines:
            state["skipped"].append({"dataset": dataset, "reason": "baseline missing"})
            write_json(STATE_JSON, state)
            continue
        for cfg in configs():
            print(f"\n=== {dataset} {cfg['name']} ===", flush=True)
            row = run_one(dataset, cfg, baselines[dataset], readiness.get(dataset, {}))
            if cfg["mode"] == "visual_only":
                visual_status_by_dataset[dataset] = visual_repro_status(row.get("AP"), baselines[dataset]["AP"])
            row["visual_only_reproduction_status"] = visual_status_by_dataset.get(dataset, "")
            state["rows"].append(row)
            write_json(STATE_JSON, state)
    state["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write_json(STATE_JSON, state)
    print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
