#!/usr/bin/env python3
"""Run Batch17 cache-fusion full grid from an existing full cache."""

from __future__ import annotations

import csv
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path("/workspace/RAXO")
CACHE_DIR = Path("/workspace/RAXO/outputs_rmdx/cache/det_compass_groundingDINO_100_0_full")
OUT_DIR = Path("/workspace/RAXO/outputs_rmdx/preargmax_cache")
RESULT_DIR = Path("/workspace/RAXO/outputs_rmdx/rmdx_results")
LOG_DIR = Path("/workspace/RAXO/outputs_rmdx/logs")
ANNOTATION = Path("/datasets/xray-datasets/DET-COMPASS/annotations/full_test.json")
REPORT = Path("/workspace/RAXO/docs_rmdx/batch17_feature_cache_report.md")

BASELINE = {"AP": 0.480, "AP50": 0.544, "AP75": 0.488}
DIRECT_BEST = {"AP": 0.501, "AP50": 0.561, "AP75": 0.515}

FIELDS = [
    "mode",
    "lambda_material",
    "lambda_reliability",
    "detection_count_before_uncertanty",
    "detection_count_after_uncertanty",
    "changed_category_count",
    "changed_category_ratio",
    "AP",
    "AP50",
    "AP75",
    "delta_AP_vs_baseline",
    "delta_AP_vs_direct_preargmax_best",
    "output_path",
    "uncertanty_output_path",
]


def slug(value: float) -> str:
    return str(value).replace(".", "p")


def configs() -> list[dict[str, Any]]:
    rows = [{"mode": "visual_only", "lm": 0.0, "lr": 0.0}]
    for lm in [0.03, 0.05, 0.07, 0.10]:
        rows.append({"mode": "material_preargmax", "lm": lm, "lr": 0.0})
    for lm in [0.05, 0.10]:
        for lr in [0.25, 0.50]:
            rows.append({"mode": "material_reliability_preargmax", "lm": lm, "lr": lr})
    for row in rows:
        row["name"] = f"cache_{row['mode']}_lm{slug(row['lm'])}_lr{slug(row['lr'])}"
    return rows


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
    return {"passed": proc.returncode == 0, "returncode": proc.returncode, "log_path": str(log_path)}


def parse_eval(log_path: Path) -> dict[str, Any]:
    text = log_path.read_text() if log_path.exists() else ""
    aps = re.findall(r"Average Precision.*=\s+(-?\d+\.\d+)", text)
    return {
        "AP": float(aps[0]) if len(aps) > 0 else None,
        "AP50": float(aps[1]) if len(aps) > 1 else None,
        "AP75": float(aps[2]) if len(aps) > 2 else None,
    }


def load_dets(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return json.load(f)


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
        "changed_category_ratio": (changed / len(dets)) if dets else None,
    }


def run_one(config: dict[str, Any]) -> tuple[dict[str, Any], float]:
    output = OUT_DIR / f"{config['name']}.json"
    uncert = OUT_DIR / f"{config['name']}_uncertanty.json"
    t0 = time.time()
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
        LOG_DIR / f"batch17_{config['name']}_fusion.log",
    )
    if not fusion["passed"]:
        raise RuntimeError(f"fusion failed for {config['name']}")
    raw = summarize(output)
    unc = run_cmd(
        ["python3", "raxo/uncertanty_estimation_fixed.py", "--dets", str(output)],
        LOG_DIR / f"batch17_{config['name']}_uncertanty.log",
    )
    if not unc["passed"]:
        raise RuntimeError(f"uncertanty failed for {config['name']}")
    after = summarize(uncert)
    eval_log = LOG_DIR / f"batch17_{config['name']}_eval.log"
    ev = run_cmd(
        ["python3", "raxo/cocoapi_2.py", "--cocoGt", str(ANNOTATION), "--cocoDt", str(uncert)],
        eval_log,
    )
    if not ev["passed"]:
        raise RuntimeError(f"eval failed for {config['name']}")
    metrics = parse_eval(eval_log)
    ap = metrics["AP"]
    row = {
        "mode": config["mode"],
        "lambda_material": config["lm"],
        "lambda_reliability": config["lr"],
        "detection_count_before_uncertanty": raw["count"],
        "detection_count_after_uncertanty": after["count"],
        "changed_category_count": raw["changed_category_count"],
        "changed_category_ratio": raw["changed_category_ratio"],
        "AP": ap,
        "AP50": metrics["AP50"],
        "AP75": metrics["AP75"],
        "delta_AP_vs_baseline": (ap - BASELINE["AP"]) if ap is not None else None,
        "delta_AP_vs_direct_preargmax_best": (ap - DIRECT_BEST["AP"]) if ap is not None else None,
        "output_path": str(output),
        "uncertanty_output_path": str(uncert),
    }
    return row, time.time() - t0


def write_tables(rows: list[dict[str, Any]]) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULT_DIR / "cache_fusion_grid_results.csv"
    md_path = RESULT_DIR / "cache_fusion_grid_results.md"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with md_path.open("w") as f:
        f.write("| " + " | ".join(FIELDS) + " |\n")
        f.write("|" + "|".join(["---"] * len(FIELDS)) + "|\n")
        for row in rows:
            f.write("| " + " | ".join(str(row.get(field, "")) for field in FIELDS) + " |\n")


def write_report(rows: list[dict[str, Any]], timings: list[float], total_time: float) -> None:
    cache_summary = json.load((CACHE_DIR / "cache_summary.json").open())
    cache_files = [(path, path.stat().st_size) for path in sorted(CACHE_DIR.glob("*")) if path.is_file()]
    visual = next((row for row in rows if row["mode"] == "visual_only"), None)
    best_lm005 = next(
        (
            row
            for row in rows
            if row["mode"] == "material_preargmax"
            and abs(float(row["lambda_material"]) - 0.05) < 1e-9
        ),
        None,
    )
    warnings = []
    if visual and abs(float(visual["AP"]) - BASELINE["AP"]) > 0.002:
        warnings.append("cache visual_only AP differs from baseline by more than 0.002")
    if best_lm005 and abs(float(best_lm005["AP"]) - DIRECT_BEST["AP"]) > 0.002:
        warnings.append("cache lm=0.05 AP differs from direct pre-argmax best by more than 0.002")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Batch17 Feature Cache Report",
        "",
        "## A. 新增文件",
        "",
        "- `/workspace/RAXO/rmdx/cache_builder.py`",
        "- `/workspace/RAXO/rmdx/cache_fusion.py`",
        "- `/workspace/RAXO/scripts_rmdx/13_build_preargmax_cache_smoke.py`",
        "- `/workspace/RAXO/scripts_rmdx/14_run_cache_fusion_smoke.py`",
        "- `/workspace/RAXO/scripts_rmdx/15_build_preargmax_cache_full.py`",
        "- `/workspace/RAXO/scripts_rmdx/16_run_cache_fusion_grid.py`",
        "",
        "## B. cache schema",
        "",
        "见 `/workspace/RAXO/docs_rmdx/batch17_cache_schema.md`。",
        "",
        "## C. smoke cache build 是否通过",
        "",
        "`True`。见 `/workspace/RAXO/docs_rmdx/batch17_cache_smoke_build_report.md`。",
        "",
        "## D. smoke cache fusion 是否通过",
        "",
        "`True`。见 `/workspace/RAXO/docs_rmdx/batch17_cache_fusion_smoke_report.md`。",
        "",
        "## E. full cache 是否构建",
        "",
        "`True`。",
        "",
        "## F. full cache 文件大小",
        "",
        "| file | size_bytes |",
        "|---|---:|",
    ]
    for path, size in cache_files:
        lines.append(f"| `{path}` | {size} |")
    lines += [
        "",
        "## G. cache visual_only 与 baseline 对比",
        "",
        f"- baseline: `{BASELINE}`",
        f"- cache visual_only: `{visual}`",
        "",
        "## H. cache lm=0.05 与 direct pre-argmax 对比",
        "",
        f"- direct pre-argmax best: `{DIRECT_BEST}`",
        f"- cache lm=0.05: `{best_lm005}`",
        f"- warnings: `{warnings}`",
        "",
        "## I. 加速效果",
        "",
        f"- direct preargmax 单配置耗时: `not_measured_in_log`",
        f"- cache build 耗时: `{cache_summary.get('build_time_seconds')}` seconds",
        f"- cache fusion 单配置耗时均值: `{sum(timings) / len(timings):.3f}` seconds",
        f"- grid 总耗时: `{total_time:.3f}` seconds",
        "",
        "## J. 是否可以支撑后续多 detector / 多 dataset",
        "",
        "`True`，前提是为每个 detector/dataset/setting 单独构建 cache，并在后续 grid 中复用 cache fusion。",
        "",
        "## K. 是否修改 RAXO 原始代码",
        "",
        "否。未修改 `raxo/*.py`、官方 run 脚本、baseline 输出或 annotation。",
        "",
        "## L. 下一步建议",
        "",
        "进入 Batch18 cache 复现锁定；若锁定通过，再进入 Batch19 多 detector。",
    ]
    REPORT.write_text("\n".join(lines) + "\n")


def main() -> int:
    if not (CACHE_DIR / "cache_summary.json").exists():
        raise SystemExit(f"missing full cache: {CACHE_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    timings = []
    t0 = time.time()
    for config in configs():
        print(f"[Batch17] cache fusion {config['name']}", flush=True)
        row, elapsed = run_one(config)
        rows.append(row)
        timings.append(elapsed)
        write_tables(rows)
        print(row, flush=True)
    total = time.time() - t0
    write_tables(rows)
    write_report(rows, timings, total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
