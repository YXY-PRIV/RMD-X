"""Inference helpers for XDA-GDINO scripts."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from rmdx.xda_gdino_env import get_assets


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "rmdx/groundingdino_calibration_infer.py"


def read_status_csv(path: Path) -> dict[str, str]:
    import csv

    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else {}


def run_wrapper(coco_json: Path, image_root: Path, output_json: Path, log_csv: Path, *, limit_images: int | None, device: str = "cuda", box_threshold: float = 0.10, text_threshold: float = 0.10) -> tuple[bool, str, dict[str, Any]]:
    assets = get_assets()
    cmd = [
        sys.executable,
        str(WRAPPER),
        "--config-path",
        str(assets["config"]),
        "--checkpoint-path",
        str(assets["checkpoint"]),
        "--coco-json",
        str(coco_json),
        "--image-root",
        str(image_root),
        "--output-json",
        str(output_json),
        "--device",
        device,
        "--box-threshold",
        str(box_threshold),
        "--text-threshold",
        str(text_threshold),
        "--prompt-mode",
        "all_categories",
        "--max-dets-per-image",
        "300",
        "--log-path",
        str(log_csv),
        "--source-path",
        str(assets["source"]),
    ]
    if limit_images:
        cmd.extend(["--limit-images", str(limit_images)])
    env = os.environ.copy()
    env["PYTHONPATH"] = str(assets["source"]) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=24 * 3600, env=env, check=False)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    log_txt = output_json.with_suffix(".log")
    log_txt.write_text("$ " + " ".join(cmd) + "\n\n" + proc.stdout + "\n" + proc.stderr, encoding="utf-8")
    data: dict[str, Any] = {}
    for line in reversed(proc.stdout.splitlines()):
        if line.strip().startswith("{"):
            try:
                data = json.loads(line)
                break
            except Exception:
                pass
    return proc.returncode == 0, proc.stdout + proc.stderr, data


def parse_coco_ap(text: str) -> tuple[str, str, str]:
    vals = re.findall(r"Average Precision\s+\(AP\).*?=\s+([0-9.\-]+)", text)
    return (vals[0], vals[1], vals[2]) if len(vals) >= 3 else ("", "", "")

