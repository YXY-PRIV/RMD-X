"""Runner helpers for calibration with_masks generation."""

from __future__ import annotations

import subprocess
from pathlib import Path


def run_with_masks(gt_json: Path, proposals_json: Path, image_root: Path, log_path: Path) -> subprocess.CompletedProcess[str]:
    cmd = [
        "python3",
        "raxo/obtain_masks_res_inference.py",
        "--gt",
        str(gt_json),
        "--res",
        str(proposals_json),
        "--image_path",
        str(image_root),
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=24 * 3600, check=False)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("$ " + " ".join(cmd) + "\n\n" + proc.stdout + "\n" + proc.stderr, encoding="utf-8")
    return proc
