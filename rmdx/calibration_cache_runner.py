"""Runner helpers for calibration cache generation."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def run_cache_builder(
    with_masks_json: Path,
    image_root: Path,
    output_cache_dir: Path,
    prototypes: Path,
    material_bank: Path,
    log_path: Path,
    gpu_id: str = "0",
    max_images: int | None = None,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        "python3",
        "rmdx/cache_builder.py",
        "--json-res",
        str(with_masks_json),
        "--image-path",
        str(image_root),
        "--prototypes",
        str(prototypes),
        "--material-bank",
        str(material_bank),
        "--output-cache-dir",
        str(output_cache_dir),
        "--nms",
        "0.8",
    ]
    if max_images is not None:
        cmd.extend(["--max-images", str(max_images)])
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    proc = subprocess.run(cmd, env=env, text=True, capture_output=True, timeout=24 * 3600, check=False)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("$ CUDA_VISIBLE_DEVICES=" + gpu_id + " " + " ".join(cmd) + "\n\n" + proc.stdout + "\n" + proc.stderr, encoding="utf-8")
    return proc
