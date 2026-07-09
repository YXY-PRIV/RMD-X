#!/usr/bin/env python3
"""Build the full DET-COMPASS pre-argmax cache for Batch17."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


REPO_ROOT = Path("/workspace/RAXO")
CACHE_DIR = Path("/workspace/RAXO/outputs_rmdx/cache/det_compass_groundingDINO_100_0_full")
REPORT = Path("/workspace/RAXO/docs_rmdx/batch17_cache_full_build_report.md")


def main() -> int:
    cmd = [
        "python3",
        "/workspace/RAXO/rmdx/cache_builder.py",
        "--json-res",
        "/models/ICCV25_experimentation/initial_detections/groundingDINO/coco_results.bbox_DET-COMPASS_with_masks.json",
        "--image-path",
        "/datasets/xray-datasets/DET-COMPASS/test",
        "--prototypes",
        "/models/ICCV25_experimentation/known_prototypes_sam2.pt",
        "--material-bank",
        "/workspace/RAXO/outputs_rmdx/material/material_colour_bank_support_full_bbox_only.json",
        "--output-cache-dir",
        str(CACHE_DIR),
        "--nms",
        "0.8",
        "--dtype",
        "float32",
    ]
    print("[Batch17] full cache build command:", " ".join(cmd), flush=True)
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / "build_stdout.log").write_text(proc.stdout)
    if proc.returncode != 0:
        print(proc.stdout, flush=True)
        return proc.returncode

    summary = json.load((CACHE_DIR / "cache_summary.json").open())
    files = []
    for path in sorted(CACHE_DIR.glob("*")):
        if path.is_file():
            files.append((str(path), path.stat().st_size))
    lines = [
        "# Batch17 Full Cache Build Report",
        "",
        f"- cache dir: `{CACHE_DIR}`",
        f"- proposal count: `{summary.get('proposal_count')}`",
        f"- image count: `{summary.get('image_count')}`",
        f"- prototype count: `{summary.get('prototype_count')}`",
        f"- category count: `{summary.get('category_count')}`",
        f"- feature dim: `{summary.get('feature_dim')}`",
        f"- dtype: `{summary.get('dtype')}`",
        f"- build time seconds: `{summary.get('build_time_seconds'):.3f}`",
        "",
        "| file | size_bytes |",
        "|---|---:|",
    ]
    for path, size in files:
        lines.append(f"| `{path}` | {size} |")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
