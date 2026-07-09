#!/usr/bin/env python3
"""Build the Batch17 5-image pre-argmax cache smoke."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


REPO_ROOT = Path("/workspace/RAXO")
CACHE_DIR = Path("/workspace/RAXO/outputs_rmdx/cache/det_compass_groundingDINO_100_0_smoke")
REPORT = Path("/workspace/RAXO/docs_rmdx/batch17_cache_smoke_build_report.md")


def run() -> dict:
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
        "--max-images",
        "5",
        "--dtype",
        "float32",
    ]
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (CACHE_DIR / "build_stdout.log").write_text(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout)
    return json.load((CACHE_DIR / "cache_summary.json").open())


def file_table() -> list[tuple[str, int]]:
    rows = []
    for path in sorted(CACHE_DIR.glob("*")):
        if path.is_file():
            rows.append((str(path), path.stat().st_size))
    return rows


def write_report(summary: dict) -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Batch17 Cache Smoke Build Report",
        "",
        f"- cache dir: `{CACHE_DIR}`",
        f"- proposal count: `{summary.get('proposal_count')}`",
        f"- image count: `{summary.get('image_count')}`",
        f"- prototype count: `{summary.get('prototype_count')}`",
        f"- category count: `{summary.get('category_count')}`",
        f"- feature shape: `[{summary.get('proposal_count')}, {summary.get('feature_dim')}]`",
        f"- dtype: `{summary.get('dtype')}`",
        f"- build time seconds: `{summary.get('build_time_seconds'):.3f}`",
        f"- success: `True`",
        "",
        "## Cache Files",
        "",
        "| file | size_bytes |",
        "|---|---:|",
    ]
    for path, size in file_table():
        lines.append(f"| `{path}` | {size} |")
    REPORT.write_text("\n".join(lines) + "\n")


def main() -> int:
    summary = run()
    write_report(summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
