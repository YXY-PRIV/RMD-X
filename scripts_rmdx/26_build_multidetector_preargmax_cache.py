#!/usr/bin/env python3
"""Build or reuse detector-specific RMD-X pre-argmax caches for Batch21."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path("/workspace/RAXO")
LOG_DIR = ROOT / "outputs_rmdx" / "logs"
RESULT_DIR = ROOT / "outputs_rmdx" / "rmdx_results"
REPORT_JSON = RESULT_DIR / "multidetector_rmdx_cache_build_state.json"

ANNOTATION = Path("/datasets/xray-datasets/DET-COMPASS/annotations/full_test.json")
IMAGE_DIR = Path("/datasets/xray-datasets/DET-COMPASS/test")
PROTOTYPES = Path("/models/ICCV25_experimentation/known_prototypes_sam2.pt")
MATERIAL_BANK = ROOT / "outputs_rmdx" / "material" / "material_colour_bank_support_full_bbox_only.json"
BASELINE_TABLE = ROOT / "outputs_rmdx" / "rmdx_results" / "multidetector_raxo_baseline_results.csv"
CACHE_ROOT = ROOT / "outputs_rmdx" / "cache"

DETECTORS = {
    "groundingDINO": {
        "with_masks": Path("/models/ICCV25_experimentation/initial_detections/groundingDINO/coco_results.bbox_DET-COMPASS_with_masks.json"),
        "cache_dir": CACHE_ROOT / "det_compass_groundingDINO_100_0_full",
        "locked": True,
    },
    "VLDet": {
        "with_masks": Path("/models/ICCV25_experimentation/initial_detections/VLDet/coco_results.bbox_DET-COMPASS_with_masks.json"),
        "cache_dir": CACHE_ROOT / "det_compass_VLDet_100_0_full",
        "locked": False,
    },
    "CoDet": {
        "with_masks": Path("/models/ICCV25_experimentation/initial_detections/CoDet/coco_results.bbox_DET-COMPASS_with_masks.json"),
        "cache_dir": CACHE_ROOT / "det_compass_CoDet_100_0_full",
        "locked": False,
    },
}

REQUIRED_CACHE_FILES = [
    "cache_summary.json",
    "proposal_meta.jsonl",
    "prototype_meta.json",
    "category_meta.json",
    "visual_similarity.pt",
    "material_class_similarity.pt",
]


def read_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def cache_valid(cache_dir: Path) -> tuple[bool, dict[str, Any]]:
    info: dict[str, Any] = {
        "cache_dir": str(cache_dir),
        "exists": cache_dir.is_dir(),
        "missing_files": [],
        "summary": None,
        "valid": False,
    }
    if not cache_dir.is_dir():
        info["missing_files"] = REQUIRED_CACHE_FILES
        return False, info
    missing = [name for name in REQUIRED_CACHE_FILES if not (cache_dir / name).is_file()]
    info["missing_files"] = missing
    if missing:
        return False, info
    try:
        summary = read_json(cache_dir / "cache_summary.json")
        info["summary"] = summary
        info["valid"] = all(
            summary.get(key) is not None
            for key in ["proposal_count", "image_count", "prototype_count", "category_count", "feature_dim"]
        )
    except Exception as exc:  # noqa: BLE001
        info["summary_error"] = f"{type(exc).__name__}: {exc}"
        return False, info
    return bool(info["valid"]), info


def run_cmd(cmd: list[str], log_path: Path) -> dict[str, Any]:
    print(f"$ {' '.join(cmd)}", flush=True)
    start = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=os.environ.copy(),
    )
    chunks: list[str] = []
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        log.write(f"$ {' '.join(cmd)}\n")
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
            chunks.append(line)
    rc = proc.wait()
    return {
        "passed": rc == 0,
        "returncode": rc,
        "log_path": str(log_path),
        "elapsed_seconds": round(time.time() - start, 3),
        "output_tail": "".join(chunks[-80:]),
    }


def preflight() -> dict[str, Any]:
    checks: dict[str, Any] = {
        "annotation_exists": ANNOTATION.exists(),
        "image_dir_exists": IMAGE_DIR.is_dir(),
        "prototypes_exists": PROTOTYPES.exists(),
        "support_material_bank_exists": MATERIAL_BANK.exists(),
        "baseline_table_exists": BASELINE_TABLE.exists(),
        "rmdx_cache_builder_exists": (ROOT / "rmdx" / "cache_builder.py").exists(),
        "rmdx_cache_fusion_exists": (ROOT / "rmdx" / "cache_fusion.py").exists(),
        "detectors": {},
    }
    for det, cfg in DETECTORS.items():
        valid, cache_info = cache_valid(cfg["cache_dir"])
        checks["detectors"][det] = {
            "with_masks": str(cfg["with_masks"]),
            "with_masks_exists": cfg["with_masks"].exists(),
            "cache_valid": valid,
            "cache_info": cache_info,
        }
    proc = subprocess.run(
        [
            "python3",
            "-c",
            (
                "import torch; "
                "print('cuda_available', torch.cuda.is_available()); "
                "print('gpu_count', torch.cuda.device_count()); "
                "print('torch_hub_dir', torch.hub.get_dir())"
            ),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    checks["torch_check"] = proc.stdout
    hub = Path("/root/.cache/torch/hub/facebookresearch_dinov2_main")
    checks["dinov2_local_cache_exists"] = hub.is_dir()
    checks["dinov2_local_cache_path"] = str(hub)
    print(json.dumps(checks, indent=2), flush=True)
    return checks


def build_detector_cache(detector: str, cfg: dict[str, Any]) -> dict[str, Any]:
    valid, info = cache_valid(cfg["cache_dir"])
    if detector == "groundingDINO":
        status = "REUSED_LOCKED_CACHE" if valid else "FAILED_LOCKED_CACHE_MISSING"
        return {
            "detector": detector,
            "status": status,
            "reused_existing": valid,
            "built": False,
            "cache_dir": str(cfg["cache_dir"]),
            "cache_info": info,
            "summary": info.get("summary"),
        }
    if valid:
        return {
            "detector": detector,
            "status": "REUSED_EXISTING_CACHE",
            "reused_existing": True,
            "built": False,
            "cache_dir": str(cfg["cache_dir"]),
            "cache_info": info,
            "summary": info.get("summary"),
        }

    cmd = [
        "python3",
        "/workspace/RAXO/rmdx/cache_builder.py",
        "--json-res",
        str(cfg["with_masks"]),
        "--image-path",
        str(IMAGE_DIR),
        "--prototypes",
        str(PROTOTYPES),
        "--material-bank",
        str(MATERIAL_BANK),
        "--output-cache-dir",
        str(cfg["cache_dir"]),
        "--nms",
        "0.8",
        "--dtype",
        "float32",
    ]
    result = run_cmd(cmd, LOG_DIR / f"batch21_{detector}_cache_build.log")
    valid_after, info_after = cache_valid(cfg["cache_dir"])
    return {
        "detector": detector,
        "status": "BUILT_SUCCESS" if result["passed"] and valid_after else "FAILED_BUILD",
        "reused_existing": False,
        "built": result["passed"] and valid_after,
        "cache_dir": str(cfg["cache_dir"]),
        "build_log": result["log_path"],
        "build_elapsed_seconds": result["elapsed_seconds"],
        "build_returncode": result["returncode"],
        "cache_info": info_after,
        "summary": info_after.get("summary"),
        "failure_tail": "" if result["passed"] and valid_after else result.get("output_tail", ""),
    }


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    print("Batch21 cache build preflight", flush=True)
    state: dict[str, Any] = {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "preflight": preflight(),
        "detectors": {},
        "constraints": {
            "uses_support_material_bank": True,
            "uses_full_test_material_bank": False,
            "runs_ap_eval": False,
            "runs_rmdx_fusion": False,
            "runs_mask_extraction": False,
        },
    }
    REPORT_JSON.write_text(json.dumps(state, indent=2) + "\n")
    for detector, cfg in DETECTORS.items():
        print(f"\n=== cache build/reuse: {detector} ===", flush=True)
        state["detectors"][detector] = build_detector_cache(detector, cfg)
        REPORT_JSON.write_text(json.dumps(state, indent=2) + "\n")
    state["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    REPORT_JSON.write_text(json.dumps(state, indent=2) + "\n")
    print(json.dumps({k: v["status"] for k, v in state["detectors"].items()}, indent=2), flush=True)
    return 0 if all(not v["status"].startswith("FAILED") for v in state["detectors"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
