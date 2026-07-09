#!/usr/bin/env python3
"""Run DET-COMPASS RAXO baselines for ready detectors.

Batch20 constraints:
- do not modify RAXO source code or official run scripts
- do not overwrite existing known_final outputs
- do not run any RMD-X method or cache build
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path("/workspace/RAXO")
LOG_DIR = ROOT / "outputs_rmdx" / "logs"
RESULT_DIR = ROOT / "outputs_rmdx" / "rmdx_results"
STATE_JSON = RESULT_DIR / "multidetector_raxo_baseline_run_state.json"
FILE_INDEX_JSON = RESULT_DIR / "multidetector_raxo_baseline_file_index.json"

ANNOTATION = Path("/datasets/xray-datasets/DET-COMPASS/annotations/full_test.json")
IMAGE_DIR = Path("/datasets/xray-datasets/DET-COMPASS/test")
PROTOTYPES = Path("/models/ICCV25_experimentation/known_prototypes_sam2.pt")
INITIAL_ROOT = Path("/models/ICCV25_experimentation/initial_detections")
DATASET = "DET-COMPASS"
NMS = "0.8"
NAME = "known_final"

DETECTORS = {
    "groundingDINO": {
        "included": True,
        "initial": INITIAL_ROOT / "groundingDINO" / "coco_results.bbox_DET-COMPASS.recovered_from_with_masks.json",
        "with_masks": INITIAL_ROOT / "groundingDINO" / "coco_results.bbox_DET-COMPASS_with_masks.json",
    },
    "VLDet": {
        "included": True,
        "initial": INITIAL_ROOT / "VLDet" / "coco_results.bbox_DET-COMPASS.json",
        "with_masks": INITIAL_ROOT / "VLDet" / "coco_results.bbox_DET-COMPASS_with_masks.json",
    },
    "CoDet": {
        "included": True,
        "initial": INITIAL_ROOT / "CoDet" / "coco_results.bbox_DET-COMPASS.json",
        "with_masks": INITIAL_ROOT / "CoDet" / "coco_results.bbox_DET-COMPASS_with_masks.json",
    },
    "detic": {
        "included": False,
        "exclusion_reason": "official DET-COMPASS Detic proposals not provided",
        "initial": INITIAL_ROOT / "detic" / "coco_results.bbox_DET-COMPASS.json",
        "with_masks": INITIAL_ROOT / "detic" / "coco_results.bbox_DET-COMPASS_with_masks.json",
    },
}


def print_header(title: str) -> None:
    print("\n" + "=" * 80, flush=True)
    print(title, flush=True)
    print("=" * 80, flush=True)


def known_final_path(with_masks: Path) -> Path:
    return Path(str(with_masks).replace(".json", f"_nms_{NMS}_our_method_{NAME}.json"))


def uncertainty_path(known_final: Path) -> Path:
    return Path(str(known_final).replace(".json", "_uncertanty.json"))


def json_count(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "exists": path.exists(),
        "valid": False,
        "top_type": None,
        "count": None,
        "images": None,
        "annotations": None,
        "categories": None,
        "size": path.stat().st_size if path.exists() else None,
        "error": None,
    }
    if not path.exists():
        return info
    try:
        with path.open("r") as f:
            data = json.load(f)
        info["valid"] = True
        info["top_type"] = type(data).__name__
        if isinstance(data, list):
            info["count"] = len(data)
        elif isinstance(data, dict):
            info["images"] = len(data.get("images", []))
            info["annotations"] = len(data.get("annotations", []))
            info["categories"] = len(data.get("categories", []))
            info["count"] = info["annotations"]
    except Exception as exc:  # noqa: BLE001
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def light_json_health(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "exists": path.exists(),
        "size": path.stat().st_size if path.exists() else None,
        "nonempty": False,
        "head_reasonable": False,
        "tail_reasonable": False,
        "contains_images_annotations_categories": False,
        "full_load": None,
    }
    if not path.exists() or path.stat().st_size == 0:
        return info
    info["nonempty"] = True
    with path.open("rb") as f:
        head = f.read(8192)
        f.seek(max(0, path.stat().st_size - 8192))
        tail = f.read(8192)
    head_text = head.decode("utf-8", errors="replace")
    tail_text = tail.decode("utf-8", errors="replace").strip()
    info["head_reasonable"] = head_text.lstrip().startswith(("{", "["))
    info["tail_reasonable"] = tail_text.endswith(("}", "]"))
    info["contains_images_annotations_categories"] = all(
        token in head_text for token in ['"images"', '"annotations"', '"categories"']
    )
    if path.stat().st_size < 600 * 1024 * 1024:
        info["full_load"] = json_count(path)
    else:
        info["full_load"] = {"skipped": True, "reason": "file larger than 600MB"}
    return info


def parse_ap(output: str) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {"AP": None, "AP50": None, "AP75": None}
    for line in output.splitlines():
        if "Average Precision" not in line or "maxDets=100" not in line:
            continue
        m = re.search(r"=\s*([0-9.]+)\s*$", line)
        if not m:
            continue
        value = float(m.group(1))
        if "IoU=0.50:0.95" in line and metrics["AP"] is None:
            metrics["AP"] = value
        elif "IoU=0.50      " in line and metrics["AP50"] is None:
            metrics["AP50"] = value
        elif "IoU=0.75      " in line and metrics["AP75"] is None:
            metrics["AP75"] = value
    return metrics


def is_dinov2_remote_failure(output: str) -> bool:
    terms = [
        "torch.hub.load",
        "github.com/facebookresearch/dinov2",
        "RemoteDisconnected",
        "urlopen",
        "HTTP",
        "SSL",
        "network",
    ]
    return any(term in output for term in terms)


def run_logged(cmd: list[str], log_path: Path, env: dict[str, str] | None = None) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"$ {' '.join(cmd)}", flush=True)
    start = time.time()
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=merged_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    chunks: list[str] = []
    with log_path.open("w") as log:
        log.write(f"$ {' '.join(cmd)}\n")
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
            chunks.append(line)
    rc = proc.wait()
    output = "".join(chunks)
    return {
        "returncode": rc,
        "passed": rc == 0,
        "output": output,
        "log_path": str(log_path),
        "elapsed_sec": round(time.time() - start, 2),
    }


def eval_coco(detector: str, label: str, det_path: Path) -> dict[str, Any]:
    log_path = LOG_DIR / f"batch20_{detector}_{label}_eval.log"
    result = run_logged(
        [
            "python3",
            "raxo/cocoapi_2.py",
            "--cocoGt",
            str(ANNOTATION),
            "--cocoDt",
            str(det_path),
        ],
        log_path,
    )
    metrics = parse_ap(result["output"])
    result.update(metrics)
    return result


def preflight() -> dict[str, Any]:
    print_header("Batch20 preflight")
    checks: dict[str, Any] = {
        "annotation_exists": ANNOTATION.exists(),
        "image_dir_exists": IMAGE_DIR.exists(),
        "prototypes_exists": PROTOTYPES.exists(),
        "annotation": str(ANNOTATION),
        "image_dir": str(IMAGE_DIR),
        "prototypes": str(PROTOTYPES),
        "detectors": {},
    }
    if ANNOTATION.exists():
        ann = json_count(ANNOTATION)
        checks["annotation_image_count"] = ann["images"]
        checks["annotation_count"] = ann["annotations"]
        checks["category_count"] = ann["categories"]
    if IMAGE_DIR.exists():
        checks["test_image_count"] = sum(1 for p in IMAGE_DIR.rglob("*") if p.is_file())
    for det, cfg in DETECTORS.items():
        kf = known_final_path(cfg["with_masks"])
        ku = uncertainty_path(kf)
        checks["detectors"][det] = {
            "included": cfg.get("included", False),
            "initial_exists": cfg["initial"].exists(),
            "with_masks_exists": cfg["with_masks"].exists(),
            "known_final_exists": kf.exists(),
            "known_final_uncertanty_exists": ku.exists(),
            "initial_health": json_count(cfg["initial"]) if cfg["initial"].exists() else None,
            "with_masks_health": light_json_health(cfg["with_masks"]) if cfg["with_masks"].exists() else None,
        }
        print(f"{det}: {checks['detectors'][det]}", flush=True)

    try:
        proc = subprocess.run(
            [
                "python3",
                "-c",
                (
                    "import torch, os; "
                    "print('torch', torch.__version__); "
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
        checks["torch_gpu_check"] = proc.stdout
        print(proc.stdout, flush=True)
    except Exception as exc:  # noqa: BLE001
        checks["torch_gpu_check"] = f"{type(exc).__name__}: {exc}"

    hub_candidates = []
    for base in [Path("/root/.cache/torch"), Path("/workspace"), Path("/models")]:
        if not base.exists():
            continue
        for path in base.rglob("*dinov2*"):
            hub_candidates.append(str(path))
            if len(hub_candidates) >= 30:
                break
        if len(hub_candidates) >= 30:
            break
    checks["dinov2_cache_candidates"] = hub_candidates
    print(f"DINOv2 cache candidates: {hub_candidates}", flush=True)
    return checks


def save_state(state: dict[str, Any]) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    STATE_JSON.write_text(json.dumps(state, indent=2) + "\n")
    file_index = {
        det: {
            "initial_detection_path": rec.get("initial_detection_path"),
            "with_masks_path": rec.get("with_masks_path"),
            "known_final_path": rec.get("known_final_path"),
            "known_final_uncertanty_path": rec.get("known_final_uncertanty_path"),
            "main_log": rec.get("main_log"),
            "initial_eval_log": rec.get("initial_eval_log"),
            "final_eval_log": rec.get("eval_log"),
        }
        for det, rec in state["detectors"].items()
    }
    FILE_INDEX_JSON.write_text(json.dumps(file_index, indent=2) + "\n")


def process_detector(detector: str, cfg: dict[str, Any]) -> dict[str, Any]:
    print_header(f"Detector: {detector}")
    rec: dict[str, Any] = {
        "detector": detector,
        "included_in_main_multidetector_experiment": cfg.get("included", False),
        "exclusion_reason": cfg.get("exclusion_reason", ""),
        "initial_detection_path": str(cfg["initial"]),
        "with_masks_path": str(cfg["with_masks"]),
        "known_final_path": str(known_final_path(cfg["with_masks"])),
        "known_final_uncertanty_path": str(uncertainty_path(known_final_path(cfg["with_masks"]))),
        "status": "PENDING",
        "failure_reason": "",
        "reused_existing_known_final": False,
        "generated_known_final": False,
        "generated_uncertanty": False,
    }
    if not cfg.get("included", False):
        rec["status"] = "EXCLUDED"
        return rec

    initial = cfg["initial"]
    with_masks = cfg["with_masks"]
    known = known_final_path(with_masks)
    unc = uncertainty_path(known)
    for label, path in [("initial detection", initial), ("with_masks", with_masks)]:
        if not path.exists():
            rec["status"] = "FAILED"
            rec["failure_reason"] = f"missing {label}: {path}"
            return rec

    initial_info = json_count(initial)
    rec["initial_detection_count"] = initial_info["count"]
    if not initial_info["valid"]:
        rec["status"] = "FAILED"
        rec["failure_reason"] = f"initial detection invalid: {initial_info.get('error')}"
        return rec

    wm_health = light_json_health(with_masks)
    rec["with_masks_annotation_count"] = (
        wm_health.get("full_load", {}).get("annotations")
        if isinstance(wm_health.get("full_load"), dict)
        else None
    )
    rec["with_masks_health"] = wm_health
    if not wm_health["exists"] or not wm_health["nonempty"] or not wm_health["head_reasonable"] or not wm_health["tail_reasonable"]:
        rec["status"] = "FAILED"
        rec["failure_reason"] = "with_masks failed light JSON health check"
        return rec

    print(f"Evaluating initial detection for {detector}", flush=True)
    initial_eval = eval_coco(detector, "initial", initial)
    rec["initial_eval_log"] = initial_eval["log_path"]
    rec["initial_AP"] = initial_eval["AP"]
    rec["initial_AP50"] = initial_eval["AP50"]
    rec["initial_AP75"] = initial_eval["AP75"]
    if not initial_eval["passed"]:
        rec["status"] = "FAILED"
        rec["failure_reason"] = "initial COCO evaluation failed"
        return rec

    known_info = json_count(known) if known.exists() else {"exists": False, "valid": False}
    unc_info = json_count(unc) if unc.exists() else {"exists": False, "valid": False}

    if unc.exists() and unc_info["valid"]:
        print(f"Reusing existing known_final_uncertanty for {detector}: {unc}", flush=True)
        rec["reused_existing_known_final"] = True
    else:
        if known.exists():
            if not known_info["valid"]:
                rec["status"] = "FAILED"
                rec["failure_reason"] = f"known_final exists but is invalid; refusing to overwrite: {known}"
                return rec
            print(f"Reusing existing known_final and generating uncertainty for {detector}: {known}", flush=True)
            rec["reused_existing_known_final"] = True
        else:
            print(f"Generating known_final for {detector}", flush=True)
            main_log = LOG_DIR / f"batch20_{detector}_raxo_known_branch.log"
            main = run_logged(
                [
                    "python3",
                    "raxo/main_with_masks_prop.py",
                    "--json_res",
                    str(with_masks),
                    "--image_path",
                    str(IMAGE_DIR),
                    "--prototypes",
                    str(PROTOTYPES),
                    "--nms",
                    NMS,
                    "--name",
                    NAME,
                    "--branch",
                    "known",
                ],
                main_log,
                env={"CUDA_VISIBLE_DEVICES": "0"},
            )
            rec["main_log"] = main["log_path"]
            if not main["passed"]:
                rec["status"] = "FAILED"
                rec["failure_reason"] = (
                    "DINOv2_REMOTE_LOADING_FAILURE"
                    if is_dinov2_remote_failure(main["output"])
                    else "RAXO known branch failed"
                )
                return rec
            rec["generated_known_final"] = True
            known_info = json_count(known)
            if not known_info["valid"]:
                rec["status"] = "FAILED"
                rec["failure_reason"] = f"known_final was not generated or invalid: {known}"
                return rec

        if not unc.exists():
            print(f"Generating uncertainty output for {detector}", flush=True)
            unc_log = LOG_DIR / f"batch20_{detector}_uncertanty.log"
            unc_run = run_logged(
                ["python3", "raxo/uncertanty_estimation_fixed.py", "--dets", str(known)],
                unc_log,
            )
            rec["uncertanty_log"] = unc_run["log_path"]
            if not unc_run["passed"]:
                rec["status"] = "FAILED"
                rec["failure_reason"] = "uncertanty_estimation_fixed.py failed"
                return rec
            rec["generated_uncertanty"] = True
        unc_info = json_count(unc)
        if not unc_info["valid"]:
            rec["status"] = "FAILED"
            rec["failure_reason"] = f"known_final_uncertanty missing or invalid: {unc}"
            return rec

    known_info = json_count(known) if known.exists() else known_info
    unc_info = json_count(unc) if unc.exists() else unc_info
    rec["known_final_count"] = known_info.get("count")
    rec["known_final_uncertanty_count"] = unc_info.get("count")

    print(f"Evaluating final RAXO known branch for {detector}", flush=True)
    final_eval = eval_coco(detector, "raxo_final", unc)
    rec["eval_log"] = final_eval["log_path"]
    rec["raxo_AP"] = final_eval["AP"]
    rec["raxo_AP50"] = final_eval["AP50"]
    rec["raxo_AP75"] = final_eval["AP75"]
    if not final_eval["passed"]:
        rec["status"] = "FAILED"
        rec["failure_reason"] = "final COCO evaluation failed"
        return rec

    rec["status"] = "SUCCESS"
    return rec


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    print_header("Batch20 multi-detector RAXO baseline")
    state: dict[str, Any] = {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "constraints": {
            "modified_raxo_source": False,
            "modified_official_run_scripts": False,
            "ran_rmdx": False,
            "used_full_test_material_bank": False,
        },
        "preflight": preflight(),
        "detectors": {},
    }
    save_state(state)

    for detector, cfg in DETECTORS.items():
        try:
            state["detectors"][detector] = process_detector(detector, cfg)
        except Exception as exc:  # noqa: BLE001
            state["detectors"][detector] = {
                "detector": detector,
                "included_in_main_multidetector_experiment": cfg.get("included", False),
                "status": "FAILED",
                "failure_reason": f"{type(exc).__name__}: {exc}",
                "initial_detection_path": str(cfg["initial"]),
                "with_masks_path": str(cfg["with_masks"]),
                "known_final_path": str(known_final_path(cfg["with_masks"])),
                "known_final_uncertanty_path": str(uncertainty_path(known_final_path(cfg["with_masks"]))),
            }
        save_state(state)

    state["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_state(state)
    print_header("Batch20 run complete")
    print(json.dumps({k: v.get("status") for k, v in state["detectors"].items()}, indent=2), flush=True)
    print(f"state={STATE_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
