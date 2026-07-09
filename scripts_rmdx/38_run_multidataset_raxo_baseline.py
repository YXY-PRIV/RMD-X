#!/usr/bin/env python3
"""Run or reuse groundingDINO RAXO known-branch baselines across datasets.

Batch23 constraints:
- do not modify RAXO source code or official run scripts
- do not modify annotations, original detections, or with_masks files
- do not run RMD-X or build RMD-X cache
- do not run mask extraction
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path("/workspace/RAXO")
LOG_DIR = ROOT / "outputs_rmdx" / "logs"
RESULT_DIR = ROOT / "outputs_rmdx" / "rmdx_results"
STATE_JSON = RESULT_DIR / "multidataset_raxo_baseline_run_state.json"
FILE_INDEX_JSON = RESULT_DIR / "multidataset_raxo_baseline_file_index.json"
FAILURES_JSON = RESULT_DIR / "multidataset_raxo_baseline_failures.json"

DATASET_ROOT = Path("/datasets/xray-datasets")
MODEL_ROOT = Path("/models/ICCV25_experimentation")
DETECTOR_DIR = MODEL_ROOT / "initial_detections" / "groundingDINO"
PROTOTYPES = MODEL_ROOT / "known_prototypes_sam2.pt"
DINOV2_CACHE = Path("/root/.cache/torch/hub/facebookresearch_dinov2_main")
DATASETS = ["DET-COMPASS", "PIXray", "pidray", "CLCXray", "HiXray", "DvXray"]
NMS = "0.8"
NAME = "known_final"


def print_header(title: str) -> None:
    print("\n" + "=" * 100, flush=True)
    print(title, flush=True)
    print("=" * 100, flush=True)


def annotation_path(dataset: str) -> Path:
    return DATASET_ROOT / dataset / "annotations" / "full_test.json"


def image_dir(dataset: str) -> Path:
    return DATASET_ROOT / dataset / "test"


def initial_path(dataset: str) -> Path:
    return DETECTOR_DIR / f"coco_results.bbox_{dataset}.json"


def recovered_path(dataset: str) -> Path:
    return DETECTOR_DIR / f"coco_results.bbox_{dataset}.recovered_from_with_masks.json"


def with_masks_path(dataset: str) -> Path:
    return DETECTOR_DIR / f"coco_results.bbox_{dataset}_with_masks.json"


def known_final_path(dataset: str) -> Path:
    return DETECTOR_DIR / f"coco_results.bbox_{dataset}_with_masks_nms_{NMS}_our_method_{NAME}.json"


def uncertainty_path(dataset: str) -> Path:
    return DETECTOR_DIR / f"coco_results.bbox_{dataset}_with_masks_nms_{NMS}_our_method_{NAME}_uncertanty.json"


def fmt_cmd(cmd: list[str]) -> str:
    return " ".join(cmd)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def json_health(path: Path, full_limit_bytes: int = 900 * 1024 * 1024) -> dict[str, Any]:
    info: dict[str, Any] = {
        "exists": path.exists(),
        "size": path.stat().st_size if path.exists() else None,
        "valid": False,
        "top_type": None,
        "count": None,
        "images": None,
        "annotations": None,
        "categories": None,
        "skipped_full_validation": False,
        "error": "",
    }
    if not path.exists():
        return info
    if path.stat().st_size > full_limit_bytes:
        with path.open("rb") as f:
            head = f.read(8192).decode("utf-8", errors="replace")
            f.seek(max(0, path.stat().st_size - 8192))
            tail = f.read(8192).decode("utf-8", errors="replace").strip()
        info["skipped_full_validation"] = True
        # Large with_masks JSON files are multi-GB and often written as a very
        # long line, so a tiny tail sample is not a reliable completeness check.
        # Treat a non-empty JSON-looking head as usable here; the official RAXO
        # loader will perform the real parse if the file is actually consumed.
        info["valid"] = head.lstrip().startswith(("{", "["))
        info["top_type"] = "large_json_light_check"
        info["error"] = "full json.load skipped because file is large"
        info["tail_sample_ends_like_json"] = tail.endswith(("}", "]"))
        return info
    try:
        data = load_json(path)
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


def list_count(path: Path) -> int | None:
    info = json_health(path, full_limit_bytes=20 * 1024 * 1024 * 1024)
    return info.get("count") if info.get("valid") else None


def annotation_info(dataset: str) -> dict[str, Any]:
    path = annotation_path(dataset)
    info = json_health(path, full_limit_bytes=200 * 1024 * 1024)
    if not info["valid"]:
        return info
    data = load_json(path)
    test = image_dir(dataset)
    names = [img["file_name"] for img in data.get("images", [])]
    missing = [name for name in names if not (test / Path(name).name).exists()]
    info.update(
        {
            "image_count": len(names),
            "annotation_count": len(data.get("annotations", [])),
            "category_count": len(data.get("categories", [])),
            "missing_test_images": len(missing),
            "test_image_status": "COMPLETE" if not missing and test.exists() else "PARTIAL",
        }
    )
    return info


def parse_ap(output: str) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {"AP": None, "AP50": None, "AP75": None}
    for line in output.splitlines():
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


def is_dinov2_remote_failure(output: str) -> bool:
    terms = [
        "torch.hub.load",
        "github.com/facebookresearch/dinov2",
        "RemoteDisconnected",
        "urlopen",
        "SSL",
        "HTTP",
        "network",
    ]
    return any(term in output for term in terms)


def run_logged(cmd: list[str], log_path: Path) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"$ {fmt_cmd(cmd)}", flush=True)
    start = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    chunks: list[str] = []
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {fmt_cmd(cmd)}\n")
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


def recover_initial_from_with_masks(dataset: str) -> dict[str, Any]:
    target = recovered_path(dataset)
    if target.exists() and json_health(target, full_limit_bytes=20 * 1024 * 1024 * 1024)["valid"]:
        return {"path": str(target), "created": False, "status": "REUSED"}
    source = with_masks_path(dataset)
    if not source.exists():
        return {"path": str(target), "created": False, "status": "FAILED", "reason": "with_masks missing"}
    try:
        data = load_json(source)
        annotations = data["annotations"] if isinstance(data, dict) else data
        recovered = []
        for ann in annotations:
            if not all(key in ann for key in ["image_id", "category_id", "bbox", "score"]):
                return {
                    "path": str(target),
                    "created": False,
                    "status": "FAILED",
                    "reason": "with_masks annotation missing image_id/category_id/bbox/score",
                }
            recovered.append(
                {
                    "image_id": ann["image_id"],
                    "category_id": ann["category_id"],
                    "bbox": ann["bbox"],
                    "score": ann["score"],
                }
            )
        with target.open("w", encoding="utf-8") as f:
            json.dump(recovered, f)
        return {"path": str(target), "created": True, "status": "CREATED", "count": len(recovered)}
    except Exception as exc:  # noqa: BLE001
        return {"path": str(target), "created": False, "status": "FAILED", "reason": f"{type(exc).__name__}: {exc}"}


def eval_coco(dataset: str, label: str, det_path: Path) -> dict[str, Any]:
    log_path = LOG_DIR / f"batch23_{dataset}_{label}_eval.log"
    result = run_logged(
        [
            "python3",
            "raxo/cocoapi_2.py",
            "--cocoGt",
            str(annotation_path(dataset)),
            "--cocoDt",
            str(det_path),
        ],
        log_path,
    )
    result.update(parse_ap(result["output"]))
    return result


def preflight() -> dict[str, Any]:
    print_header("Batch23 preflight")
    checks: dict[str, Any] = {
        "container_expected": "raxo_iccv25_gpu_mnt",
        "dataset_root": str(DATASET_ROOT),
        "model_root": str(MODEL_ROOT),
        "prototypes_exists": PROTOTYPES.exists(),
        "dinov2_cache_exists": DINOV2_CACHE.exists(),
        "datasets": {},
    }
    gpu = subprocess.run(
        ["python3", "-c", "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    checks["gpu_check_output"] = gpu.stdout.strip()
    for dataset in DATASETS:
        ann = annotation_info(dataset)
        init = initial_path(dataset)
        rec = recovered_path(dataset)
        wm = with_masks_path(dataset)
        kf = known_final_path(dataset)
        ku = uncertainty_path(dataset)
        init_health = json_health(init, full_limit_bytes=1200 * 1024 * 1024) if init.exists() else {"exists": False, "valid": False}
        rec_health = json_health(rec, full_limit_bytes=1200 * 1024 * 1024) if rec.exists() else {"exists": False, "valid": False}
        with_masks_health = json_health(wm, full_limit_bytes=900 * 1024 * 1024) if wm.exists() else {"exists": False, "valid": False}
        kf_health = json_health(kf, full_limit_bytes=1200 * 1024 * 1024) if kf.exists() else {"exists": False, "valid": False}
        ku_health = json_health(ku, full_limit_bytes=1200 * 1024 * 1024) if ku.exists() else {"exists": False, "valid": False}
        if ku_health.get("valid"):
            action = "EVALUATE_EXISTING_KNOWN_FINAL_UNCERTANTY"
        elif kf_health.get("valid"):
            action = "RUN_UNCERTANTY_THEN_EVALUATE"
        elif with_masks_health.get("valid"):
            action = "RUN_RAXO_KNOWN_BRANCH_THEN_UNCERTANTY_THEN_EVALUATE"
        elif init_health.get("valid") or rec_health.get("valid"):
            action = "SKIP_BLOCKED"
        else:
            action = "SKIP_BLOCKED"
        checks["datasets"][dataset] = {
            "annotation": ann,
            "test_dir_exists": image_dir(dataset).exists(),
            "initial_detection_status": init_health,
            "recovered_detection_status": rec_health,
            "with_masks_status": with_masks_health,
            "known_final_status": kf_health,
            "known_final_uncertanty_status": ku_health,
            "action_plan": action,
        }
        print(f"{dataset}: {json.dumps(checks['datasets'][dataset], ensure_ascii=False)[:2000]}", flush=True)
    return checks


def initial_detection_for_eval(dataset: str, rec: dict[str, Any]) -> tuple[Path | None, str, dict[str, Any] | None]:
    init = initial_path(dataset)
    rec_path = recovered_path(dataset)
    init_health = json_health(init, full_limit_bytes=1200 * 1024 * 1024) if init.exists() else {"valid": False}
    if init_health.get("valid"):
        return init, "initial_valid", None
    rec_health = json_health(rec_path, full_limit_bytes=1200 * 1024 * 1024) if rec_path.exists() else {"valid": False}
    if rec_health.get("valid"):
        return rec_path, "recovered_existing", None
    wm_health = rec.get("with_masks_status", {})
    if wm_health.get("valid"):
        recovery = recover_initial_from_with_masks(dataset)
        if recovery.get("status") in {"CREATED", "REUSED"}:
            return rec_path, "recovered_from_with_masks", recovery
        return None, f"recovery_failed: {recovery.get('reason', '')}", recovery
    return None, "initial_invalid_and_with_masks_unavailable", None


def process_dataset(dataset: str, pre: dict[str, Any]) -> dict[str, Any]:
    print_header(f"Dataset {dataset}")
    rec: dict[str, Any] = {
        "dataset": dataset,
        "image_count": pre["annotation"].get("image_count"),
        "category_count": pre["annotation"].get("category_count"),
        "annotation_count": pre["annotation"].get("annotation_count"),
        "test_image_status": pre["annotation"].get("test_image_status"),
        "initial_detection_path_used": "",
        "initial_detection_count": None,
        "initial_AP": None,
        "initial_AP50": None,
        "initial_AP75": None,
        "initial_eval_status": "NOT_RUN",
        "initial_eval_log": "",
        "with_masks_path": str(with_masks_path(dataset)),
        "with_masks_annotation_count": pre.get("with_masks_status", {}).get("annotations"),
        "known_final_path": str(known_final_path(dataset)),
        "known_final_count": None,
        "known_final_uncertanty_path": str(uncertainty_path(dataset)),
        "known_final_uncertanty_count": None,
        "raxo_AP": None,
        "raxo_AP50": None,
        "raxo_AP75": None,
        "delta_AP_vs_initial": None,
        "delta_AP50_vs_initial": None,
        "delta_AP75_vs_initial": None,
        "reused_existing_known_final": False,
        "reused_existing_known_final_uncertanty": False,
        "generated_known_final": False,
        "generated_uncertanty": False,
        "generated_recovered_detection": False,
        "status": "PENDING",
        "failure_reason": "",
        "main_log": "",
        "eval_log": "",
    }

    if not pre["annotation"].get("valid") or pre["annotation"].get("test_image_status") != "COMPLETE":
        rec["status"] = "FAILED"
        rec["failure_reason"] = "annotation/test image preflight failed"
        return rec
    if not PROTOTYPES.exists():
        rec["status"] = "FAILED"
        rec["failure_reason"] = "prototypes missing"
        return rec

    det_path, det_reason, recovery = initial_detection_for_eval(dataset, pre)
    if recovery and recovery.get("created"):
        rec["generated_recovered_detection"] = True
    if det_path is not None:
        rec["initial_detection_path_used"] = str(det_path)
        rec["initial_detection_count"] = list_count(det_path)
        initial_eval = eval_coco(dataset, "initial", det_path)
        rec["initial_eval_log"] = initial_eval["log_path"]
        if initial_eval["passed"]:
            rec["initial_eval_status"] = "SUCCESS"
            rec["initial_AP"] = initial_eval["AP"]
            rec["initial_AP50"] = initial_eval["AP50"]
            rec["initial_AP75"] = initial_eval["AP75"]
        else:
            rec["initial_eval_status"] = "FAILED"
            rec["failure_reason"] += f"initial eval failed; "
    else:
        rec["initial_eval_status"] = f"SKIPPED: {det_reason}"

    kf = known_final_path(dataset)
    ku = uncertainty_path(dataset)
    ku_health = json_health(ku, full_limit_bytes=1200 * 1024 * 1024) if ku.exists() else {"valid": False}
    kf_health = json_health(kf, full_limit_bytes=1200 * 1024 * 1024) if kf.exists() else {"valid": False}

    if ku_health.get("valid"):
        rec["reused_existing_known_final"] = kf.exists()
        rec["reused_existing_known_final_uncertanty"] = True
    elif kf_health.get("valid"):
        rec["reused_existing_known_final"] = True
        unc = run_logged(
            ["python3", "raxo/uncertanty_estimation_fixed.py", "--dets", str(kf)],
            LOG_DIR / f"batch23_{dataset}_uncertanty.log",
        )
        rec["main_log"] = unc["log_path"]
        if not unc["passed"]:
            rec["status"] = "FAILED"
            rec["failure_reason"] += "uncertanty_estimation_failed"
            return rec
        rec["generated_uncertanty"] = True
    else:
        wm = with_masks_path(dataset)
        wm_health = pre.get("with_masks_status", {})
        if not wm.exists() or not wm_health.get("valid"):
            rec["status"] = "FAILED"
            rec["failure_reason"] += "with_masks missing or invalid"
            return rec
        main = run_logged(
            [
                "python3",
                "raxo/main_with_masks_prop.py",
                "--json_res",
                str(wm),
                "--image_path",
                str(image_dir(dataset)),
                "--prototypes",
                str(PROTOTYPES),
                "--nms",
                NMS,
                "--name",
                NAME,
                "--branch",
                "known",
            ],
            LOG_DIR / f"batch23_{dataset}_known_branch.log",
        )
        rec["main_log"] = main["log_path"]
        if not main["passed"]:
            rec["status"] = "FAILED"
            rec["failure_reason"] = "DINOV2_REMOTE_LOADING_FAILURE" if is_dinov2_remote_failure(main["output"]) else "RAXO_KNOWN_BRANCH_FAILED"
            return rec
        rec["generated_known_final"] = True
        unc = run_logged(
            ["python3", "raxo/uncertanty_estimation_fixed.py", "--dets", str(kf)],
            LOG_DIR / f"batch23_{dataset}_uncertanty.log",
        )
        if rec["main_log"]:
            rec["main_log"] += f"; {unc['log_path']}"
        else:
            rec["main_log"] = unc["log_path"]
        if not unc["passed"]:
            rec["status"] = "FAILED"
            rec["failure_reason"] = "UNCERTANTY_ESTIMATION_FAILED"
            return rec
        rec["generated_uncertanty"] = True

    rec["known_final_count"] = list_count(kf) if kf.exists() else None
    rec["known_final_uncertanty_count"] = list_count(ku) if ku.exists() else None
    final_eval = eval_coco(dataset, "raxo_known_final_uncertanty", ku)
    rec["eval_log"] = final_eval["log_path"]
    if final_eval["passed"]:
        rec["raxo_AP"] = final_eval["AP"]
        rec["raxo_AP50"] = final_eval["AP50"]
        rec["raxo_AP75"] = final_eval["AP75"]
        if rec["initial_AP"] is not None:
            rec["delta_AP_vs_initial"] = rec["raxo_AP"] - rec["initial_AP"]
            rec["delta_AP50_vs_initial"] = rec["raxo_AP50"] - rec["initial_AP50"]
            rec["delta_AP75_vs_initial"] = rec["raxo_AP75"] - rec["initial_AP75"]
        rec["status"] = "SUCCESS"
    else:
        rec["status"] = "FAILED"
        rec["failure_reason"] += "final eval failed"
    return rec


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    checks = preflight()
    state: dict[str, Any] = {
        "batch": "Batch23",
        "container": "raxo_iccv25_gpu_mnt",
        "dataset_root": str(DATASET_ROOT),
        "detector": "groundingDINO",
        "prototypes": str(PROTOTYPES),
        "preflight": checks,
        "datasets": {},
    }
    file_index: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    for dataset in DATASETS:
        try:
            rec = process_dataset(dataset, checks["datasets"][dataset])
        except Exception as exc:  # noqa: BLE001
            rec = {
                "dataset": dataset,
                "status": "FAILED",
                "failure_reason": f"unhandled exception: {type(exc).__name__}: {exc}",
            }
        state["datasets"][dataset] = rec
        file_index[dataset] = {
            "annotation": str(annotation_path(dataset)),
            "image_dir": str(image_dir(dataset)),
            "initial_detection_path_used": rec.get("initial_detection_path_used", ""),
            "with_masks": str(with_masks_path(dataset)),
            "known_final": str(known_final_path(dataset)),
            "known_final_uncertanty": str(uncertainty_path(dataset)),
            "main_log": rec.get("main_log", ""),
            "initial_eval_log": rec.get("initial_eval_log", ""),
            "eval_log": rec.get("eval_log", ""),
        }
        if rec.get("status") != "SUCCESS":
            failures.append({"dataset": dataset, "failure_reason": rec.get("failure_reason", "")})
        STATE_JSON.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        FILE_INDEX_JSON.write_text(json.dumps(file_index, ensure_ascii=False, indent=2), encoding="utf-8")
        FAILURES_JSON.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    print_header("Batch23 finished")
    print(json.dumps(state["datasets"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
