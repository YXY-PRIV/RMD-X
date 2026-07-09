#!/usr/bin/env python3
"""Batch25: build multi-dataset RMD-X pre-argmax caches with task-level GPUs."""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import FIRST_COMPLETED, Future, wait
from pathlib import Path
from typing import Any


ROOT = Path("/workspace/RAXO")
if not ROOT.exists():
    ROOT = Path("/mnt/data/yxy/program/RAXO")

DATASET_ROOT = Path("/datasets/xray-datasets")
MODEL_ROOT = Path("/models/ICCV25_experimentation")
DETECTOR_ROOT = MODEL_ROOT / "initial_detections" / "groundingDINO"
PROTOTYPES = MODEL_ROOT / "known_prototypes_sam2.pt"
LOG_DIR = ROOT / "outputs_rmdx" / "logs"
RESULT_DIR = ROOT / "outputs_rmdx" / "rmdx_results"
CACHE_ROOT = ROOT / "outputs_rmdx" / "cache_multidataset"
LOCKED_DET_CACHE = ROOT / "outputs_rmdx" / "cache" / "det_compass_groundingDINO_100_0_full"
BASELINE_TABLE = RESULT_DIR / "multidataset_raxo_baseline_results_v2.csv"
READINESS_TABLE = RESULT_DIR / "multidataset_material_bank_readiness_v2.csv"
STATE_JSON = RESULT_DIR / "multidataset_cache_build_state.json"

DATASETS = {
    "DET-COMPASS": {
        "group": "main",
        "cache_dir": LOCKED_DET_CACHE,
        "material_bank": ROOT / "outputs_rmdx" / "material" / "material_colour_bank_support_full_bbox_only.json",
        "reuse_locked": True,
    },
    "PIXray": {
        "group": "main",
        "cache_dir": CACHE_ROOT / "PIXray_groundingDINO_100_0_full",
        "material_bank": ROOT / "outputs_rmdx" / "material_multidataset" / "banks" / "PIXray_material_bank_transfer_from_det_compass_support.json",
    },
    "CLCXray": {
        "group": "main",
        "cache_dir": CACHE_ROOT / "CLCXray_groundingDINO_100_0_full",
        "material_bank": ROOT / "outputs_rmdx" / "material_multidataset" / "banks" / "CLCXray_material_bank_transfer_from_det_compass_support.json",
    },
    "HiXray": {
        "group": "main",
        "cache_dir": CACHE_ROOT / "HiXray_groundingDINO_100_0_full",
        "material_bank": ROOT / "outputs_rmdx" / "material_multidataset" / "banks" / "HiXray_material_bank_transfer_from_det_compass_support.json",
    },
    "DvXray": {
        "group": "main",
        "cache_dir": CACHE_ROOT / "DvXray_groundingDINO_100_0_full",
        "material_bank": ROOT / "outputs_rmdx" / "material_multidataset" / "banks" / "DvXray_material_bank_transfer_from_det_compass_support.json",
    },
    "pidray": {
        "group": "exploratory",
        "cache_dir": CACHE_ROOT / "pidray_groundingDINO_100_0_full",
        "material_bank": ROOT / "outputs_rmdx" / "material_multidataset" / "banks" / "pidray_material_bank_transfer_from_det_compass_support.json",
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
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, title: str, rows: list[dict[str, Any]], fields: list[str]) -> None:
    lines = [f"# {title}", "", "| " + " | ".join(fields) + " |", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def annotation(dataset: str) -> Path:
    return DATASET_ROOT / dataset / "annotations" / "full_test.json"


def image_dir(dataset: str) -> Path:
    return DATASET_ROOT / dataset / "test"


def with_masks(dataset: str) -> Path:
    return DETECTOR_ROOT / f"coco_results.bbox_{dataset}_with_masks.json"


def light_json_check(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    if path.stat().st_size == 0:
        return False, "empty"
    with path.open("rb") as f:
        head = f.read(8192)
        try:
            f.seek(max(path.stat().st_size - 8192, 0))
            tail = f.read(8192)
        except OSError:
            tail = b""
    if b"annotations" not in head or b"images" not in head or b"categories" not in head:
        return False, "missing expected COCO keys in head"
    stripped = tail.strip()
    if not stripped.endswith(b"}") and not stripped.endswith(b"]"):
        return False, "tail does not look closed"
    return True, "light_check_passed"


def cache_valid(cache_dir: Path) -> tuple[bool, dict[str, Any]]:
    info: dict[str, Any] = {"cache_dir": str(cache_dir), "exists": cache_dir.is_dir(), "missing_files": [], "summary": None}
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
        valid = all(summary.get(key) is not None for key in ["proposal_count", "image_count", "prototype_count", "category_count", "feature_dim"])
        return bool(valid), info
    except Exception as exc:  # noqa: BLE001
        info["summary_error"] = f"{type(exc).__name__}: {exc}"
        return False, info


def torch_info() -> dict[str, Any]:
    proc = subprocess.run(
        [
            "python3",
            "-c",
            "import torch, json; print(json.dumps({'cuda_available': torch.cuda.is_available(), 'gpu_count': torch.cuda.device_count(), 'torch': torch.__version__, 'hub_dir': torch.hub.get_dir()}))",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        return {"error": proc.stdout, "cuda_available": False, "gpu_count": 0}


def preflight() -> dict[str, Any]:
    info = {
        "root": str(ROOT),
        "container_expected": "raxo_iccv25_gpu_mnt",
        "container_note": os.environ.get("BATCH25_CONTAINER_NOTE", ""),
        "dataset_root_exists": DATASET_ROOT.is_dir(),
        "prototypes_exists": PROTOTYPES.is_file(),
        "baseline_table_exists": BASELINE_TABLE.is_file(),
        "readiness_table_exists": READINESS_TABLE.is_file(),
        "cache_builder_exists": (ROOT / "rmdx" / "cache_builder.py").is_file(),
        "cache_fusion_exists": (ROOT / "rmdx" / "cache_fusion.py").is_file(),
        "output_dirs_writable": True,
        "torch": torch_info(),
        "disk_usage": shutil.disk_usage(str(ROOT)),
        "datasets": {},
    }
    for path in [CACHE_ROOT, LOG_DIR, RESULT_DIR]:
        path.mkdir(parents=True, exist_ok=True)
        info["output_dirs_writable"] = info["output_dirs_writable"] and os.access(path, os.W_OK)
    for dataset, cfg in DATASETS.items():
        ok_light, note_light = light_json_check(with_masks(dataset))
        valid_cache, cache_info = cache_valid(cfg["cache_dir"])
        ann_exists = annotation(dataset).is_file()
        img_dir = image_dir(dataset)
        image_count = len([p for p in img_dir.iterdir() if p.is_file()]) if img_dir.is_dir() else 0
        material_ok = False
        material_note = ""
        try:
            read_json(cfg["material_bank"])
            material_ok = True
        except Exception as exc:  # noqa: BLE001
            material_note = f"{type(exc).__name__}: {exc}"
        info["datasets"][dataset] = {
            "group": cfg["group"],
            "annotation": str(annotation(dataset)),
            "annotation_exists": ann_exists,
            "image_dir": str(img_dir),
            "image_dir_exists": img_dir.is_dir(),
            "test_image_count_light": image_count,
            "with_masks": str(with_masks(dataset)),
            "with_masks_exists": with_masks(dataset).is_file(),
            "with_masks_light_check": ok_light,
            "with_masks_light_note": note_light,
            "material_bank": str(cfg["material_bank"]),
            "material_bank_json_valid": material_ok,
            "material_bank_note": material_note,
            "cache_valid": valid_cache,
            "cache_info": cache_info,
        }
    return info


def build_dataset(dataset: str, gpu_id: int) -> dict[str, Any]:
    cfg = DATASETS[dataset]
    start = time.time()
    log_path = LOG_DIR / f"batch25_cache_build_{dataset}.log"
    valid, cache_info = cache_valid(cfg["cache_dir"])
    if valid:
        summary = cache_info.get("summary") or {}
        return row_from_summary(dataset, gpu_id, "SKIPPED_REUSE", True, time.time() - start, "", log_path, summary)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env.setdefault("RMDX_DINOV2_REPO", "/root/.cache/torch/hub/facebookresearch_dinov2_main")
    cmd = [
        "python3",
        str(ROOT / "rmdx" / "cache_builder.py"),
        "--json-res",
        str(with_masks(dataset)),
        "--image-path",
        str(image_dir(dataset)),
        "--prototypes",
        str(PROTOTYPES),
        "--material-bank",
        str(cfg["material_bank"]),
        "--output-cache-dir",
        str(cfg["cache_dir"]),
        "--nms",
        "0.8",
        "--dtype",
        "float32",
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + f"\nCUDA_VISIBLE_DEVICES={gpu_id}\n")
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        tail: list[str] = []
        for line in proc.stdout:
            print(f"[{dataset}][GPU{gpu_id}] {line}", end="", flush=True)
            log.write(line)
            tail.append(line)
            tail = tail[-80:]
        rc = proc.wait()
    valid_after, cache_after = cache_valid(cfg["cache_dir"])
    if rc == 0 and valid_after:
        return row_from_summary(dataset, gpu_id, "SUCCESS", False, time.time() - start, "", log_path, cache_after.get("summary") or {})
    reason = f"returncode={rc}; valid_after={valid_after}; tail={''.join(tail[-20:])}"
    return row_from_summary(dataset, gpu_id, "FAILED", False, time.time() - start, reason, log_path, cache_after.get("summary") or {})


def row_from_summary(
    dataset: str,
    gpu_id: int | str,
    status: str,
    reused: bool,
    elapsed: float,
    failure: str,
    log_path: Path,
    summary: dict[str, Any],
) -> dict[str, Any]:
    cfg = DATASETS[dataset]
    return {
        "dataset": dataset,
        "group": cfg["group"],
        "gpu_id": gpu_id,
        "cache_dir": str(cfg["cache_dir"]),
        "material_bank_path": str(cfg["material_bank"]),
        "with_masks_path": str(with_masks(dataset)),
        "proposal_count": summary.get("proposal_count", ""),
        "image_count": summary.get("image_count", ""),
        "category_count": summary.get("category_count", ""),
        "build_time_sec": round(elapsed, 3),
        "status": status,
        "failure_reason": failure,
        "log_path": str(log_path),
        "reused_existing_cache": reused,
    }


def run_group(datasets: list[str], gpu_ids: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pending = list(datasets)
    running: dict[Future[dict[str, Any]], tuple[str, int]] = {}
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=max(1, len(gpu_ids))) as executor:
        available = list(gpu_ids)
        while pending or running:
            while pending and available:
                dataset = pending.pop(0)
                gpu_id = available.pop(0)
                future = executor.submit(build_dataset, dataset, gpu_id)
                running[future] = (dataset, gpu_id)
            done, _ = wait(running.keys(), return_when=FIRST_COMPLETED)
            for future in done:
                dataset, gpu_id = running.pop(future)
                available.append(gpu_id)
                try:
                    row = future.result()
                except Exception as exc:  # noqa: BLE001
                    row = row_from_summary(dataset, gpu_id, "FAILED", False, 0.0, f"{type(exc).__name__}: {exc}", LOG_DIR / f"batch25_cache_build_{dataset}.log", {})
                rows.append(row)
                write_status(rows)
    return rows


FIELDS = [
    "dataset",
    "group",
    "gpu_id",
    "cache_dir",
    "material_bank_path",
    "with_masks_path",
    "proposal_count",
    "image_count",
    "category_count",
    "build_time_sec",
    "status",
    "failure_reason",
    "log_path",
    "reused_existing_cache",
]


def write_status(rows: list[dict[str, Any]]) -> None:
    ordered = sorted(rows, key=lambda r: list(DATASETS).index(r["dataset"]))
    write_csv(RESULT_DIR / "multidataset_cache_build_status.csv", ordered, FIELDS)
    write_md(RESULT_DIR / "multidataset_cache_build_status.md", "Batch25 多 Dataset cache build 状态", ordered, FIELDS)
    write_json(RESULT_DIR / "multidataset_cache_build_status.json", ordered)


def main() -> int:
    for path in [CACHE_ROOT, LOG_DIR, RESULT_DIR, ROOT / "docs_rmdx"]:
        path.mkdir(parents=True, exist_ok=True)
    pf = preflight()
    gpu_count = int(pf.get("torch", {}).get("gpu_count") or 0)
    gpu_ids = list(range(max(1, min(4, gpu_count))))
    degraded = gpu_count < 4
    write_json(RESULT_DIR / "multidataset_cache_build_preflight.json", {**pf, "gpu_ids_used": gpu_ids, "degraded_parallelism": degraded})
    rows: list[dict[str, Any]] = []

    det_valid, det_info = cache_valid(LOCKED_DET_CACHE)
    rows.append(row_from_summary("DET-COMPASS", "reuse", "SKIPPED_REUSE" if det_valid else "FAILED", det_valid, 0.0, "" if det_valid else "locked DET-COMPASS cache invalid/missing", LOG_DIR / "batch25_cache_build_DET-COMPASS.log", det_info.get("summary") or {}))
    write_status(rows)

    main_datasets = ["PIXray", "CLCXray", "HiXray", "DvXray"]
    rows.extend(run_group(main_datasets, gpu_ids))
    write_status(rows)

    main_failed = any(row["group"] == "main" and row["status"] == "FAILED" for row in rows)
    free = shutil.disk_usage(str(ROOT)).free
    if main_failed:
        rows.append(row_from_summary("pidray", "not_run", "NOT_RUN_EXPLORATORY", False, 0.0, "主结果 dataset 有失败，按优先级不运行 pidray exploratory", LOG_DIR / "batch25_cache_build_pidray.log", {}))
    elif free < 200 * 1024**3:
        rows.append(row_from_summary("pidray", "not_run", "NOT_RUN_EXPLORATORY", False, 0.0, "磁盘剩余空间不足 200GB，跳过 pidray exploratory", LOG_DIR / "batch25_cache_build_pidray.log", {}))
    else:
        rows.extend(run_group(["pidray"], gpu_ids[:1]))
    write_status(rows)

    state = {
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "preflight": pf,
        "gpu_ids_used": gpu_ids,
        "degraded_parallelism": degraded,
        "rows": rows,
    }
    write_json(STATE_JSON, state)
    print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
