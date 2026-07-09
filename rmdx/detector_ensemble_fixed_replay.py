"""Fixed detector ensemble replay helpers for Batch55."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rmdx.detector_ensemble import load_detections, run_ensemble


def load_lock(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as f:
        return json.load(f)


def replay_from_lock(lock: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    inputs = {}
    for detector in lock["input_detectors"]:
        inputs[detector] = load_detections(lock["input_json_paths"][detector], detector)
    candidate = {
        "candidate_id": lock["candidate_id"],
        "branch_type": lock["branch_type"],
        "input_detectors": ",".join(lock["input_detectors"]),
        "ensemble_method": lock["ensemble_method"],
        "detector_weights": lock["detector_weights"],
        "score_calibration": lock["score_calibration"],
        "gamma": str(lock.get("gamma", "")),
        "iou_thr": str(lock["iou_thr"]),
        "score_thr": str(lock["score_thr"]),
        "max_dets_per_image": str(lock["max_dets_per_image"]),
    }
    return run_ensemble(inputs, candidate)
