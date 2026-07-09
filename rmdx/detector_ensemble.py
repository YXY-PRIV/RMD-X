"""Detector ensemble orchestration for Batch54."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rmdx.ensemble_calibration import apply_detector_weights, calibrate_scores, weights_for
from rmdx.ensemble_nms import class_aware_nms, soft_nms
from rmdx.weighted_box_fusion import weighted_box_fusion


def load_detections(path: str | Path, detector: str) -> list[dict[str, Any]]:
    data = json.load(open(path))
    out = []
    for det in data:
        row = dict(det)
        row["detector"] = detector
        row.setdefault("dist_cosine_inter", 1.0)
        row.setdefault("dist_cosine_intra", 0.0)
        out.append(row)
    return out


def clean_for_coco(dets: list[dict[str, Any]], *, candidate_id: str, method: str, input_detectors: list[str]) -> list[dict[str, Any]]:
    keep = []
    for det in dets:
        row = {
            "image_id": int(det["image_id"]),
            "category_id": int(det["category_id"]),
            "bbox": [float(v) for v in det["bbox"]],
            "score": float(det.get("score", 0.0)),
            "dist_cosine_inter": float(det.get("dist_cosine_inter", 1.0)),
            "dist_cosine_intra": float(det.get("dist_cosine_intra", 0.0)),
            "rmdx_mode": "detector_ensemble_rmdx",
            "ensemble_candidate_id": candidate_id,
            "ensemble_method": method,
            "ensemble_input_detectors": input_detectors,
        }
        if "ensemble_cluster_size" in det:
            row["ensemble_cluster_size"] = det["ensemble_cluster_size"]
        keep.append(row)
    return keep


def run_ensemble(inputs: dict[str, list[dict[str, Any]]], candidate: dict[str, str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    detectors = [d for d in candidate["input_detectors"].split(",") if d]
    weights = weights_for(candidate["detector_weights"])
    all_dets = []
    for detector in detectors:
        for det in inputs[detector]:
            row = dict(det)
            row["detector"] = detector
            all_dets.append(row)
    before = len(all_dets)
    calibration = candidate["score_calibration"]
    gamma = float(candidate.get("gamma") or 1.0)
    calibrated = calibrate_scores(all_dets, calibration=calibration, gamma=gamma)
    weighted = apply_detector_weights(calibrated, weights)
    method = candidate["ensemble_method"]
    iou_thr = float(candidate["iou_thr"])
    score_thr = float(candidate["score_thr"])
    max_dets = int(candidate["max_dets_per_image"])
    if method == "weighted_box_fusion":
        fused = weighted_box_fusion(weighted, iou_thr=iou_thr, score_thr=score_thr, max_dets_per_image=max_dets)
    elif method == "class_aware_nms":
        fused = class_aware_nms(weighted, iou_thr=iou_thr, score_thr=score_thr, max_dets_per_image=max_dets)
    elif method == "soft_nms":
        fused = soft_nms(weighted, iou_thr=iou_thr, sigma=0.5, score_thr=score_thr, max_dets_per_image=max_dets)
    elif method == "reference":
        fused = weighted
    else:
        raise ValueError(f"Unsupported ensemble method: {method}")
    cleaned = clean_for_coco(fused, candidate_id=candidate["candidate_id"], method=method, input_detectors=detectors)
    stats = {
        "detection_count_before": before,
        "detection_count_after": len(cleaned),
        "input_detectors": detectors,
        "detector_weights": weights,
    }
    return cleaned, stats
