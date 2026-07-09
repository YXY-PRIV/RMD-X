"""Detector score calibration helpers for Batch54."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


WEIGHT_PRESETS = {
    "equal": {"groundingDINO": 1.0, "VLDet": 1.0, "CoDet": 1.0},
    "AP_prior_locked": {"groundingDINO": 1.0, "VLDet": 0.75, "CoDet": 0.70},
    "conservative": {"groundingDINO": 1.0, "VLDet": 0.5, "CoDet": 0.5},
    "groundingDINO_plus_VLDet": {"groundingDINO": 1.0, "VLDet": 0.75, "CoDet": 0.0},
    "groundingDINO_plus_CoDet": {"groundingDINO": 1.0, "VLDet": 0.0, "CoDet": 0.7},
}


def weights_for(name: str) -> dict[str, float]:
    return dict(WEIGHT_PRESETS[name])


def calibrate_scores(dets: list[dict[str, Any]], *, calibration: str, gamma: float = 1.0) -> list[dict[str, Any]]:
    rows = [dict(d) for d in dets]
    if calibration == "score_power":
        for row in rows:
            row["score"] = float(row.get("score", 0.0)) ** float(gamma)
    elif calibration == "minmax":
        by_detector: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_detector[str(row.get("detector", ""))].append(row)
        for group in by_detector.values():
            vals = [float(r.get("score", 0.0)) for r in group]
            lo, hi = min(vals), max(vals)
            for row in group:
                row["score"] = (float(row.get("score", 0.0)) - lo) / max(hi - lo, 1e-9)
    elif calibration == "zscore":
        by_detector = defaultdict(list)
        for row in rows:
            by_detector[str(row.get("detector", ""))].append(row)
        for group in by_detector.values():
            vals = [float(r.get("score", 0.0)) for r in group]
            mean = sum(vals) / len(vals)
            std = (sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)) ** 0.5
            for row in group:
                row["score"] = 1.0 / (1.0 + pow(2.718281828, -((float(row.get("score", 0.0)) - mean) / max(std, 1e-9))))
    return rows


def apply_detector_weights(dets: list[dict[str, Any]], weights: dict[str, float]) -> list[dict[str, Any]]:
    rows = []
    for det in dets:
        row = dict(det)
        weight = float(weights.get(str(row.get("detector", "")), 0.0))
        if weight <= 0:
            continue
        row["_detector_weight"] = weight
        row["score"] = float(row.get("score", 0.0)) * weight
        rows.append(row)
    return rows
