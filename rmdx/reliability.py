"""Reliability-only RMD-X post-processing.

This module uses only existing RAXO result-list fields:
score, dist_cosine_inter, and dist_cosine_intra. It does not use material,
risk, image pixels, masks, or ground-truth labels.
"""

from __future__ import annotations

import math
from statistics import mean, pstdev
from typing import Any, Iterable


REQUIRED_FIELDS = {
    "image_id",
    "category_id",
    "bbox",
    "score",
    "dist_cosine_inter",
    "dist_cosine_intra",
}


def safe_float(x: Any) -> float:
    try:
        value = float(x)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Cannot convert {x!r} to float") from exc
    if not math.isfinite(value):
        raise ValueError(f"Non-finite float value: {x!r}")
    return value


def compute_zscore(values: Iterable[float]) -> dict[str, float]:
    vals = [safe_float(v) for v in values]
    if not vals:
        raise ValueError("Cannot compute z-score stats for empty values")
    mu = mean(vals)
    sigma = pstdev(vals)
    if sigma == 0:
        sigma = 1.0
    return {"mean": mu, "std": sigma}


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def validate_required_fields(dets: list[dict[str, Any]]) -> None:
    for idx, det in enumerate(dets):
        missing = REQUIRED_FIELDS - det.keys()
        if missing:
            raise KeyError(f"Detection {idx} missing required fields: {sorted(missing)}")


def compute_stats(dets: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    validate_required_fields(dets)
    return {
        "score": compute_zscore(det["score"] for det in dets),
        "dist_cosine_inter": compute_zscore(det["dist_cosine_inter"] for det in dets),
        "dist_cosine_intra": compute_zscore(det["dist_cosine_intra"] for det in dets),
    }


def _z(value: Any, stats: dict[str, float]) -> float:
    return (safe_float(value) - stats["mean"]) / stats["std"]


def compute_reliability(
    det: dict[str, Any],
    stats: dict[str, dict[str, float]],
    alpha: float,
    beta: float,
    gamma: float,
) -> float:
    z_inter = _z(det["dist_cosine_inter"], stats["dist_cosine_inter"])
    z_intra = _z(det["dist_cosine_intra"], stats["dist_cosine_intra"])
    z_score = _z(det["score"], stats["score"])
    return sigmoid(alpha * z_inter + beta * z_intra - gamma * z_score)


def _base_output_det(
    det: dict[str, Any],
    *,
    reliability: float,
    strategy: str,
    alpha: float,
    beta: float,
    gamma: float,
    replace_score: bool,
) -> dict[str, Any]:
    original_score = safe_float(det["score"])
    score_rmdx = original_score * reliability
    out = dict(det)
    out["score_rmdx"] = score_rmdx
    out["rmdx_reliability"] = reliability
    out["rmdx_strategy"] = strategy
    out["rmdx_original_score"] = original_score
    out["rmdx_original_category_id"] = det["category_id"]
    out["rmdx_alpha"] = alpha
    out["rmdx_beta"] = beta
    out["rmdx_gamma"] = gamma
    if replace_score:
        out["score"] = score_rmdx
    return out


def apply_margin_filter(dets: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    validate_required_fields(dets)
    output = []
    for det in dets:
        if safe_float(det["dist_cosine_inter"]) > float(threshold):
            out = dict(det)
            score = safe_float(det["score"])
            out["score_rmdx"] = score
            out["rmdx_reliability"] = 1.0
            out["rmdx_strategy"] = "margin_filter"
            out["rmdx_original_score"] = score
            out["rmdx_original_category_id"] = det["category_id"]
            out["rmdx_alpha"] = None
            out["rmdx_beta"] = None
            out["rmdx_gamma"] = None
            output.append(out)
    return output


def apply_reliability_rescore(
    dets: list[dict[str, Any]],
    alpha: float,
    beta: float,
    gamma: float,
    replace_score: bool,
) -> list[dict[str, Any]]:
    stats = compute_stats(dets)
    return [
        _base_output_det(
            det,
            reliability=compute_reliability(det, stats, alpha, beta, gamma),
            strategy="reliability_score_rescore",
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            replace_score=replace_score,
        )
        for det in dets
    ]


def apply_reliability_filter_and_rescore(
    dets: list[dict[str, Any]],
    threshold: float,
    alpha: float,
    beta: float,
    gamma: float,
) -> list[dict[str, Any]]:
    stats = compute_stats(dets)
    output = []
    for det in dets:
        reliability = compute_reliability(det, stats, alpha, beta, gamma)
        if reliability >= float(threshold):
            output.append(
                _base_output_det(
                    det,
                    reliability=reliability,
                    strategy="reliability_filter_and_rescore",
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                    replace_score=True,
                )
            )
    return output


def summarize_reliability(dets: list[dict[str, Any]]) -> dict[str, Any]:
    if not dets:
        return {
            "count": 0,
            "score_range": {"min": None, "max": None, "mean": None},
            "score_rmdx_range": {"min": None, "max": None, "mean": None},
            "reliability_range": {"min": None, "max": None, "mean": None},
        }

    def range_summary(field: str) -> dict[str, float | None]:
        vals = [safe_float(det[field]) for det in dets if field in det]
        if not vals:
            return {"min": None, "max": None, "mean": None}
        return {"min": min(vals), "max": max(vals), "mean": sum(vals) / len(vals)}

    return {
        "count": len(dets),
        "score_range": range_summary("score"),
        "score_rmdx_range": range_summary("score_rmdx"),
        "reliability_range": range_summary("rmdx_reliability"),
    }

