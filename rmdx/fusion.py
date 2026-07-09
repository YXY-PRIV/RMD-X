"""Material + reliability fusion helpers for RMD-X.

This module is a wrapper-level post-processor. It does not change bbox,
category_id, annotations, or RAXO internals.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from rmdx.material import load_material_bank, safe_float, score_detections_with_material
from rmdx.reliability import compute_stats, sigmoid


def compute_centered(values: list[float]) -> list[float]:
    if not values:
        return []
    mean_value = sum(values) / len(values)
    return [value - mean_value for value in values]


def _z(value: Any, stats: dict[str, float]) -> float:
    std = stats["std"] if stats["std"] != 0 else 1.0
    return (safe_float(value) - stats["mean"]) / std


def compute_corrected_reliability_scores(
    dets: list[dict[str, Any]],
    alpha: float,
    beta: float,
    gamma: float,
) -> list[float]:
    """Compute Batch11 reliability with the audited Batch9 direction.

    Formula:
    sigmoid(alpha*z(dist_cosine_inter) + beta*z(dist_cosine_intra) - gamma*z(score))
    """

    stats = compute_stats(dets)
    scores = []
    for det in dets:
        value = (
            alpha * _z(det["dist_cosine_inter"], stats["dist_cosine_inter"])
            + beta * _z(det["dist_cosine_intra"], stats["dist_cosine_intra"])
            - gamma * _z(det["score"], stats["score"])
        )
        scores.append(sigmoid(value))
    return scores


def attach_material_scores(
    dets: list[dict[str, Any]],
    material_bank_path: str,
    annotation: dict[str, Any],
    image_dir: str,
    mode: str = "bbox_only",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    bank = load_material_bank(material_bank_path)
    return score_detections_with_material(
        dets,
        annotation,
        image_dir,
        bank,
        mode=mode,
    )


def attach_reliability_scores(
    dets: list[dict[str, Any]],
    alpha: float,
    beta: float,
    gamma: float,
) -> list[dict[str, Any]]:
    reliability_scores = compute_corrected_reliability_scores(dets, alpha, beta, gamma)
    output = []
    for det, reliability in zip(dets, reliability_scores):
        out = dict(det)
        out["rmdx_reliability"] = reliability
        output.append(out)
    return output


def _base_det(
    det: dict[str, Any],
    *,
    strategy: str,
    score_rmdx: float,
    lambda_material: float | None,
    lambda_reliability: float | None,
    threshold_reliability: float | None = None,
    threshold_material: float | None = None,
    replace_score: bool = True,
) -> dict[str, Any]:
    out = dict(det)
    original_score = safe_float(det["score"])
    out["rmdx_original_score"] = original_score
    out["score_rmdx"] = max(0.0, float(score_rmdx))
    out["rmdx_fusion_strategy"] = strategy
    out["rmdx_lambda_material"] = lambda_material
    out["rmdx_lambda_reliability"] = lambda_reliability
    if threshold_reliability is not None:
        out["rmdx_reliability_threshold"] = threshold_reliability
    if threshold_material is not None:
        out["rmdx_material_threshold"] = threshold_material
    if replace_score:
        out["score"] = out["score_rmdx"]
    return out


def apply_material_only_reproduce(
    dets: list[dict[str, Any]],
    lambda_material: float = 0.3,
) -> list[dict[str, Any]]:
    centered_material = compute_centered(
        [safe_float(det["rmdx_material_score"]) for det in dets]
    )
    output = []
    for det, cm in zip(dets, centered_material):
        score = safe_float(det["score"])
        score_rmdx = score * (1.0 + lambda_material * cm)
        output.append(
            _base_det(
                det,
                strategy="material_only_reproduce",
                score_rmdx=score_rmdx,
                lambda_material=lambda_material,
                lambda_reliability=None,
            )
        )
    return output


def apply_reliability_gated_material(
    dets: list[dict[str, Any]],
    lambda_material: float,
    threshold_reliability: float,
) -> list[dict[str, Any]]:
    centered_material = compute_centered(
        [safe_float(det["rmdx_material_score"]) for det in dets]
    )
    output = []
    for det, cm in zip(dets, centered_material):
        score = safe_float(det["score"])
        reliability = safe_float(det["rmdx_reliability"])
        if reliability > threshold_reliability:
            score_rmdx = score * (1.0 + lambda_material * cm)
        else:
            score_rmdx = score
        output.append(
            _base_det(
                det,
                strategy="reliability_gated_material",
                score_rmdx=score_rmdx,
                lambda_material=lambda_material,
                lambda_reliability=None,
                threshold_reliability=threshold_reliability,
            )
        )
    return output


def apply_weighted_additive_fusion(
    dets: list[dict[str, Any]],
    lambda_material: float,
    lambda_reliability: float,
) -> list[dict[str, Any]]:
    centered_material = compute_centered(
        [safe_float(det["rmdx_material_score"]) for det in dets]
    )
    centered_reliability = compute_centered(
        [safe_float(det["rmdx_reliability"]) for det in dets]
    )
    output = []
    for det, cm, cr in zip(dets, centered_material, centered_reliability):
        score = safe_float(det["score"])
        score_rmdx = score * (1.0 + lambda_material * cm + lambda_reliability * cr)
        output.append(
            _base_det(
                det,
                strategy="weighted_additive_fusion",
                score_rmdx=score_rmdx,
                lambda_material=lambda_material,
                lambda_reliability=lambda_reliability,
            )
        )
    return output


def apply_material_filter_reliability_rescore(
    dets: list[dict[str, Any]],
    threshold_material: float,
    lambda_material: float,
    lambda_reliability: float,
) -> list[dict[str, Any]]:
    filtered = [
        det for det in dets if safe_float(det["rmdx_material_score"]) >= threshold_material
    ]
    centered_material = compute_centered(
        [safe_float(det["rmdx_material_score"]) for det in filtered]
    )
    centered_reliability = compute_centered(
        [safe_float(det["rmdx_reliability"]) for det in filtered]
    )
    output = []
    for det, cm, cr in zip(filtered, centered_material, centered_reliability):
        score = safe_float(det["score"])
        score_rmdx = score * (1.0 + lambda_material * cm + lambda_reliability * cr)
        output.append(
            _base_det(
                det,
                strategy="material_filter_reliability_rescore",
                score_rmdx=score_rmdx,
                lambda_material=lambda_material,
                lambda_reliability=lambda_reliability,
                threshold_material=threshold_material,
            )
        )
    return output


def summarize_fusion_result(
    input_dets: list[dict[str, Any]],
    output_dets: list[dict[str, Any]],
) -> dict[str, Any]:
    def range_summary(field: str, dets: list[dict[str, Any]]) -> dict[str, float | None]:
        vals = [safe_float(det[field]) for det in dets if field in det]
        if not vals:
            return {"min": None, "max": None, "mean": None}
        return {"min": min(vals), "max": max(vals), "mean": sum(vals) / len(vals)}

    return {
        "input_count": len(input_dets),
        "output_count": len(output_dets),
        "removed_count": len(input_dets) - len(output_dets),
        "removed_ratio": ((len(input_dets) - len(output_dets)) / len(input_dets))
        if input_dets
        else None,
        "image_count": len({int(det["image_id"]) for det in output_dets}),
        "category_count": len({int(det["category_id"]) for det in output_dets}),
        "category_top10": Counter(int(det["category_id"]) for det in output_dets).most_common(10),
        "score_range": range_summary("score", output_dets),
        "score_rmdx_range": range_summary("score_rmdx", output_dets),
        "material_score_range": range_summary("rmdx_material_score", output_dets),
        "reliability_range": range_summary("rmdx_reliability", output_dets),
    }
