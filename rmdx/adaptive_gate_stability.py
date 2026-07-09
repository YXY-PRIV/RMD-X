"""Fixed adaptive-gate candidates for Batch41.1 stability checks.

The candidates here are rule-based and use only inference-time cache signals.
They do not read GT labels, Batch40 oracle outputs, or train any model.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import torch

from rmdx.adaptive_gating import GateConfig, quantile_threshold


@dataclass(frozen=True)
class FixedGateCandidate:
    candidate_id: str
    gate_name: str
    lambda_material: float
    threshold_type: str
    visual_margin_threshold_abs: float | None = None
    visual_margin_quantile: float | None = None
    material_margin_quantile: float | None = None
    sample_threshold: int | None = None
    selected_from_grid: bool = False
    uses_gt: bool = False
    fair_method_candidate: bool = True
    notes: str = ""

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


def fixed_gate_candidates() -> list[FixedGateCandidate]:
    """Return the locked Batch41.1 candidate registry."""

    return [
        FixedGateCandidate(
            candidate_id="candidate_A_det_compass_abs_visual_margin",
            gate_name="visual_margin_gate",
            lambda_material=0.05,
            threshold_type="absolute",
            visual_margin_threshold_abs=0.034,
            selected_from_grid=True,
            notes="DET-COMPASS Batch41 grid best; tuned candidate, not a preregistered universal gate.",
        ),
        FixedGateCandidate(
            candidate_id="candidate_B_dataset_q70_visual_margin",
            gate_name="visual_margin_gate",
            lambda_material=0.05,
            threshold_type="dataset_quantile",
            visual_margin_quantile=0.70,
            notes="Unsupervised per-target visual-margin q70 threshold.",
        ),
        FixedGateCandidate(
            candidate_id="candidate_C_conservative_hybrid_safe",
            gate_name="hybrid_safe_gate",
            lambda_material=0.05,
            threshold_type="dataset_quantile",
            visual_margin_quantile=0.70,
            material_margin_quantile=0.50,
            sample_threshold=5,
            notes="Safe hybrid gate intended to reduce material interference on fragile targets.",
        ),
        FixedGateCandidate(
            candidate_id="candidate_D_low_lambda_q70_visual_margin",
            gate_name="visual_margin_gate",
            lambda_material=0.03,
            threshold_type="dataset_quantile",
            visual_margin_quantile=0.70,
            notes="Lower-material-weight robust visual-margin gate.",
        ),
    ]


def instantiate_candidate(
    candidate: FixedGateCandidate,
    features: dict[str, torch.Tensor],
) -> tuple[GateConfig, dict[str, Any]]:
    """Convert a fixed candidate into a concrete GateConfig for one target."""

    visual_thr = candidate.visual_margin_threshold_abs
    material_thr = None
    resolved: dict[str, Any] = {
        "threshold_type": candidate.threshold_type,
        "visual_margin_quantile": candidate.visual_margin_quantile,
        "material_margin_quantile": candidate.material_margin_quantile,
        "sample_threshold": candidate.sample_threshold,
    }
    if candidate.threshold_type == "dataset_quantile":
        if candidate.visual_margin_quantile is not None:
            visual_thr = quantile_threshold(features["visual_margin"], float(candidate.visual_margin_quantile))
        if candidate.material_margin_quantile is not None:
            material_thr = quantile_threshold(features["material_margin"], float(candidate.material_margin_quantile))
    resolved["threshold_value"] = visual_thr
    resolved["material_threshold_value"] = material_thr
    cfg = GateConfig(
        gate_name=candidate.gate_name,
        lambda_material=float(candidate.lambda_material),
        margin_threshold=visual_thr,
        material_threshold=material_thr,
        sample_threshold=candidate.sample_threshold,
        agree_boost=1.0,
        notes=candidate.notes,
    )
    return cfg, resolved


def slug(value: float | int | str | None) -> str:
    if value is None:
        return "none"
    if isinstance(value, str):
        return value.replace("/", "_").replace(" ", "_")
    return f"{float(value):.3f}".rstrip("0").rstrip(".").replace(".", "p")


def md_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "无记录。"
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(h, "")).replace("|", "/") for h in headers) + " |")
    return "\n".join(lines)
