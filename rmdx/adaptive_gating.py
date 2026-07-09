"""Fair adaptive material gating utilities.

These gates use only inference-time visual/material similarity signals plus
support/train material bank statistics. They do not read GT labels or Batch40
oracle results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class GateConfig:
    gate_name: str
    lambda_material: float = 0.0
    margin_threshold: float | None = None
    material_threshold: float | None = None
    sample_threshold: int | None = None
    agree_boost: float = 1.0
    notes: str = ""


def top2_scores(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return top1 value/index and top2 value/index for a 2D tensor."""

    if x.shape[1] == 1:
        vals = x[:, :1]
        idx = torch.zeros((x.shape[0], 1), dtype=torch.long)
        return vals[:, 0], idx[:, 0], torch.zeros_like(vals[:, 0]), idx[:, 0]
    vals, idx = torch.topk(x, k=2, dim=1)
    return vals[:, 0], idx[:, 0], vals[:, 1], idx[:, 1]


def entropy_from_scores(scores: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Compute entropy after softmax normalization."""

    probs = torch.softmax(scores.float(), dim=dim)
    return -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=dim)


def quantile_threshold(values: torch.Tensor, q: float) -> float:
    """Compute a CPU scalar quantile threshold."""

    if values.numel() == 0:
        return 0.0
    return float(torch.quantile(values.float(), float(q)).item())


def compute_gate_weights(
    features: dict[str, torch.Tensor],
    config: GateConfig,
) -> torch.Tensor:
    """Compute proposal-level material weights for a configured gate."""

    n = int(features["visual_margin"].shape[0])
    lam = float(config.lambda_material)
    device = features["visual_margin"].device
    if config.gate_name == "visual_only":
        return torch.zeros(n, dtype=torch.float32, device=device)
    if config.gate_name == "fixed_material":
        return torch.full((n,), lam, dtype=torch.float32, device=device)

    visual_margin = features["visual_margin"].float()
    material_margin = features["material_margin"].float()
    agree = features["visual_material_agree"].bool()
    sample_count = features["class_sample_count"].float()

    margin_thr = float(config.margin_threshold or 0.0)
    material_thr = float(config.material_threshold or 0.0)
    sample_thr = float(config.sample_threshold or 0)

    if config.gate_name == "visual_margin_gate":
        enabled = visual_margin < margin_thr
        return enabled.float() * lam

    if config.gate_name == "material_confidence_gate":
        enabled = material_margin >= material_thr
        return enabled.float() * lam

    if config.gate_name == "agreement_gate":
        weights = torch.zeros(n, dtype=torch.float32, device=device)
        weights[agree] = lam * float(config.agree_boost)
        fallback = (~agree) & (visual_margin < margin_thr) & (material_margin > material_thr)
        weights[fallback] = lam
        return weights

    if config.gate_name == "sample_count_gate":
        enabled = sample_count >= sample_thr
        return enabled.float() * lam

    if config.gate_name == "hybrid_safe_gate":
        enabled = (visual_margin < margin_thr) & (material_margin >= material_thr) & (sample_count >= sample_thr)
        return enabled.float() * lam

    raise ValueError(f"unknown gate_name: {config.gate_name}")


def gate_summary(weights: torch.Tensor, config: GateConfig) -> dict[str, Any]:
    """Summarize a proposal-level gate vector."""

    enabled = weights > 0
    return {
        "gate_name": config.gate_name,
        "lambda_material": float(config.lambda_material),
        "margin_threshold": config.margin_threshold,
        "material_threshold": config.material_threshold,
        "sample_threshold": config.sample_threshold,
        "agree_boost": float(config.agree_boost),
        "proposal_count": int(weights.numel()),
        "enabled_count": int(enabled.sum().item()),
        "enabled_ratio": float(enabled.float().mean().item()) if weights.numel() else 0.0,
        "material_weight_mean": float(weights.float().mean().item()) if weights.numel() else 0.0,
        "material_weight_max": float(weights.float().max().item()) if weights.numel() else 0.0,
        "notes": config.notes,
    }
