"""VMASC unlabeled visual-material agreement calibration.

This module estimates material reliability from target-domain proposals only.
It never reads ground-truth labels or boxes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class VmascReliabilityConfig:
    q_margin: float = 0.70
    q_detector: float = 0.50
    min_pseudo_count: int = 5
    target_threshold: float = 0.05


def _minmax(values: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values
    lo = torch.min(values)
    hi = torch.max(values)
    if float((hi - lo).abs().item()) < 1.0e-8:
        return torch.ones_like(values)
    return (values - lo) / (hi - lo)


def proposal_agreement_stats(
    visual_category_scores: torch.Tensor,
    material_category_scores: torch.Tensor,
    proposal_meta: list[dict[str, Any]],
) -> dict[str, torch.Tensor]:
    """Compute proposal-level unlabeled agreement statistics."""

    visual = visual_category_scores.float()
    material = material_category_scores.float()
    topv = torch.topk(visual, k=min(2, visual.shape[1]), dim=1)
    topm = torch.topk(material, k=min(2, material.shape[1]), dim=1)
    visual_top1 = topv.indices[:, 0].long()
    visual_top1_score = topv.values[:, 0].float()
    visual_top2_score = topv.values[:, 1].float() if topv.values.shape[1] > 1 else torch.zeros_like(visual_top1_score)
    visual_margin = visual_top1_score - visual_top2_score
    material_top1 = topm.indices[:, 0].long()
    material_top1_score = topm.values[:, 0].float()
    material_top2_score = topm.values[:, 1].float() if topm.values.shape[1] > 1 else torch.zeros_like(material_top1_score)
    material_margin = material_top1_score - material_top2_score
    row = torch.arange(visual.shape[0], dtype=torch.long)
    material_for_visual = material[row, visual_top1]
    detector_score = torch.tensor([float(p.get("detector_score", 0.0)) for p in proposal_meta], dtype=torch.float32)
    return {
        "visual_top1": visual_top1,
        "visual_top1_score": visual_top1_score,
        "visual_top2_score": visual_top2_score,
        "visual_margin": visual_margin,
        "material_top1": material_top1,
        "material_top1_score": material_top1_score,
        "material_top2_score": material_top2_score,
        "material_margin": material_margin,
        "agree": visual_top1 == material_top1,
        "material_for_visual": material_for_visual,
        "material_vs_visual_top1_gap": material_top1_score - material_for_visual,
        "detector_score": detector_score,
    }


def estimate_reliability(
    visual_category_scores: torch.Tensor,
    material_category_scores: torch.Tensor,
    proposal_meta: list[dict[str, Any]],
    category_meta: list[dict[str, Any]],
    config: VmascReliabilityConfig,
) -> dict[str, Any]:
    """Estimate class and target material reliability without GT."""

    stats = proposal_agreement_stats(visual_category_scores, material_category_scores, proposal_meta)
    n = visual_category_scores.shape[0]
    c = visual_category_scores.shape[1]
    if n == 0:
        raise ValueError("empty proposal set")
    margin_threshold = float(torch.quantile(stats["visual_margin"], float(config.q_margin)).item())
    detector_threshold = float(torch.quantile(stats["detector_score"], float(config.q_detector)).item())
    material_margin_threshold = float(torch.quantile(stats["material_margin"], 0.70).item())
    high_conf = (stats["visual_margin"] >= margin_threshold) & (stats["detector_score"] >= detector_threshold)

    pseudo_counts = torch.zeros(c, dtype=torch.float32)
    agreement = torch.zeros(c, dtype=torch.float32)
    conflict = torch.zeros(c, dtype=torch.float32)
    support = torch.zeros(c, dtype=torch.float32)
    for class_index in range(c):
        mask = high_conf & (stats["visual_top1"] == class_index)
        count = int(mask.sum().item())
        pseudo_counts[class_index] = float(count)
        if count == 0:
            continue
        agreement[class_index] = stats["agree"][mask].float().mean()
        conflict_mask = (stats["material_top1"][mask] != class_index) & (stats["material_margin"][mask] >= material_margin_threshold)
        conflict[class_index] = conflict_mask.float().mean()
        support[class_index] = stats["material_for_visual"][mask].float().mean()

    support_norm = _minmax(support)
    reliability = agreement * support_norm * (1.0 - conflict)
    low_support = pseudo_counts < float(config.min_pseudo_count)
    reliability = reliability.masked_fill(low_support, 0.0).clamp(min=0.0, max=1.0)
    enabled = reliability > 0.0
    if pseudo_counts.sum() > 0:
        target_reliability = float((reliability * pseudo_counts).sum().item() / max(float(pseudo_counts.sum().item()), 1.0))
    else:
        target_reliability = 0.0

    class_rows = []
    for idx, category in enumerate(category_meta):
        class_rows.append(
            {
                "category_index": idx,
                "category_id": int(category.get("category_id", idx)),
                "category_name": str(category.get("category_name", idx)),
                "pseudo_count": int(pseudo_counts[idx].item()),
                "agreement_rate": float(agreement[idx].item()),
                "conflict_rate": float(conflict[idx].item()),
                "material_support": float(support[idx].item()),
                "reliability": float(reliability[idx].item()),
                "enabled": bool(enabled[idx].item()),
                "low_support": bool(low_support[idx].item()),
                "notes": "unlabeled target proposal statistics; no GT used",
            }
        )

    target_row = {
        "q_margin": float(config.q_margin),
        "q_detector": float(config.q_detector),
        "margin_threshold": margin_threshold,
        "detector_threshold": detector_threshold,
        "material_margin_threshold": material_margin_threshold,
        "proposal_count": int(n),
        "high_conf_proposal_count": int(high_conf.sum().item()),
        "target_reliability": target_reliability,
        "mean_agreement_rate": float(agreement[pseudo_counts > 0].mean().item()) if bool((pseudo_counts > 0).any().item()) else 0.0,
        "mean_conflict_rate": float(conflict[pseudo_counts > 0].mean().item()) if bool((pseudo_counts > 0).any().item()) else 0.0,
        "class_enabled_count": int(enabled.sum().item()),
        "low_support_class_count": int(low_support.sum().item()),
        "target_material_enabled": bool(target_reliability >= float(config.target_threshold)),
        "notes": "transductive unlabeled test-time self-calibration; no GT used",
    }
    return {
        "target": target_row,
        "classes": class_rows,
        "reliability": reliability,
        "agreement_minus_conflict": (agreement - conflict).clamp(min=0.0),
        "low_support": low_support,
        "proposal_stats": stats,
    }

