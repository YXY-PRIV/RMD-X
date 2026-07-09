"""Distribution-calibrated no-regret material fusion.

This module operates only on cached inference-time visual/material similarity
tensors.  It does not read ground-truth boxes, does not train a model, and keeps
material decisions inside each proposal's visual top-K category set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class DcnrConfig:
    candidate_id: str
    method: str
    topk: int = 3
    alpha: float = 0.35
    tau: float = 1.0
    margin: float = 0.0


@dataclass
class DcnrResult:
    fused_category_scores: torch.Tensor
    winner_category_indices: torch.Tensor
    visual_top1_category_indices: torch.Tensor
    visual_topk_indices: torch.Tensor
    enabled_mask: torch.Tensor
    stats: dict[str, Any]


def compute_visual_topk(scores: torch.Tensor, topk: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    k = max(1, min(int(topk), scores.shape[1]))
    values, indices = torch.topk(scores.float(), k=k, dim=1)
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(1, indices, True)
    return values, indices.long(), mask


def robust_z(selected: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    median = selected.median(dim=1, keepdim=True).values
    q75 = torch.quantile(selected.float(), 0.75, dim=1, keepdim=True)
    q25 = torch.quantile(selected.float(), 0.25, dim=1, keepdim=True)
    iqr = (q75 - q25).abs().clamp(min=eps)
    return (selected.float() - median.float()) / iqr


def rank_calibrate(selected: torch.Tensor) -> torch.Tensor:
    n, k = selected.shape
    if k <= 1:
        return torch.ones_like(selected, dtype=torch.float32)
    order = torch.argsort(selected.float(), dim=1, descending=True)
    rank_values = torch.linspace(1.0, 0.0, steps=k, dtype=torch.float32)
    out = torch.zeros((n, k), dtype=torch.float32)
    out.scatter_(1, order, rank_values.unsqueeze(0).expand(n, k))
    return out


def entropy_from_scores(selected: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    probs = torch.softmax(selected.float(), dim=1).clamp(min=eps)
    return -(probs * probs.log()).sum(dim=1)


def _std_norm(values: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return values.float().std(dim=1, keepdim=True, unbiased=False).clamp(min=eps)


def _pairwise_no_regret_winner(
    visual_topk: torch.Tensor,
    material_topk: torch.Tensor,
    topk_indices: torch.Tensor,
    *,
    tau: float,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n, k = visual_topk.shape
    visual_gap = visual_topk[:, [0]] - visual_topk
    material_adv = material_topk - material_topk[:, [0]]
    visual_gap_norm = visual_gap / _std_norm(visual_topk)
    material_adv_norm = material_adv / _std_norm(material_topk)
    benefit = material_adv_norm - float(tau) * visual_gap_norm
    allowed = material_adv_norm > (float(tau) * visual_gap_norm + float(margin))
    allowed[:, 0] = False
    benefit_masked = benefit.masked_fill(~allowed, -1.0e9)
    best_pos = torch.argmax(benefit_masked, dim=1)
    has_override = allowed.any(dim=1)
    winner_pos = torch.where(has_override, best_pos, torch.zeros(n, dtype=torch.long))
    winner = topk_indices.gather(1, winner_pos.view(-1, 1)).view(-1).long()
    selected_gap = visual_gap_norm.gather(1, winner_pos.view(-1, 1)).view(-1)
    selected_adv = material_adv_norm.gather(1, winner_pos.view(-1, 1)).view(-1)
    return winner, has_override, selected_gap.float(), selected_adv.float()


def _make_fused_from_topk(
    visual_scores: torch.Tensor,
    topk_indices: torch.Tensor,
    selected_fused: torch.Tensor,
) -> torch.Tensor:
    fused = visual_scores.clone().float()
    fused.scatter_(1, topk_indices, selected_fused.float())
    return fused


def _stats(
    config: DcnrConfig,
    visual_scores: torch.Tensor,
    material_scores: torch.Tensor,
    topk_indices: torch.Tensor,
    winner: torch.Tensor,
    visual_top1: torch.Tensor,
    enabled: torch.Tensor,
    selected_gap_norm: torch.Tensor | None,
    selected_adv_norm: torch.Tensor | None,
    notes: str,
) -> dict[str, Any]:
    proposal_count = int(visual_scores.shape[0])
    changed = winner.long() != visual_top1.long()
    in_topk = (topk_indices == winner.view(-1, 1)).any(dim=1) if proposal_count else torch.zeros(0, dtype=torch.bool)
    changed_to_outside = int((changed & ~in_topk).sum().item()) if proposal_count else 0
    changed_count = int(changed.sum().item()) if proposal_count else 0
    selected_visual = visual_scores.gather(1, topk_indices)
    selected_material = material_scores.gather(1, topk_indices)
    changed_gap = selected_gap_norm[changed] if selected_gap_norm is not None and changed_count else torch.empty(0)
    changed_adv = selected_adv_norm[changed] if selected_adv_norm is not None and changed_count else torch.empty(0)
    return {
        "candidate_id": config.candidate_id,
        "method": config.method,
        "topk": int(config.topk),
        "alpha": float(config.alpha),
        "tau": float(config.tau),
        "margin": float(config.margin),
        "proposal_count": proposal_count,
        "enabled_ratio": round(float(enabled.float().mean().item()), 6) if proposal_count else 0.0,
        "changed_category_count": changed_count,
        "changed_category_ratio": round(changed_count / proposal_count, 6) if proposal_count else 0.0,
        "changed_to_outside_topk_count": changed_to_outside,
        "mean_visual_gap_changed": round(float(changed_gap.mean().item()), 6) if changed_gap.numel() else 0.0,
        "mean_material_adv_changed": round(float(changed_adv.mean().item()), 6) if changed_adv.numel() else 0.0,
        "local_score_scale_summary": {
            "visual_topk_std_mean": round(float(selected_visual.std(dim=1, unbiased=False).mean().item()), 6)
            if proposal_count
            else 0.0,
            "material_topk_std_mean": round(float(selected_material.std(dim=1, unbiased=False).mean().item()), 6)
            if proposal_count
            else 0.0,
        },
        "notes": notes,
    }


def fuse_dcnr(
    visual_category_scores: torch.Tensor,
    material_category_scores: torch.Tensor,
    config: DcnrConfig,
) -> DcnrResult:
    visual = visual_category_scores.float()
    material = material_category_scores.float()
    if visual.shape != material.shape:
        raise ValueError(f"visual/material shape mismatch: {tuple(visual.shape)} vs {tuple(material.shape)}")

    method = config.method
    if method == "visual_only":
        topk_values, topk_indices, _mask = compute_visual_topk(visual, max(1, visual.shape[1]))
        visual_top1 = torch.argmax(visual, dim=1).long()
        enabled = torch.zeros(visual.shape[0], dtype=torch.bool)
        stats = _stats(config, visual, material, topk_indices, visual_top1, visual_top1, enabled, None, None, "visual_only")
        return DcnrResult(visual, visual_top1, visual_top1, topk_indices, enabled, stats)

    topk_values, topk_indices, topk_mask = compute_visual_topk(visual, config.topk)
    material_topk = material.gather(1, topk_indices)
    visual_top1 = topk_indices[:, 0].long()
    enabled = torch.ones(visual.shape[0], dtype=torch.bool)
    gap_for_stats = torch.zeros(visual.shape[0], dtype=torch.float32)
    adv_for_stats = torch.zeros(visual.shape[0], dtype=torch.float32)
    notes = ""

    if method == "fixed_material_reference":
        centered = material - material.mean(dim=1, keepdim=True)
        fused = visual + float(config.alpha) * centered
        winner = torch.argmax(fused, dim=1).long()
        # This reference is intentionally not top-K constrained.
        topk_for_stats = topk_indices
        enabled = torch.ones(visual.shape[0], dtype=torch.bool)
        stats = _stats(config, visual, material, topk_for_stats, winner, visual_top1, enabled, None, None, "fixed material reference")
        return DcnrResult(fused, winner, visual_top1, topk_for_stats, enabled, stats)

    if method == "local_z_topk_fusion":
        visual_z = robust_z(topk_values)
        material_z = robust_z(material_topk)
        selected_fused = visual_z + float(config.alpha) * material_z
        winner_pos = torch.argmax(selected_fused, dim=1)
        winner = topk_indices.gather(1, winner_pos.view(-1, 1)).view(-1).long()
        fused = _make_fused_from_topk(visual, topk_indices, selected_fused)
        gap_for_stats = ((topk_values[:, 0] - topk_values.gather(1, winner_pos.view(-1, 1)).view(-1)) / _std_norm(topk_values).view(-1)).float()
        adv_for_stats = ((material_topk.gather(1, winner_pos.view(-1, 1)).view(-1) - material_topk[:, 0]) / _std_norm(material_topk).view(-1)).float()
        notes = "local robust z-score fusion inside visual top-K"

    elif method == "rank_calibrated_topk_fusion":
        visual_rank = rank_calibrate(topk_values)
        material_rank = rank_calibrate(material_topk)
        selected_fused = visual_rank + float(config.alpha) * material_rank
        winner_pos = torch.argmax(selected_fused, dim=1)
        winner = topk_indices.gather(1, winner_pos.view(-1, 1)).view(-1).long()
        fused = _make_fused_from_topk(visual, topk_indices, selected_fused)
        gap_for_stats = ((topk_values[:, 0] - topk_values.gather(1, winner_pos.view(-1, 1)).view(-1)) / _std_norm(topk_values).view(-1)).float()
        adv_for_stats = ((material_topk.gather(1, winner_pos.view(-1, 1)).view(-1) - material_topk[:, 0]) / _std_norm(material_topk).view(-1)).float()
        notes = "rank-calibrated top-K fusion"

    elif method == "no_regret_pairwise_override":
        winner, enabled, gap_for_stats, adv_for_stats = _pairwise_no_regret_winner(
            topk_values,
            material_topk,
            topk_indices,
            tau=config.tau,
            margin=config.margin,
        )
        fused = visual.clone().float()
        notes = "pairwise no-regret override inside visual top-K"

    elif method in {"distribution_calibrated_no_regret_fusion", "hixray_safe_dcnr"}:
        visual_z = robust_z(topk_values)
        material_z = robust_z(material_topk)
        selected_fused = visual_z + float(config.alpha) * material_z
        fused_winner_pos = torch.argmax(selected_fused, dim=1)
        fused_winner = topk_indices.gather(1, fused_winner_pos.view(-1, 1)).view(-1).long()
        proposed_gap = ((topk_values[:, 0] - topk_values.gather(1, fused_winner_pos.view(-1, 1)).view(-1)) / _std_norm(topk_values).view(-1)).float()
        proposed_adv = ((material_topk.gather(1, fused_winner_pos.view(-1, 1)).view(-1) - material_topk[:, 0]) / _std_norm(material_topk).view(-1)).float()
        passes = proposed_adv > (float(config.tau) * proposed_gap + float(config.margin))
        passes = passes & (fused_winner != visual_top1)
        if method == "hixray_safe_dcnr":
            material_top1 = torch.argmax(material, dim=1).long()
            material_top1_in_topk = topk_mask.gather(1, material_top1.view(-1, 1)).view(-1)
            safe_entropy = entropy_from_scores(material_topk) <= entropy_from_scores(topk_values)
            safe_mask = material_top1_in_topk & safe_entropy
            passes = passes & safe_mask
            notes = "DCNR plus material-top1-in-topK and entropy safety fallback"
        else:
            notes = "local robust z fusion gated by no-regret pairwise check"
        winner = torch.where(passes, fused_winner, visual_top1).long()
        enabled = passes
        fused = _make_fused_from_topk(visual, topk_indices, selected_fused)
        gap_for_stats = proposed_gap
        adv_for_stats = proposed_adv

    else:
        raise ValueError(f"Unsupported DCNR method: {method}")

    # Hard guarantee: fair DCNR variants cannot choose outside visual top-K.
    if method != "fixed_material_reference":
        winner = torch.where(topk_mask.gather(1, winner.view(-1, 1)).view(-1), winner, visual_top1).long()
    stats = _stats(config, visual, material, topk_indices, winner, visual_top1, enabled, gap_for_stats, adv_for_stats, notes)
    return DcnrResult(fused.float(), winner.long(), visual_top1.long(), topk_indices.long(), enabled, stats)
