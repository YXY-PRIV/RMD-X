"""Visual top-K constrained OpenCLIP crop-text fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class FCRTextConfig:
    candidate_id: str
    method: str
    topk: int
    beta: float


def fuse_fcr_text(visual_scores: torch.Tensor, text_scores: torch.Tensor, config: FCRTextConfig) -> dict[str, Any]:
    if config.method == "visual_only":
        visual_top1 = torch.argmax(visual_scores, dim=1).long()
        return {
            "fused": visual_scores.float(),
            "winner": visual_top1,
            "visual_top1": visual_top1,
            "topk_indices": torch.topk(visual_scores, k=visual_scores.shape[1], dim=1).indices.long(),
            "enabled_mask": torch.zeros(visual_scores.shape[0], dtype=torch.bool),
            "stats": _stats(visual_scores, text_scores, visual_top1, visual_top1, torch.empty((visual_scores.shape[0], 0), dtype=torch.long), torch.zeros(visual_scores.shape[0], dtype=torch.bool), 0, config, "visual_only"),
        }

    k = max(1, min(int(config.topk), visual_scores.shape[1]))
    topk = torch.topk(visual_scores.float(), k=k, dim=1).indices.long()
    mask = torch.zeros_like(visual_scores, dtype=torch.bool)
    mask.scatter_(1, topk, True)
    visual_top1 = topk[:, 0].long()
    selected_text = text_scores.float().gather(1, topk)
    enabled = torch.ones(visual_scores.shape[0], dtype=torch.bool)
    notes = ""

    if config.method == "raw_topk_text_fusion":
        delta = selected_text * float(config.beta)
    elif config.method == "centered_topk_text_fusion":
        delta = (selected_text - selected_text.mean(dim=1, keepdim=True)) * float(config.beta)
    elif config.method == "rank_topk_text_fusion":
        order = torch.argsort(selected_text, dim=1, descending=True)
        bonus = torch.linspace(float(config.beta), -float(config.beta), steps=k, dtype=torch.float32).view(1, -1).expand_as(selected_text)
        delta = torch.zeros_like(selected_text)
        delta.scatter_(1, order, bonus)
    elif config.method == "agreement_safe_text_fusion":
        text_top1 = torch.argmax(text_scores.float(), dim=1)
        enabled = mask.gather(1, text_top1.view(-1, 1)).view(-1)
        delta = (selected_text - selected_text.mean(dim=1, keepdim=True)) * float(config.beta)
        delta = delta * enabled.float().view(-1, 1)
        notes = "text top1 outside visual top-K disabled"
    elif config.method == "margin_protected_text_fusion":
        top2 = torch.topk(visual_scores.float(), k=2, dim=1).values
        margins = top2[:, 0] - top2[:, 1]
        threshold = float(torch.quantile(margins, 0.85).item())
        enabled = margins < threshold
        delta = (selected_text - selected_text.mean(dim=1, keepdim=True)) * float(config.beta)
        delta = delta * enabled.float().view(-1, 1)
        notes = f"visual margin q85 protected threshold={threshold:.6f}"
    else:
        raise ValueError(f"Unsupported FCR text method: {config.method}")

    fused = visual_scores.float().clone()
    fused.scatter_add_(1, topk, delta.float())
    constrained = fused.masked_fill(~mask, -1.0e9)
    winner = torch.argmax(constrained, dim=1).long()
    outside = int((~mask.gather(1, winner.view(-1, 1)).view(-1)).sum().item())
    return {
        "fused": fused,
        "winner": winner,
        "visual_top1": visual_top1,
        "topk_indices": topk,
        "enabled_mask": enabled,
        "stats": _stats(visual_scores, text_scores, visual_top1, winner, topk, enabled, outside, config, notes),
    }


def _stats(
    visual_scores: torch.Tensor,
    text_scores: torch.Tensor,
    visual_top1: torch.Tensor,
    winner: torch.Tensor,
    topk_indices: torch.Tensor,
    enabled: torch.Tensor,
    outside: int,
    config: FCRTextConfig,
    notes: str,
) -> dict[str, Any]:
    changed = winner != visual_top1
    rows = torch.arange(visual_scores.shape[0], dtype=torch.long)
    text_advantage = text_scores[rows, winner] - text_scores[rows, visual_top1]
    top2 = torch.topk(visual_scores.float(), k=2, dim=1).values
    margin = top2[:, 0] - top2[:, 1]
    changed_mask = changed if changed.numel() else torch.zeros(0, dtype=torch.bool)
    return {
        "candidate_id": config.candidate_id,
        "method": config.method,
        "topk": int(config.topk),
        "beta": float(config.beta),
        "enabled_ratio": round(float(enabled.float().mean().item()), 6) if enabled.numel() else 0.0,
        "changed_category_count": int(changed.sum().item()),
        "changed_category_ratio": round(float(changed.float().mean().item()), 6) if changed.numel() else 0.0,
        "changed_to_outside_topk_count": int(outside),
        "mean_visual_margin_changed": round(float(margin[changed_mask].mean().item()), 6) if changed_mask.any() else 0.0,
        "mean_text_advantage_changed": round(float(text_advantage[changed_mask].mean().item()), 6) if changed_mask.any() else 0.0,
        "notes": notes,
    }

