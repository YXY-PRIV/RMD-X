"""FCR reranking score helpers."""

from __future__ import annotations

import torch


def category_scores_from_prototype_scores(
    visual_similarity: torch.Tensor,
    prototypes_by_category: list[list[int]],
) -> torch.Tensor:
    scores = torch.full((visual_similarity.shape[0], len(prototypes_by_category)), -1.0e9, dtype=torch.float32)
    for category_index, proto_indices in enumerate(prototypes_by_category):
        if not proto_indices:
            continue
        idx = torch.tensor(proto_indices, dtype=torch.long)
        scores[:, category_index] = visual_similarity[:, idx].max(dim=1).values.float()
    return scores


def fuse_topk(visual_scores: torch.Tensor, fcr_scores: torch.Tensor, *, topk: int, beta: float, mode: str = "raw") -> dict:
    k = max(1, min(int(topk), visual_scores.shape[1]))
    top_values, top_indices = torch.topk(visual_scores.float(), k=k, dim=1)
    top_mask = torch.zeros_like(visual_scores, dtype=torch.bool)
    top_mask.scatter_(1, top_indices, True)
    selected = fcr_scores.float().gather(1, top_indices)
    if mode == "centered":
        selected = selected - selected.mean(dim=1, keepdim=True)
    elif mode == "rank":
        rank = torch.argsort(selected, dim=1, descending=True)
        bonus = torch.linspace(float(beta), -float(beta), steps=k).view(1, -1).expand_as(selected)
        delta = torch.zeros_like(selected)
        delta.scatter_(1, rank, bonus)
        selected_delta = delta
    else:
        selected_delta = float(beta) * selected
    if mode != "rank":
        selected_delta = float(beta) * selected
    fused = visual_scores.float().clone()
    fused.scatter_add_(1, top_indices, selected_delta.float())
    constrained = fused.masked_fill(~top_mask, -1.0e9)
    winner = torch.argmax(constrained, dim=1).long()
    visual_top1 = top_indices[:, 0].long()
    changed = winner != visual_top1
    outside = int((~top_mask.gather(1, winner.view(-1, 1)).view(-1)).sum().item())
    return {
        "fused": fused,
        "winner": winner,
        "visual_top1": visual_top1,
        "topk_indices": top_indices,
        "changed_category_count": int(changed.sum().item()),
        "changed_category_ratio": float(changed.float().mean().item()) if changed.numel() else 0.0,
        "changed_to_outside_topk_count": outside,
    }

