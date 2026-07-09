"""Apply JCL reranker scores inside visual top-K."""

from __future__ import annotations

from typing import Any

import torch


def rerank_topk(
    visual_category_scores: torch.Tensor,
    jcl_category_scores: torch.Tensor,
    *,
    topk: int,
    beta: float,
) -> dict[str, Any]:
    visual = visual_category_scores.float()
    jcl = jcl_category_scores.float()
    k = max(1, min(int(topk), visual.shape[1]))
    top_values, top_indices = torch.topk(visual, k=k, dim=1)
    mask = torch.zeros_like(visual, dtype=torch.bool)
    mask.scatter_(1, top_indices, True)
    fused = visual.clone()
    fused.scatter_add_(1, top_indices, float(beta) * jcl.gather(1, top_indices))
    constrained = fused.masked_fill(~mask, -1.0e9)
    winner = torch.argmax(constrained, dim=1).long()
    visual_top1 = top_indices[:, 0].long()
    changed = winner != visual_top1
    outside = int((~mask.gather(1, winner.view(-1, 1)).view(-1)).sum().item())
    return {
        "fused": fused,
        "winner": winner,
        "visual_top1": visual_top1,
        "topk_indices": top_indices.long(),
        "changed_category_count": int(changed.sum().item()),
        "changed_category_ratio": float(changed.float().mean().item()) if len(changed) else 0.0,
        "changed_to_outside_topk_count": outside,
    }
