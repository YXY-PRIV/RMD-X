"""Visual Top-K constrained material fusion for cached RMD-X tensors.

This module is independent from the original RAXO implementation.  It works on
the cache tensors produced by the RMD-X cache pipeline and restricts material
signals to the visual top-K category set for each proposal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class VisualTopKConfig:
    method: str
    topk: int
    lambda_material: float
    candidate_id: str


@dataclass
class VisualTopKResult:
    fused_category_scores: torch.Tensor
    winner_category_indices: torch.Tensor
    visual_top1_category_indices: torch.Tensor
    visual_topk_indices: torch.Tensor
    enabled_mask: torch.Tensor
    stats: dict[str, Any]


def build_category_maps(
    prototype_meta: list[dict[str, Any]],
    category_meta: list[dict[str, Any]],
) -> tuple[list[list[int]], torch.Tensor, list[int], list[str]]:
    """Map prototypes to evaluation-category indices.

    Returns prototype indices grouped by category index, a prototype-to-category
    tensor, category IDs, and category names.
    """

    category_ids = [int(cat["category_id"]) for cat in category_meta]
    category_names = [str(cat["category_name"]) for cat in category_meta]
    category_id_to_index = {category_id: index for index, category_id in enumerate(category_ids)}
    proto_to_category = torch.full((len(prototype_meta),), -1, dtype=torch.long)
    prototypes_by_category: list[list[int]] = [[] for _ in category_ids]

    for fallback_index, proto in enumerate(prototype_meta):
        proto_index = int(proto.get("prototype_index", fallback_index))
        if proto.get("is_negative"):
            continue
        category_id = proto.get("category_id")
        if category_id is None:
            continue
        category_index = category_id_to_index.get(int(category_id))
        if category_index is None:
            continue
        if proto_index >= len(proto_to_category):
            continue
        proto_to_category[proto_index] = category_index
        prototypes_by_category[category_index].append(proto_index)

    return prototypes_by_category, proto_to_category, category_ids, category_names


def category_scores_from_prototypes(
    visual_similarity: torch.Tensor,
    prototypes_by_category: list[list[int]],
) -> torch.Tensor:
    """Reduce prototype-level visual similarity to category-level max scores."""

    scores = torch.full(
        (visual_similarity.shape[0], len(prototypes_by_category)),
        -1.0e9,
        dtype=torch.float32,
    )
    for category_index, prototype_indices in enumerate(prototypes_by_category):
        if not prototype_indices:
            continue
        idx = torch.tensor(prototype_indices, dtype=torch.long)
        scores[:, category_index] = visual_similarity[:, idx].max(dim=1).values.float()
    return scores


def compute_visual_topk(
    visual_category_scores: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return visual top-K values, indices, and a boolean category mask."""

    k = max(1, min(int(topk), visual_category_scores.shape[1]))
    values, indices = torch.topk(visual_category_scores, k=k, dim=1)
    mask = torch.zeros_like(visual_category_scores, dtype=torch.bool)
    mask.scatter_(1, indices, True)
    return values, indices, mask


def _topk_mean(values: torch.Tensor, topk_indices: torch.Tensor) -> torch.Tensor:
    selected = values.gather(1, topk_indices)
    return selected.mean(dim=1, keepdim=True)


def _rank_bonus(
    material_category_scores: torch.Tensor,
    topk_indices: torch.Tensor,
    lambda_material: float,
) -> torch.Tensor:
    selected = material_category_scores.gather(1, topk_indices)
    k = selected.shape[1]
    if k <= 1:
        return torch.zeros_like(selected)
    rank_order = torch.argsort(selected, dim=1, descending=True)
    bonus_by_rank = torch.linspace(float(lambda_material), -float(lambda_material), steps=k)
    selected_bonus = torch.zeros_like(selected)
    expanded_bonus = bonus_by_rank.unsqueeze(0).expand_as(selected)
    selected_bonus.scatter_(1, rank_order, expanded_bonus)
    return selected_bonus


def fuse_category_scores(
    visual_category_scores: torch.Tensor,
    material_category_scores: torch.Tensor,
    config: VisualTopKConfig,
) -> VisualTopKResult:
    """Fuse material scores under a visual top-K category constraint.

    For all VTC methods, the final category is selected only from the visual
    top-K set.  This makes the constraint explicit and guarantees that material
    cannot move a proposal to a category outside the visual candidate set.
    """

    method = config.method
    if method == "visual_only":
        topk_values, topk_indices, topk_mask = compute_visual_topk(
            visual_category_scores, visual_category_scores.shape[1]
        )
        visual_top1 = torch.argmax(visual_category_scores, dim=1)
        enabled = torch.zeros(visual_category_scores.shape[0], dtype=torch.bool)
        stats = _stats(
            visual_category_scores,
            visual_category_scores,
            topk_indices,
            enabled,
            visual_top1,
            visual_top1,
            config,
            changed_to_outside=0,
            notes="visual_only: no material contribution",
        )
        return VisualTopKResult(
            fused_category_scores=visual_category_scores.float(),
            winner_category_indices=visual_top1.long(),
            visual_top1_category_indices=visual_top1.long(),
            visual_topk_indices=topk_indices.long(),
            enabled_mask=enabled,
            stats=stats,
        )

    topk_values, topk_indices, topk_mask = compute_visual_topk(
        visual_category_scores, config.topk
    )
    visual_top1 = topk_indices[:, 0].long()
    fused = visual_category_scores.clone().float()
    enabled = torch.ones(visual_category_scores.shape[0], dtype=torch.bool)
    notes = ""

    if method == "topk_additive_material":
        selected_delta = material_category_scores.gather(1, topk_indices) * float(
            config.lambda_material
        )
    elif method == "topk_centered_material":
        mean = _topk_mean(material_category_scores, topk_indices)
        selected_delta = (
            material_category_scores.gather(1, topk_indices) - mean
        ) * float(config.lambda_material)
    elif method == "topk_rank_material":
        selected_delta = _rank_bonus(
            material_category_scores, topk_indices, config.lambda_material
        )
    elif method == "topk_agreement_safe_material":
        material_top1 = torch.argmax(material_category_scores, dim=1)
        enabled = topk_mask.gather(1, material_top1.view(-1, 1)).view(-1)
        mean = _topk_mean(material_category_scores, topk_indices)
        selected_delta = (
            material_category_scores.gather(1, topk_indices) - mean
        ) * float(config.lambda_material)
        selected_delta = selected_delta * enabled.float().view(-1, 1)
        notes = "material top1 outside visual top-K proposals are disabled"
    elif method == "topk_margin_protected_material":
        k2 = min(2, visual_category_scores.shape[1])
        margins = torch.zeros(visual_category_scores.shape[0], dtype=torch.float32)
        if k2 == 2:
            top2 = torch.topk(visual_category_scores, k=2, dim=1).values
            margins = top2[:, 0] - top2[:, 1]
        threshold = float(torch.quantile(margins, 0.85).item()) if margins.numel() else 0.0
        enabled = margins <= threshold
        mean = _topk_mean(material_category_scores, topk_indices)
        selected_delta = (
            material_category_scores.gather(1, topk_indices) - mean
        ) * float(config.lambda_material)
        selected_delta = selected_delta * enabled.float().view(-1, 1)
        notes = f"visual margin protected, threshold_q85={threshold:.6f}"
    else:
        raise ValueError(f"Unsupported visual top-K method: {method}")

    fused.scatter_add_(1, topk_indices, selected_delta.float())
    constrained = fused.masked_fill(~topk_mask, -1.0e9)
    winner = torch.argmax(constrained, dim=1).long()
    outside = (~topk_mask.gather(1, winner.view(-1, 1)).view(-1)).sum().item()
    stats = _stats(
        visual_category_scores,
        fused,
        topk_indices,
        enabled,
        visual_top1,
        winner,
        config,
        changed_to_outside=int(outside),
        notes=notes,
    )
    return VisualTopKResult(
        fused_category_scores=fused.float(),
        winner_category_indices=winner,
        visual_top1_category_indices=visual_top1,
        visual_topk_indices=topk_indices.long(),
        enabled_mask=enabled,
        stats=stats,
    )


def _stats(
    visual_scores: torch.Tensor,
    fused_scores: torch.Tensor,
    topk_indices: torch.Tensor,
    enabled: torch.Tensor,
    visual_top1: torch.Tensor,
    winner: torch.Tensor,
    config: VisualTopKConfig,
    *,
    changed_to_outside: int,
    notes: str,
) -> dict[str, Any]:
    proposal_count = int(visual_scores.shape[0])
    changed = winner != visual_top1
    if proposal_count:
        in_topk = (topk_indices == winner.view(-1, 1)).any(dim=1)
        changed_within = changed & in_topk
        outside_suppressed = 1.0 - (float(config.topk) / float(visual_scores.shape[1]))
    else:
        changed_within = changed
        outside_suppressed = 0.0
    return {
        "candidate_id": config.candidate_id,
        "method": config.method,
        "topk": int(config.topk),
        "lambda_material": float(config.lambda_material),
        "proposal_count": proposal_count,
        "enabled_ratio": round(float(enabled.float().mean().item()), 6) if proposal_count else 0.0,
        "material_outside_topk_suppressed_ratio": round(max(0.0, outside_suppressed), 6),
        "changed_category_count": int(changed.sum().item()),
        "changed_category_ratio": round(float(changed.float().mean().item()), 6)
        if proposal_count
        else 0.0,
        "changed_within_topk_ratio": round(float(changed_within.float().mean().item()), 6)
        if proposal_count
        else 0.0,
        "changed_to_outside_topk_count": int(changed_to_outside),
        "notes": notes,
    }


def choose_best_prototypes_for_winners(
    similarity: torch.Tensor,
    winner_category_indices: torch.Tensor,
    prototypes_by_category: list[list[int]],
) -> torch.Tensor:
    """Choose the best prototype inside each already-selected category."""

    chosen = torch.full((similarity.shape[0],), -1, dtype=torch.long)
    rows_by_category: dict[int, list[int]] = {}
    for row_index, category_index in enumerate(winner_category_indices.tolist()):
        rows_by_category.setdefault(int(category_index), []).append(row_index)
    for category_index, row_indices in rows_by_category.items():
        prototype_indices = prototypes_by_category[category_index]
        if not prototype_indices:
            continue
        row_tensor = torch.tensor(row_indices, dtype=torch.long)
        proto_tensor = torch.tensor(prototype_indices, dtype=torch.long)
        local_scores = similarity.index_select(0, row_tensor).index_select(1, proto_tensor)
        local_winners = torch.argmax(local_scores, dim=1)
        chosen[row_tensor] = proto_tensor[local_winners]
    return chosen

