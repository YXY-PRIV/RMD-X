"""VMASC fusion over cached category visual/material tensors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from rmdx.vmasc_calibration import VmascReliabilityConfig, estimate_reliability
from rmdx.visual_topk_material_fusion import choose_best_prototypes_for_winners


@dataclass(frozen=True)
class VmascFusionConfig:
    candidate_id: str
    method: str
    topk: int = 3
    lambda_max: float = 0.05
    q_margin: float = 0.70
    q_detector: float = 0.50
    min_pseudo_count: int = 5
    target_threshold: float = 0.05


def fuse_vmasc(
    visual_category_scores: torch.Tensor,
    material_category_scores: torch.Tensor,
    proposal_meta: list[dict[str, Any]],
    category_meta: list[dict[str, Any]],
    config: VmascFusionConfig,
) -> dict[str, Any]:
    """Fuse material into visual top-K using unlabeled reliability."""

    visual = visual_category_scores.float()
    material = material_category_scores.float()
    reliability_pack = estimate_reliability(
        visual,
        material,
        proposal_meta,
        category_meta,
        VmascReliabilityConfig(
            q_margin=config.q_margin,
            q_detector=config.q_detector,
            min_pseudo_count=config.min_pseudo_count,
            target_threshold=config.target_threshold,
        ),
    )
    reliability = reliability_pack["reliability"].float()
    agree_minus_conflict = reliability_pack["agreement_minus_conflict"].float()
    low_support = reliability_pack["low_support"].bool()
    proposal_stats = reliability_pack["proposal_stats"]
    target_reliability = float(reliability_pack["target"]["target_reliability"])

    k = max(1, min(int(config.topk), visual.shape[1]))
    top_values, top_indices = torch.topk(visual, k=k, dim=1)
    top_mask = torch.zeros_like(visual, dtype=torch.bool)
    top_mask.scatter_(1, top_indices, True)
    visual_top1 = top_indices[:, 0].long()
    material_top1 = proposal_stats["material_top1"].long()
    material_top1_in_topk = top_mask.gather(1, material_top1.view(-1, 1)).view(-1)

    fused = visual.clone()
    enabled = torch.ones(visual.shape[0], dtype=torch.bool)
    selected_weight = torch.zeros_like(top_values)
    selected_material = material.gather(1, top_indices)
    centered_material = selected_material - selected_material.mean(dim=1, keepdim=True)

    if config.method == "visual_only":
        enabled = torch.zeros(visual.shape[0], dtype=torch.bool)
        selected_delta = torch.zeros_like(selected_material)
    elif config.method == "class_reliability_scaled_fusion":
        selected_weight = float(config.lambda_max) * reliability.gather(0, top_indices.reshape(-1)).view_as(top_indices).float()
        selected_delta = selected_weight * selected_material
    elif config.method == "agreement_filtered_fusion":
        enabled = material_top1_in_topk
        selected_weight = float(config.lambda_max) * reliability.gather(0, top_indices.reshape(-1)).view_as(top_indices).float()
        selected_delta = selected_weight * selected_material * enabled.float().view(-1, 1)
    elif config.method == "conflict_suppressed_fusion":
        class_weight = float(config.lambda_max) * agree_minus_conflict.masked_fill(low_support, 0.0)
        selected_weight = class_weight.gather(0, top_indices.reshape(-1)).view_as(top_indices).float()
        selected_delta = selected_weight * centered_material
    elif config.method == "target_reliability_gate":
        enabled = torch.full((visual.shape[0],), target_reliability >= float(config.target_threshold), dtype=torch.bool)
        class_weight = float(config.lambda_max) * agree_minus_conflict.masked_fill(low_support, 0.0)
        selected_weight = class_weight.gather(0, top_indices.reshape(-1)).view_as(top_indices).float()
        selected_delta = selected_weight * centered_material * enabled.float().view(-1, 1)
    elif config.method == "VMASC_safe_main":
        target_enabled = target_reliability >= float(config.target_threshold)
        class_enabled = reliability > 0.0
        proposal_enabled = material_top1_in_topk & torch.tensor(bool(target_enabled))
        enabled = proposal_enabled
        class_weight = float(config.lambda_max) * reliability.masked_fill(~class_enabled, 0.0)
        selected_weight = class_weight.gather(0, top_indices.reshape(-1)).view_as(top_indices).float()
        selected_delta = selected_weight * centered_material * proposal_enabled.float().view(-1, 1)
    else:
        raise ValueError(f"unsupported VMASC method: {config.method}")

    fused.scatter_add_(1, top_indices, selected_delta.float())
    constrained = fused.masked_fill(~top_mask, -1.0e9)
    winner = torch.argmax(constrained, dim=1).long()
    changed = winner != visual_top1
    outside = int((~top_mask.gather(1, winner.view(-1, 1)).view(-1)).sum().item())
    stats = {
        "candidate_id": config.candidate_id,
        "method": config.method,
        "topk": int(config.topk),
        "lambda_max": float(config.lambda_max),
        "q_margin": float(config.q_margin),
        "q_detector": float(config.q_detector),
        "min_pseudo_count": int(config.min_pseudo_count),
        "target_threshold": float(config.target_threshold),
        "target_reliability": target_reliability,
        "enabled_ratio": float(enabled.float().mean().item()) if enabled.numel() else 0.0,
        "class_enabled_count": int((reliability > 0.0).sum().item()),
        "low_support_class_count": int(low_support.sum().item()),
        "changed_category_count": int(changed.sum().item()),
        "changed_category_ratio": float(changed.float().mean().item()) if changed.numel() else 0.0,
        "changed_to_outside_topk_count": outside,
        "transductive_unlabeled": True,
        "uses_gt": False,
        "notes": "VMASC uses target-domain unlabeled proposals only for reliability estimation",
    }
    return {
        "fused_category_scores": fused.float(),
        "winner_category_indices": winner,
        "visual_top1_category_indices": visual_top1,
        "visual_topk_indices": top_indices.long(),
        "enabled_mask": enabled,
        "reliability_pack": reliability_pack,
        "stats": stats,
    }


def detections_from_vmasc(
    cache: dict[str, Any],
    visual_category_scores: torch.Tensor,
    material_category_scores: torch.Tensor,
    result: dict[str, Any],
    prototypes_by_category: list[list[int]],
    category_ids: list[int],
    category_names: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Convert fused category winners into COCO detections."""

    visual = cache["visual_similarity"].float()
    proposal_meta = cache["proposal_meta"]
    prototype_meta = cache["prototype_meta"]
    fused_category = result["fused_category_scores"].float()
    winner = result["winner_category_indices"].long()
    visual_top1 = result["visual_top1_category_indices"].long()
    chosen_prototypes = choose_best_prototypes_for_winners(visual, winner, prototypes_by_category)
    row = torch.arange(visual.shape[0], dtype=torch.long)
    predicted = fused_category[row, winner]
    if fused_category.shape[1] > 1:
        mean_other = (fused_category.sum(dim=1) - predicted) / (fused_category.shape[1] - 1)
    else:
        mean_other = torch.zeros_like(predicted)
    dist_inter = predicted - mean_other
    visual_pred = visual_category_scores[row, visual_top1]
    dist_intra = predicted - visual_pred
    negative_indices = [
        int(proto.get("prototype_index", index))
        for index, proto in enumerate(prototype_meta)
        if proto.get("is_negative")
    ]
    if negative_indices:
        negative_best = visual.index_select(1, torch.tensor(negative_indices, dtype=torch.long)).max(dim=1).values
    else:
        negative_best = torch.full((visual.shape[0],), -1.0e9, dtype=torch.float32)

    detections: list[dict[str, Any]] = []
    invalid_proto = 0
    skipped_negative = 0
    for idx, proposal in enumerate(proposal_meta):
        proto_index = int(chosen_prototypes[idx].item())
        if proto_index < 0:
            invalid_proto += 1
            continue
        if float(negative_best[idx].item()) > float(visual[idx, proto_index].item()):
            skipped_negative += 1
            continue
        w = int(winner[idx].item())
        v = int(visual_top1[idx].item())
        score = float(proposal.get("detector_score", 0.0))
        detections.append(
            {
                "category_id": int(category_ids[w]),
                "image_id": int(proposal["image_id"]),
                "bbox": proposal["bbox"],
                "score": score,
                "dist_cosine_inter": float(dist_inter[idx].item()),
                "dist_cosine_intra": float(dist_intra[idx].item()),
                "score_rmdx": score,
                "rmdx_visual_similarity": float(visual_category_scores[idx, v].item()),
                "rmdx_material_score": float(material_category_scores[idx, w].item()),
                "rmdx_reliability_gate": float(result["enabled_mask"][idx].float().item()),
                "rmdx_fused_similarity": float(fused_category[idx, w].item()),
                "rmdx_mode": str(result["stats"]["method"]),
                "rmdx_candidate_id": str(result["stats"]["candidate_id"]),
                "rmdx_topk": int(result["stats"]["topk"]),
                "rmdx_lambda_max": float(result["stats"]["lambda_max"]),
                "rmdx_original_category_id": int(category_ids[v]),
                "rmdx_original_category_name": str(category_names[v]),
                "rmdx_original_score": score,
                "rmdx_selected_prototype_index": proto_index,
            }
        )
    summary = dict(result["stats"])
    summary.update({"detection_count": len(detections), "invalid_prototype_rows": invalid_proto, "skipped_negative_wins": skipped_negative})
    return detections, summary

