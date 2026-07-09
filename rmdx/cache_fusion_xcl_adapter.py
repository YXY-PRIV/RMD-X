"""Detection writer for XCL-Adapter top-K reranking."""

from __future__ import annotations

from typing import Any

import torch

from rmdx.fcr_text_fusion import FCRTextConfig, fuse_fcr_text


def detections_from_xcl(cache: dict[str, Any], visual_scores: torch.Tensor, xcl_scores: torch.Tensor, category_ids: list[int], config: FCRTextConfig, policy_id: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    result = fuse_fcr_text(visual_scores, xcl_scores, config)
    fused = result["fused"].float()
    winner = result["winner"].long()
    visual_top1 = result["visual_top1"].long()
    rows = torch.arange(fused.shape[0], dtype=torch.long)
    pred = fused[rows, winner]
    mean_other = (fused.sum(dim=1) - pred) / max(1, fused.shape[1] - 1)
    visual_pred = fused[rows, visual_top1]
    detections = []
    for i, proposal in enumerate(cache["proposal_meta"]):
        w = int(winner[i].item())
        v = int(visual_top1[i].item())
        score = float(proposal.get("detector_score", 0.0))
        detections.append(
            {
                "category_id": int(category_ids[w]),
                "image_id": int(proposal["image_id"]),
                "bbox": proposal["bbox"],
                "score": score,
                "dist_cosine_inter": float((pred[i] - mean_other[i]).item()),
                "dist_cosine_intra": float((pred[i] - visual_pred[i]).item()),
                "score_rmdx": score,
                "rmdx_mode": "xcl_adapter_reranker",
                "rmdx_candidate_id": config.candidate_id,
                "rmdx_policy_id": policy_id,
                "rmdx_fused_similarity": float(pred[i].item()),
                "rmdx_visual_similarity": float(visual_pred[i].item()),
                "rmdx_original_category_id": int(category_ids[v]),
                "rmdx_original_score": score,
            }
        )
    return detections, result["stats"]
