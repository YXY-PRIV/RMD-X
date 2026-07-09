"""Convert FCR top-K reranking winners to COCO detections."""

from __future__ import annotations

from typing import Any

import torch


def detections_from_fcr(cache: dict[str, Any], visual_category_scores: torch.Tensor, result: dict, category_ids: list[int], config_id: str, encoder_type: str, topk: int, beta: float) -> list[dict[str, Any]]:
    fused = result["fused"].float()
    winner = result["winner"].long()
    visual_top1 = result["visual_top1"].long()
    row = torch.arange(fused.shape[0], dtype=torch.long)
    predicted = fused[row, winner]
    if fused.shape[1] > 1:
        mean_other = (fused.sum(dim=1) - predicted) / (fused.shape[1] - 1)
    else:
        mean_other = torch.zeros_like(predicted)
    dist_inter = predicted - mean_other
    visual_pred = visual_category_scores[row, visual_top1]
    dist_intra = predicted - visual_pred
    detections = []
    for idx, proposal in enumerate(cache["proposal_meta"]):
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
                "rmdx_mode": "fcr_crop_text_reranker",
                "rmdx_candidate_id": config_id,
                "rmdx_encoder_type": encoder_type,
                "rmdx_topk": int(topk),
                "rmdx_beta": float(beta),
                "rmdx_fused_similarity": float(predicted[idx].item()),
                "rmdx_visual_similarity": float(visual_pred[idx].item()),
                "rmdx_original_category_id": int(category_ids[v]),
                "rmdx_original_score": score,
            }
        )
    return detections

