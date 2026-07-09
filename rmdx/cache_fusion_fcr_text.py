"""Detection writer for FCR text fusion candidates."""

from __future__ import annotations

from typing import Any

import torch


def detections_from_fcr_text(cache: dict[str, Any], result: dict[str, Any], category_ids: list[int], candidate_id: str, encoder_type: str) -> list[dict[str, Any]]:
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
                "rmdx_mode": "fcr_text_crop_text_reranker",
                "rmdx_candidate_id": candidate_id,
                "rmdx_encoder_type": encoder_type,
                "rmdx_fused_similarity": float(pred[i].item()),
                "rmdx_visual_similarity": float(visual_pred[i].item()),
                "rmdx_original_category_id": int(category_ids[v]),
                "rmdx_original_score": score,
            }
        )
    return detections
