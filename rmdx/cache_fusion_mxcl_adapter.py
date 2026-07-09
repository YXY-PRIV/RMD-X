"""Detection writer for MXCL visual top-K reranking."""

from __future__ import annotations

from typing import Any

import torch

from rmdx.cache_fusion_xcl_adapter import detections_from_xcl
from rmdx.fcr_text_fusion import FCRTextConfig


def detections_from_mxcl(
    cache: dict[str, Any],
    visual_scores: torch.Tensor,
    mxcl_scores: torch.Tensor,
    category_ids: list[int],
    config: FCRTextConfig,
    policy_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    detections, stats = detections_from_xcl(cache, visual_scores, mxcl_scores, category_ids, config, policy_id)
    for det in detections:
        det["rmdx_mode"] = "mxcl_adapter_reranker"
    return detections, stats
