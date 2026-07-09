"""Pre-argmax fusion utilities for RMD-X smoke experiments.

These helpers operate on proposal/class similarity tensors only. They do not
read GT annotations, do not alter bbox coordinates, and tolerate missing
material prototypes by assigning zero material evidence.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from rmdx.material import (
    compute_z_stats,
    cosine_similarity,
    extract_colour_feature_from_array,
    zscore_feature,
)


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable sigmoid wrapper."""

    return torch.sigmoid(x)


def zscore_tensor(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Return z-scored tensor, avoiding division by zero."""

    if x.numel() == 0:
        return x
    return (x - x.mean()) / x.std(unbiased=False).clamp(min=eps)


def center_tensor(x: torch.Tensor) -> torch.Tensor:
    """Center tensor by subtracting its mean."""

    if x.numel() == 0:
        return x
    return x - x.mean()


def compute_visual_margin(similarity: torch.Tensor) -> torch.Tensor:
    """Compute top1-top2 visual similarity margin per proposal."""

    if similarity.shape[1] < 2:
        return torch.zeros(similarity.shape[0], dtype=similarity.dtype, device=similarity.device)
    top2 = torch.topk(similarity, k=2, dim=1).values
    return top2[:, 0] - top2[:, 1]


def compute_reliability_gate(
    detector_scores: torch.Tensor,
    visual_margins: torch.Tensor,
) -> torch.Tensor:
    """Compute proposal-level gate from detector score and visual margin."""

    return sigmoid(zscore_tensor(detector_scores.float()) + zscore_tensor(visual_margins.float()))


def load_material_bank(path: str | Path) -> dict[str, Any]:
    """Load support material bank JSON."""

    with Path(path).open() as f:
        data = json.load(f)
    if "prototypes" not in data:
        raise KeyError(f"{path} missing prototypes")
    return data


def build_category_material_matrix(
    categories: list[str],
    prototype_to_category: list[str],
    material_bank: dict[str, Any],
) -> dict[str, Any]:
    """Build material lookup structures for class/prototype candidates.

    Missing category prototypes are represented as None and later mapped to
    zero material score.
    """

    proto_by_name = {
        str(proto.get("category_name")): proto
        for proto in material_bank.get("prototypes", {}).values()
        if proto.get("category_name") is not None
    }
    category_features = {}
    all_features = []
    for category in categories:
        proto = proto_by_name.get(str(category))
        if proto is None:
            category_features[category] = None
        else:
            feature = [float(v) for v in proto["mean_feature"]]
            category_features[category] = feature
            all_features.append(feature)
    stats_source = all_features if all_features else [[0.0] * 22]
    stats = compute_z_stats(stats_source)
    proto_category_indices = [str(category) for category in prototype_to_category]
    return {
        "category_features": category_features,
        "stats": stats,
        "prototype_categories": proto_category_indices,
    }


def _proposal_features(
    annotations: list[dict[str, Any]],
    image_path: str | Path,
) -> list[list[float]]:
    image_cache: dict[int, np.ndarray] = {}
    features = []
    base = Path(image_path)
    for ann in annotations:
        image_id = int(ann["image_id"])
        file_name = ann.get("file_name")
        if file_name is None:
            raise KeyError("annotation is missing file_name; dataset wrapper must provide it")
        if image_id not in image_cache:
            image_cache[image_id] = np.asarray(Image.open(base / file_name).convert("RGB"), dtype=np.uint8)
        features.append(extract_colour_feature_from_array(image_cache[image_id], ann["bbox"]))
    return features


def compute_material_similarity_for_proposals(
    annotations: list[dict[str, Any]],
    image_path: str | Path,
    material_lookup: dict[str, Any],
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Compute proposal x prototype material similarity matrix.

    Similarity is cosine between z-scored proposal colour feature and category
    material prototype. Missing category material evidence gets score 0.
    """

    features = _proposal_features(annotations, image_path)
    if not features:
        return torch.empty((0, len(material_lookup["prototype_categories"])), device=device)
    category_features = material_lookup["category_features"]
    stats = material_lookup["stats"]
    proto_categories = material_lookup["prototype_categories"]
    rows = []
    for feature in features:
        z_feature = zscore_feature(feature, stats)
        values = []
        for category in proto_categories:
            proto_feature = category_features.get(category)
            if proto_feature is None or category == "negative":
                values.append(0.0)
            else:
                values.append(cosine_similarity(z_feature, zscore_feature(proto_feature, stats)))
        rows.append(values)
    return torch.tensor(rows, dtype=torch.float32, device=device)


def fuse_similarity_visual_only(similarity: torch.Tensor) -> torch.Tensor:
    """Return unmodified visual similarity."""

    return similarity


def fuse_similarity_material(
    similarity: torch.Tensor,
    material_similarity: torch.Tensor,
    lambda_material: float,
) -> torch.Tensor:
    """Fuse visual similarity with centered material similarity."""

    centered = material_similarity - material_similarity.mean(dim=1, keepdim=True)
    return similarity + float(lambda_material) * centered.to(similarity.device)


def fuse_similarity_material_reliability(
    similarity: torch.Tensor,
    material_similarity: torch.Tensor,
    detector_scores: torch.Tensor,
    lambda_material: float,
    lambda_reliability: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse material with proposal-level reliability-gated visual confidence."""

    visual_margins = compute_visual_margin(similarity)
    gate = compute_reliability_gate(detector_scores.to(similarity.device), visual_margins)
    effective_gate = (1.0 - float(lambda_reliability)) + float(lambda_reliability) * gate
    centered = material_similarity.to(similarity.device) - material_similarity.to(similarity.device).mean(
        dim=1, keepdim=True
    )
    fused = similarity + float(lambda_material) * effective_gate.unsqueeze(1) * centered
    return fused, gate
