"""Adaptive-gate cache fusion for RMD-X.

This module fuses existing visual/material cache tensors. It does not build
features, run DINOv2, read GT annotations, or modify baseline files.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rmdx.adaptive_gating import GateConfig, compute_gate_weights, entropy_from_scores, gate_summary, top2_scores
from rmdx.cache_fusion import (
    build_dist_index_lookup,
    load_json,
    load_jsonl,
    material_class_to_prototype,
    save_json,
)


def load_cache(cache_dir: str | Path, material_similarity_path: str | Path | None = None) -> dict[str, Any]:
    cache_dir = Path(cache_dir)
    visual = torch.load(cache_dir / "visual_similarity.pt", map_location="cpu").float()
    if material_similarity_path is None:
        material = torch.load(cache_dir / "material_class_similarity.pt", map_location="cpu").float()
    else:
        material = torch.load(material_similarity_path, map_location="cpu").float()
    if visual.shape[0] != material.shape[0]:
        raise ValueError(f"proposal count mismatch: visual={tuple(visual.shape)} material={tuple(material.shape)}")
    return {
        "cache_dir": cache_dir,
        "proposal_meta": load_jsonl(cache_dir / "proposal_meta.jsonl"),
        "prototype_meta": load_json(cache_dir / "prototype_meta.json"),
        "category_meta": load_json(cache_dir / "category_meta.json"),
        "summary": load_json(cache_dir / "cache_summary.json"),
        "visual_similarity": visual,
        "material_class_similarity": material,
    }


def sample_counts_from_bank(bank_path: str | Path | None, category_meta: list[dict[str, Any]]) -> dict[int, int]:
    if not bank_path:
        return {}
    path = Path(bank_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        bank = json.load(f)
    out: dict[int, int] = {}
    for proto in bank.get("prototypes", {}).values():
        cid = proto.get("category_id_for_eval", proto.get("category_id"))
        if cid is None:
            continue
        out[int(cid)] = int(proto.get("sample_count", 0) or 0)
    # Ensure known categories have at least zero.
    for cat in category_meta:
        out.setdefault(int(cat["category_id"]), 0)
    return out


def prototype_category_ids(prototype_meta: list[dict[str, Any]]) -> torch.Tensor:
    values = []
    for proto in prototype_meta:
        cid = proto.get("category_id")
        values.append(-1 if cid is None else int(cid))
    return torch.tensor(values, dtype=torch.long)


def class_category_ids(category_meta: list[dict[str, Any]]) -> torch.Tensor:
    return torch.tensor([int(cat["category_id"]) for cat in category_meta], dtype=torch.long)


def build_gate_feature_tensors(cache: dict[str, Any], bank_path: str | Path | None = None) -> dict[str, torch.Tensor]:
    visual = cache["visual_similarity"].float()
    material_class = cache["material_class_similarity"].float()
    v1, vi1, v2, _ = top2_scores(visual)
    m1, mi1, m2, _ = top2_scores(material_class)
    visual_margin = v1 - v2
    material_margin = m1 - m2
    proto_cids = prototype_category_ids(cache["prototype_meta"])
    class_cids = class_category_ids(cache["category_meta"])
    visual_cids = proto_cids[vi1]
    material_cids = class_cids[mi1]
    class_index_by_cid = {int(cat["category_id"]): int(cat["class_index"]) for cat in cache["category_meta"]}
    rows = torch.arange(visual.shape[0], dtype=torch.long)
    material_for_visual = torch.zeros(visual.shape[0], dtype=torch.float32)
    has_visual_class = torch.zeros(visual.shape[0], dtype=torch.bool)
    for cid, class_idx in class_index_by_cid.items():
        mask = visual_cids == int(cid)
        if mask.any():
            material_for_visual[mask] = material_class[mask, class_idx]
            has_visual_class[mask] = True
    sample_counts = sample_counts_from_bank(bank_path, cache["category_meta"])
    sample_count_tensor = torch.zeros(visual.shape[0], dtype=torch.float32)
    for cid, count in sample_counts.items():
        mask = visual_cids == int(cid)
        if mask.any():
            sample_count_tensor[mask] = float(count)
    detector_scores = torch.tensor([float(row.get("detector_score", 0.0)) for row in cache["proposal_meta"]], dtype=torch.float32)
    return {
        "visual_top1_score": v1.float(),
        "visual_top2_score": v2.float(),
        "visual_margin": visual_margin.float(),
        "visual_entropy": entropy_from_scores(visual.float()),
        "detector_score": detector_scores,
        "material_top1_score": m1.float(),
        "material_top2_score": m2.float(),
        "material_margin": material_margin.float(),
        "material_entropy": entropy_from_scores(material_class.float()),
        "visual_top1_category_id": visual_cids,
        "material_top1_category_id": material_cids,
        "visual_material_agree": (visual_cids == material_cids) & (visual_cids >= 0),
        "material_score_for_visual_category": material_for_visual,
        "material_vs_visual_category_gap": (m1.float() - material_for_visual),
        "class_sample_count": sample_count_tensor,
        "class_low_sample_flag": sample_count_tensor < 5,
        "has_visual_class": has_visual_class,
        "row_index": rows,
    }


def fuse_with_adaptive_gate(
    cache: dict[str, Any],
    config: GateConfig,
    bank_path: str | Path | None = None,
    features: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    visual = cache["visual_similarity"].float()
    material_proto = material_class_to_prototype(cache["material_class_similarity"], cache["prototype_meta"]).float()
    if features is None:
        features = build_gate_feature_tensors(cache, bank_path)
    weights = compute_gate_weights(features, config).float()
    centered = material_proto - material_proto.mean(dim=1, keepdim=True)
    fused = visual + weights.unsqueeze(1) * centered
    return fused.float(), material_proto.float(), weights, gate_summary(weights, config)


def detections_from_adaptive_fusion(
    cache: dict[str, Any],
    fused_similarity: torch.Tensor,
    material_similarity: torch.Tensor,
    material_weights: torch.Tensor,
    *,
    config: GateConfig,
) -> list[dict[str, Any]]:
    visual = cache["visual_similarity"].float()
    proposal_meta = cache["proposal_meta"]
    prototype_meta = cache["prototype_meta"]
    dist_index_lookup = build_dist_index_lookup(prototype_meta, cache["category_meta"])
    detections = []
    for row_idx, proposal in enumerate(proposal_meta):
        s_visual = visual[row_idx]
        s_fused = fused_similarity[row_idx]
        s_material = material_similarity[row_idx]
        visual_max = int(torch.argmax(s_visual).item())
        max_index = int(torch.argmax(s_fused).item())
        proto = prototype_meta[max_index]
        if proto.get("is_negative"):
            continue
        category_id = int(proto["category_id"])
        canonical_index = int(proto["canonical_index"])
        s_predicted = s_fused[canonical_index]
        other_indices = dist_index_lookup.get(max_index)
        if other_indices is not None and other_indices.numel() > 0:
            s_mean = s_fused[other_indices].mean()
        else:
            s_mean = torch.tensor(0.0)
        dist_cosine_inter = s_predicted - s_mean
        dist_cosine_intra = s_fused[max_index] - s_predicted
        visual_proto = prototype_meta[visual_max]
        visual_class = visual_proto.get("category_id")
        score = float(proposal["detector_score"])
        detections.append(
            {
                "category_id": category_id,
                "image_id": int(proposal["image_id"]),
                "bbox": proposal["bbox"],
                "score": score,
                "dist_cosine_inter": float(dist_cosine_inter.item()),
                "dist_cosine_intra": float(dist_cosine_intra.item()),
                "score_rmdx": score,
                "rmdx_visual_similarity": float(s_visual[visual_max].item()),
                "rmdx_material_score": float(s_material[max_index].item()),
                "rmdx_material_weight": float(material_weights[row_idx].item()),
                "rmdx_fused_similarity": float(s_fused[max_index].item()),
                "rmdx_mode": "adaptive_material_gate",
                "rmdx_gate_name": config.gate_name,
                "rmdx_lambda_material": float(config.lambda_material),
                "rmdx_margin_threshold": config.margin_threshold,
                "rmdx_material_threshold": config.material_threshold,
                "rmdx_sample_threshold": config.sample_threshold,
                "rmdx_original_category_id": None if visual_class is None else int(visual_class),
                "rmdx_original_score": float(proposal["detector_score"]),
            }
        )
    return detections


def save_adaptive_output(
    cache: dict[str, Any],
    config: GateConfig,
    output_path: str | Path,
    bank_path: str | Path | None = None,
    features: dict[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    fused, material, weights, summary = fuse_with_adaptive_gate(cache, config, bank_path, features=features)
    detections = detections_from_adaptive_fusion(cache, fused, material, weights, config=config)
    save_json(Path(output_path), detections)
    summary.update(
        {
            "output_path": str(output_path),
            "proposal_count": len(cache["proposal_meta"]),
            "detection_count": len(detections),
            "visual_shape": list(cache["visual_similarity"].shape),
            "material_shape": list(cache["material_class_similarity"].shape),
        }
    )
    return summary
