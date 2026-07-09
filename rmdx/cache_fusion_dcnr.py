#!/usr/bin/env python3
"""Run DCNR-RMD-X fusion from cached RMD-X tensors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rmdx.cache_fusion import load_cache, save_json
from rmdx.dcnr_fusion import DcnrConfig, fuse_dcnr
from rmdx.visual_topk_material_fusion import build_category_maps, category_scores_from_prototypes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DCNR fusion from cache")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--material-similarity", default="")
    parser.add_argument("--summary-path", default="")
    return parser.parse_args()


def load_material_similarity(cache: dict[str, Any], override: str) -> torch.Tensor:
    if override:
        return torch.load(Path(override), map_location="cpu").float()
    return cache["material_class_similarity"].float()


def canonical_prototypes_by_category(
    prototype_meta: list[dict[str, Any]],
    category_meta: list[dict[str, Any]],
) -> torch.Tensor:
    category_id_to_index = {int(category["category_id"]): index for index, category in enumerate(category_meta)}
    canonical = torch.full((len(category_meta),), -1, dtype=torch.long)
    for fallback_index, proto in enumerate(prototype_meta):
        if proto.get("is_negative"):
            continue
        category_id = proto.get("category_id")
        if category_id is None:
            continue
        category_index = category_id_to_index.get(int(category_id))
        if category_index is None or canonical[category_index] >= 0:
            continue
        canonical[category_index] = int(proto.get("canonical_index", proto.get("prototype_index", fallback_index)))
    missing = [idx for idx, value in enumerate(canonical.tolist()) if value < 0]
    if missing:
        raise ValueError(f"Missing canonical prototypes for category indices: {missing[:10]}")
    return canonical


def choose_best_prototypes(
    visual_similarity: torch.Tensor,
    winner_category_indices: torch.Tensor,
    prototypes_by_category: list[list[int]],
) -> torch.Tensor:
    chosen = torch.full((visual_similarity.shape[0],), -1, dtype=torch.long)
    for category_index, proto_indices in enumerate(prototypes_by_category):
        rows = (winner_category_indices == category_index).nonzero(as_tuple=False).view(-1)
        if rows.numel() == 0 or not proto_indices:
            continue
        idx = torch.tensor(proto_indices, dtype=torch.long)
        local = visual_similarity.index_select(0, rows).index_select(1, idx)
        best_local = torch.argmax(local, dim=1)
        chosen[rows] = idx[best_local]
    return chosen


def output_detections(
    cache: dict[str, Any],
    material_category_scores: torch.Tensor,
    result,
    *,
    config: DcnrConfig,
    prototypes_by_category: list[list[int]],
    category_ids: list[int],
    category_names: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    visual = cache["visual_similarity"].float()
    proposal_meta = cache["proposal_meta"]
    prototype_meta = cache["prototype_meta"]
    category_meta = cache["category_meta"]
    canonical_by_category = canonical_prototypes_by_category(prototype_meta, category_meta)
    visual_category_scores = category_scores_from_prototypes(visual, prototypes_by_category)
    chosen_prototypes = choose_best_prototypes(visual, result.winner_category_indices, prototypes_by_category)

    row_indices = torch.arange(visual.shape[0], dtype=torch.long)
    winner = result.winner_category_indices.long()
    visual_top1 = result.visual_top1_category_indices.long()
    canonical_scores = result.fused_category_scores.float()
    predicted_scores = canonical_scores[row_indices, winner]
    if canonical_scores.shape[1] > 1:
        mean_other = (canonical_scores.sum(dim=1) - predicted_scores) / (canonical_scores.shape[1] - 1)
    else:
        mean_other = torch.zeros_like(predicted_scores)
    dist_inter = predicted_scores - mean_other

    selected_scores = visual[row_indices, chosen_prototypes.clamp(min=0)]
    predicted_canonical_visual = visual.index_select(1, canonical_by_category)[row_indices, winner]
    dist_intra = selected_scores - predicted_canonical_visual
    negative_indices = [
        int(proto.get("prototype_index", index))
        for index, proto in enumerate(prototype_meta)
        if proto.get("is_negative")
    ]
    if negative_indices:
        negative_best = visual.index_select(1, torch.tensor(negative_indices, dtype=torch.long)).max(dim=1).values
    else:
        negative_best = torch.full_like(selected_scores, -1.0e9)

    detections: list[dict[str, Any]] = []
    invalid_proto = 0
    skipped_negative_wins = 0
    for row_index, proposal in enumerate(proposal_meta):
        proto_index = int(chosen_prototypes[row_index].item())
        if proto_index < 0:
            invalid_proto += 1
            continue
        if float(negative_best[row_index].item()) > float(selected_scores[row_index].item()):
            skipped_negative_wins += 1
            continue
        winner_index = int(winner[row_index].item())
        visual_index = int(visual_top1[row_index].item())
        score = float(proposal["detector_score"])
        detections.append(
            {
                "category_id": int(category_ids[winner_index]),
                "image_id": int(proposal["image_id"]),
                "bbox": proposal["bbox"],
                "score": score,
                "dist_cosine_inter": float(dist_inter[row_index].item()),
                "dist_cosine_intra": float(dist_intra[row_index].item()),
                "score_rmdx": score,
                "rmdx_visual_similarity": float(visual_category_scores[row_index, visual_index].item()),
                "rmdx_material_score": float(material_category_scores[row_index, winner_index].item()),
                "rmdx_reliability_gate": 1.0,
                "rmdx_fused_similarity": float(result.fused_category_scores[row_index, winner_index].item()),
                "rmdx_mode": config.method,
                "rmdx_candidate_id": config.candidate_id,
                "rmdx_topk": int(config.topk),
                "rmdx_alpha": float(config.alpha),
                "rmdx_tau": float(config.tau),
                "rmdx_margin": float(config.margin),
                "rmdx_original_category_id": int(category_ids[visual_index]),
                "rmdx_original_category_name": str(category_names[visual_index]),
                "rmdx_original_score": score,
                "rmdx_selected_prototype_index": proto_index,
            }
        )
    summary = dict(result.stats)
    summary.update(
        {
            "detection_count": len(detections),
            "invalid_prototype_rows": invalid_proto,
            "skipped_negative_wins": skipped_negative_wins,
            "visual_shape": list(visual.shape),
            "material_shape": list(material_category_scores.shape),
        }
    )
    return detections, summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    cache = load_cache(args.cache_dir)
    material = load_material_similarity(cache, args.material_similarity)
    prototypes_by_category, _proto_to_category, category_ids, category_names = build_category_maps(
        cache["prototype_meta"], cache["category_meta"]
    )
    visual_category_scores = category_scores_from_prototypes(cache["visual_similarity"].float(), prototypes_by_category)
    if material.shape != visual_category_scores.shape:
        raise ValueError(f"material shape {tuple(material.shape)} does not match category scores {tuple(visual_category_scores.shape)}")
    config = DcnrConfig(
        candidate_id=args.candidate_id,
        method=args.method,
        topk=args.topk,
        alpha=args.alpha,
        tau=args.tau,
        margin=args.margin,
    )
    result = fuse_dcnr(visual_category_scores, material, config)
    detections, summary = output_detections(
        cache,
        material,
        result,
        config=config,
        prototypes_by_category=prototypes_by_category,
        category_ids=category_ids,
        category_names=category_names,
    )
    output_path = Path(args.output_path)
    save_json(output_path, detections)
    summary.update(
        {
            "cache_dir": str(args.cache_dir),
            "material_similarity": args.material_similarity or str(Path(args.cache_dir) / "material_class_similarity.pt"),
            "output_path": str(output_path),
        }
    )
    summary_path = Path(args.summary_path) if args.summary_path else output_path.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
