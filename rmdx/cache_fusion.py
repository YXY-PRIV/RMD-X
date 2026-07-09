"""Fuse cached RMD-X pre-argmax similarity tensors into COCO detections."""

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

from rmdx.preargmax_fusion import (
    compute_reliability_gate,
    fuse_similarity_material,
    fuse_similarity_material_reliability,
    fuse_similarity_visual_only,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RMD-X pre-argmax fusion from cache")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument(
        "--mode",
        choices=["visual_only", "material_preargmax", "material_reliability_preargmax"],
        default="visual_only",
    )
    parser.add_argument("--lambda-material", type=float, default=0.0)
    parser.add_argument("--lambda-reliability", type=float, default=0.0)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--replace-score-with-fused", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f)


def load_cache(cache_dir: str | Path) -> dict[str, Any]:
    cache_dir = Path(cache_dir)
    return {
        "cache_dir": cache_dir,
        "proposal_meta": load_jsonl(cache_dir / "proposal_meta.jsonl"),
        "prototype_meta": load_json(cache_dir / "prototype_meta.json"),
        "category_meta": load_json(cache_dir / "category_meta.json"),
        "summary": load_json(cache_dir / "cache_summary.json"),
        "visual_similarity": torch.load(cache_dir / "visual_similarity.pt", map_location="cpu").float(),
        "material_class_similarity": torch.load(
            cache_dir / "material_class_similarity.pt", map_location="cpu"
        ).float(),
    }


def material_class_to_prototype(
    material_class_similarity: torch.Tensor,
    prototype_meta: list[dict[str, Any]],
) -> torch.Tensor:
    cols = []
    for proto in prototype_meta:
        class_index = proto.get("material_class_index")
        if class_index is None:
            cols.append(torch.zeros(material_class_similarity.shape[0], dtype=torch.float32))
        else:
            cols.append(material_class_similarity[:, int(class_index)].float())
    if not cols:
        return torch.empty((material_class_similarity.shape[0], 0), dtype=torch.float32)
    return torch.stack(cols, dim=1)


def fuse_from_cache(
    cache: dict[str, Any],
    *,
    mode: str,
    lambda_material: float,
    lambda_reliability: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    visual = cache["visual_similarity"].float()
    material_proto = material_class_to_prototype(cache["material_class_similarity"], cache["prototype_meta"])
    detector_scores = torch.tensor(
        [float(row["detector_score"]) for row in cache["proposal_meta"]],
        dtype=torch.float32,
    )
    if mode == "visual_only":
        gate = torch.ones(visual.shape[0], dtype=torch.float32)
        fused = fuse_similarity_visual_only(visual)
    elif mode == "material_preargmax":
        gate = torch.ones(visual.shape[0], dtype=torch.float32)
        fused = fuse_similarity_material(visual, material_proto, lambda_material)
    else:
        fused, gate = fuse_similarity_material_reliability(
            visual,
            material_proto,
            detector_scores,
            lambda_material,
            lambda_reliability,
        )
    return fused.float(), material_proto.float(), gate.float()


def category_indices_for_dist(
    predicted_name: str,
    prototype_meta: list[dict[str, Any]],
    category_meta: list[dict[str, Any]],
) -> list[int]:
    indices = []
    categories = [str(cat["category_name"]) for cat in category_meta]
    for category_name in categories:
        if category_name == predicted_name:
            continue
        for proto in prototype_meta:
            if proto.get("category_name") == category_name:
                indices.append(int(proto["canonical_index"]))
                break
    return indices


def build_dist_index_lookup(
    prototype_meta: list[dict[str, Any]],
    category_meta: list[dict[str, Any]],
) -> dict[int, torch.Tensor]:
    """Precompute category-canonical indices used by dist_cosine_inter.

    The direct Batch14 implementation collects one canonical prototype per
    evaluation category except the predicted category. Doing this inside the
    per-detection loop is very expensive for full cache fusion, so the exact
    index lists are prepared once per predicted prototype index.
    """

    category_names = [str(cat["category_name"]) for cat in category_meta]
    first_canonical_by_category: dict[str, int] = {}
    for proto in prototype_meta:
        name = proto.get("category_name")
        if name is None:
            continue
        first_canonical_by_category.setdefault(str(name), int(proto["canonical_index"]))

    lookup: dict[int, torch.Tensor] = {}
    for proto in prototype_meta:
        proto_index = int(proto["prototype_index"])
        predicted_name = proto.get("category_name")
        if predicted_name is None:
            lookup[proto_index] = torch.empty(0, dtype=torch.long)
            continue
        indices = [
            first_canonical_by_category[name]
            for name in category_names
            if name != predicted_name and name in first_canonical_by_category
        ]
        lookup[proto_index] = torch.tensor(indices, dtype=torch.long)
    return lookup


def detections_from_fusion(
    cache: dict[str, Any],
    fused_similarity: torch.Tensor,
    material_similarity: torch.Tensor,
    reliability_gate: torch.Tensor,
    *,
    mode: str,
    lambda_material: float,
    lambda_reliability: float,
    replace_score_with_fused: bool = False,
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
        predicted_name = str(proto["category_name"])
        canonical_index = int(proto["canonical_index"])
        s_predicted = s_fused[canonical_index]
        other_indices = dist_index_lookup.get(max_index)
        if other_indices is not None and other_indices.numel() > 0:
            s_mean = s_fused[other_indices].mean()
        else:
            s_mean = torch.tensor(0.0)
        dist_cosine_inter = s_predicted - s_mean
        dist_cosine_intra = s_fused[max_index] - s_predicted
        score = float(proposal["detector_score"])
        if replace_score_with_fused:
            score = max(0.0, score * (1.0 + float(s_fused[max_index] - s_visual[visual_max])))
        visual_proto = prototype_meta[visual_max]
        visual_class = visual_proto.get("category_id")

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
                "rmdx_reliability_gate": float(reliability_gate[row_idx].item()),
                "rmdx_fused_similarity": float(s_fused[max_index].item()),
                "rmdx_mode": mode,
                "rmdx_lambda_material": float(lambda_material),
                "rmdx_lambda_reliability": float(lambda_reliability),
                "rmdx_original_category_id": None if visual_class is None else int(visual_class),
                "rmdx_original_score": float(proposal["detector_score"]),
            }
        )
    return detections


def run_fusion(args: argparse.Namespace) -> dict[str, Any]:
    cache = load_cache(args.cache_dir)
    fused, material, gate = fuse_from_cache(
        cache,
        mode=args.mode,
        lambda_material=args.lambda_material,
        lambda_reliability=args.lambda_reliability,
    )
    detections = detections_from_fusion(
        cache,
        fused,
        material,
        gate,
        mode=args.mode,
        lambda_material=args.lambda_material,
        lambda_reliability=args.lambda_reliability,
        replace_score_with_fused=args.replace_score_with_fused,
    )
    output_path = Path(args.output_path)
    save_json(output_path, detections)
    summary = {
        "output_path": str(output_path),
        "mode": args.mode,
        "lambda_material": args.lambda_material,
        "lambda_reliability": args.lambda_reliability,
        "proposal_count": len(cache["proposal_meta"]),
        "detection_count": len(detections),
        "visual_shape": list(cache["visual_similarity"].shape),
        "material_shape": list(cache["material_class_similarity"].shape),
    }
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> int:
    run_fusion(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
