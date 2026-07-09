"""Cache fusion entrypoint for external multi-channel material similarity.

The visual cache remains the locked Batch17/18 cache. This module only swaps
in a bank-specific material_class_similarity tensor built from support
material banks and inference-time proposal crops.
"""

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

from rmdx.cache_fusion import detections_from_fusion, fuse_from_cache, load_json, load_jsonl, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RMD-X cache fusion with external multi-channel material")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--material-similarity", required=True)
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


def load_cache_with_external_material(cache_dir: str | Path, material_similarity_path: str | Path) -> dict[str, Any]:
    cache_dir = Path(cache_dir)
    material = torch.load(material_similarity_path, map_location="cpu").float()
    return {
        "cache_dir": cache_dir,
        "proposal_meta": load_jsonl(cache_dir / "proposal_meta.jsonl"),
        "prototype_meta": load_json(cache_dir / "prototype_meta.json"),
        "category_meta": load_json(cache_dir / "category_meta.json"),
        "summary": load_json(cache_dir / "cache_summary.json"),
        "visual_similarity": torch.load(cache_dir / "visual_similarity.pt", map_location="cpu").float(),
        "material_class_similarity": material,
        "material_similarity_path": str(material_similarity_path),
    }


def run_fusion_with_external_material(
    *,
    cache_dir: str | Path,
    material_similarity_path: str | Path,
    mode: str,
    lambda_material: float,
    lambda_reliability: float,
    output_path: str | Path,
    replace_score_with_fused: bool = False,
) -> dict[str, Any]:
    cache = load_cache_with_external_material(cache_dir, material_similarity_path)
    fused, material, gate = fuse_from_cache(
        cache,
        mode=mode,
        lambda_material=lambda_material,
        lambda_reliability=lambda_reliability,
    )
    detections = detections_from_fusion(
        cache,
        fused,
        material,
        gate,
        mode=mode,
        lambda_material=lambda_material,
        lambda_reliability=lambda_reliability,
        replace_score_with_fused=replace_score_with_fused,
    )
    output_path = Path(output_path)
    save_json(output_path, detections)
    changed_category_count = sum(
        1
        for det in detections
        if det.get("rmdx_original_category_id") is not None
        and int(det["category_id"]) != int(det["rmdx_original_category_id"])
    )
    summary = {
        "output_path": str(output_path),
        "mode": mode,
        "lambda_material": float(lambda_material),
        "lambda_reliability": float(lambda_reliability),
        "proposal_count": len(cache["proposal_meta"]),
        "detection_count": len(detections),
        "changed_category_count": changed_category_count,
        "changed_category_ratio": changed_category_count / max(len(detections), 1),
        "visual_shape": list(cache["visual_similarity"].shape),
        "material_shape": list(cache["material_class_similarity"].shape),
        "material_similarity_path": str(material_similarity_path),
    }
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> int:
    args = parse_args()
    run_fusion_with_external_material(
        cache_dir=args.cache_dir,
        material_similarity_path=args.material_similarity,
        mode=args.mode,
        lambda_material=args.lambda_material,
        lambda_reliability=args.lambda_reliability,
        output_path=args.output_path,
        replace_score_with_fused=args.replace_score_with_fused,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
