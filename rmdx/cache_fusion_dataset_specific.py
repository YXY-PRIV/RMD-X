"""Cache fusion using an external dataset-specific material similarity tensor."""

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

from rmdx.cache_fusion import detections_from_fusion, fuse_from_cache, load_json, load_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cache fusion with dataset-specific material similarity")
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


def save_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f)


def load_cache(cache_dir: str | Path, material_similarity_path: str | Path) -> dict[str, Any]:
    cache_dir = Path(cache_dir)
    material = torch.load(material_similarity_path, map_location="cpu").float()
    visual = torch.load(cache_dir / "visual_similarity.pt", map_location="cpu").float()
    if material.shape[0] != visual.shape[0]:
        raise ValueError(f"proposal count mismatch: material={tuple(material.shape)} visual={tuple(visual.shape)}")
    return {
        "cache_dir": cache_dir,
        "proposal_meta": load_jsonl(cache_dir / "proposal_meta.jsonl"),
        "prototype_meta": load_json(cache_dir / "prototype_meta.json"),
        "category_meta": load_json(cache_dir / "category_meta.json"),
        "summary": load_json(cache_dir / "cache_summary.json"),
        "visual_similarity": visual,
        "material_class_similarity": material,
    }


def main() -> int:
    args = parse_args()
    cache = load_cache(args.cache_dir, args.material_similarity)
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
    save_json(args.output_path, detections)
    print(
        json.dumps(
            {
                "output_path": args.output_path,
                "mode": args.mode,
                "lambda_material": args.lambda_material,
                "lambda_reliability": args.lambda_reliability,
                "proposal_count": len(cache["proposal_meta"]),
                "detection_count": len(detections),
                "visual_shape": list(cache["visual_similarity"].shape),
                "material_shape": list(cache["material_class_similarity"].shape),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
