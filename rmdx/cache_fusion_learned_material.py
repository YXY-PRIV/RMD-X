"""Cache fusion entrypoint for Batch32 learned material similarity.

This file is external to original RAXO code. It swaps in a learned
material_class_similarity tensor while reusing the locked visual cache.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rmdx.cache_fusion_multichannel import run_fusion_with_external_material


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RMD-X cache fusion with learned material similarity")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--learned-material-similarity", required=True)
    parser.add_argument(
        "--mode",
        choices=["visual_only", "material_preargmax", "material_reliability_preargmax"],
        default="visual_only",
    )
    parser.add_argument("--lambda-material", type=float, default=0.0)
    parser.add_argument("--lambda-reliability", type=float, default=0.0)
    parser.add_argument("--output-path", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_fusion_with_external_material(
        cache_dir=args.cache_dir,
        material_similarity_path=args.learned_material_similarity,
        mode=args.mode,
        lambda_material=args.lambda_material,
        lambda_reliability=args.lambda_reliability,
        output_path=args.output_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

