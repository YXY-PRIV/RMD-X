"""Fixed adaptive-gate cache fusion helpers for Batch41.1."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rmdx.adaptive_gate_stability import FixedGateCandidate, instantiate_candidate
from rmdx.cache_fusion_adaptive_gate import build_gate_feature_tensors, load_cache, save_adaptive_output


def run_fixed_candidate(
    *,
    cache_dir: str | Path,
    output_path: str | Path,
    candidate: FixedGateCandidate,
    material_similarity_path: str | Path | None,
    bank_path: str | Path | None,
) -> dict[str, Any]:
    """Run one locked fixed gate candidate from an existing cache."""

    cache = load_cache(cache_dir, material_similarity_path)
    features = build_gate_feature_tensors(cache, bank_path)
    config, resolved = instantiate_candidate(candidate, features)
    summary = save_adaptive_output(cache, config, output_path, bank_path, features=features)
    summary.update(
        {
            "candidate_id": candidate.candidate_id,
            "threshold_type": candidate.threshold_type,
            "threshold_value": resolved.get("threshold_value"),
            "material_threshold_value": resolved.get("material_threshold_value"),
            "selected_from_grid": candidate.selected_from_grid,
            "uses_gt": candidate.uses_gt,
            "fair_method_candidate": candidate.fair_method_candidate,
        }
    )
    return summary
