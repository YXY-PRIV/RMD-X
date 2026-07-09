"""Centralized paths for RMD-X read-only contract checks."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RMDXPaths:
    repo_root: Path
    results_path: Path
    dataset_path: Path
    det_compass_annotation: Path
    det_compass_image_dir: Path
    prototypes_path: Path
    with_masks_path: Path
    known_final_path: Path
    known_final_uncertanty_path: Path
    output_root: Path
    contracts_dir: Path
    smoke_dir: Path
    rmdx_results_dir: Path
    docs_dir: Path


def default_paths() -> RMDXPaths:
    repo_root = Path(os.environ.get("RMDX_REPO_ROOT", "/workspace/RAXO"))
    results_path = Path(os.environ.get("RMDX_RESULTS_PATH", "/models/ICCV25_experimentation"))
    dataset_path = Path(os.environ.get("RMDX_DATASET_PATH", "/datasets/xray-datasets"))
    output_root = Path(os.environ.get("RMDX_OUTPUT_ROOT", str(repo_root / "outputs_rmdx")))

    dataset = "DET-COMPASS"
    detector = "groundingDINO"
    det_dir = results_path / "initial_detections" / detector

    return RMDXPaths(
        repo_root=repo_root,
        results_path=results_path,
        dataset_path=dataset_path,
        det_compass_annotation=dataset_path / dataset / "annotations" / "full_test.json",
        det_compass_image_dir=dataset_path / dataset / "test",
        prototypes_path=results_path / "known_prototypes_sam2.pt",
        with_masks_path=det_dir / f"coco_results.bbox_{dataset}_with_masks.json",
        known_final_path=det_dir / f"coco_results.bbox_{dataset}_with_masks_nms_0.8_our_method_known_final.json",
        known_final_uncertanty_path=det_dir / f"coco_results.bbox_{dataset}_with_masks_nms_0.8_our_method_known_final_uncertanty.json",
        output_root=output_root,
        contracts_dir=output_root / "contracts",
        smoke_dir=output_root / "smoke",
        rmdx_results_dir=output_root / "rmdx_results",
        docs_dir=repo_root / "docs_rmdx",
    )

