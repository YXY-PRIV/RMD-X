"""Dataset-specific colour material similarity helpers for Batch35.

This module is external to RAXO. It uses detector proposal boxes and test
images only for inference-time material scoring, while the material banks are
built from safe train annotations in Batch34.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from rmdx.material import FEATURE_NAMES as LEGACY_COLOUR_FEATURE_NAMES
from rmdx.material_multichannel import bank_matrix_for_categories, cosine_class_similarity
from rmdx.material_multichannel import extract_legacy_colour_feature_fast


def load_json(path: str | Path) -> Any:
    with Path(path).open() as f:
        return json.load(f)


def save_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_material_bank(path: str | Path) -> dict[str, Any]:
    bank = load_json(path)
    if "feature_names" not in bank or "prototypes" not in bank:
        raise ValueError(f"Invalid material bank schema: {path}")
    feature_names = list(bank["feature_names"])
    if feature_names != LEGACY_COLOUR_FEATURE_NAMES:
        raise ValueError(
            f"Dataset-specific bank must use legacy 22D colour schema. "
            f"Got {len(feature_names)} features from {path}"
        )
    return bank


def _open_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


def extract_dataset_specific_proposal_features(
    proposal_meta: list[dict[str, Any]],
    image_dir: str | Path,
    feature_names: list[str],
    *,
    progress_prefix: str = "dataset_specific_material",
    progress_every: int = 200,
) -> torch.Tensor:
    """Extract 22D colour features for cached detector proposals.

    Proposals are grouped by file name so each test image is opened once.
    The bbox is the detector proposal bbox, not GT.
    """

    if feature_names != LEGACY_COLOUR_FEATURE_NAMES:
        raise ValueError("Only legacy 22D colour feature schema is supported in Batch35")
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(image_dir)
    by_file: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for idx, proposal in enumerate(proposal_meta):
        by_file[str(proposal["file_name"])].append((idx, proposal))

    features: list[list[float] | None] = [None] * len(proposal_meta)
    missing_images = []
    for file_idx, (file_name, rows) in enumerate(sorted(by_file.items())):
        image_path = image_dir / Path(file_name).name
        if not image_path.exists():
            missing_images.append(str(image_path))
            continue
        rgb = _open_rgb(image_path)
        for idx, proposal in rows:
            feat = extract_legacy_colour_feature_fast(rgb, proposal["bbox"])
            features[idx] = [float(v) if math.isfinite(float(v)) else 0.0 for v in feat]
        if (file_idx + 1) % progress_every == 0:
            print(
                f"[{progress_prefix}] extracted {file_idx + 1}/{len(by_file)} images; "
                f"proposal rows={sum(len(v) for _, v in list(by_file.items())[:file_idx+1])}",
                flush=True,
            )
    if missing_images:
        raise FileNotFoundError(f"Missing {len(missing_images)} proposal images, first: {missing_images[:5]}")
    missing_rows = sum(row is None for row in features)
    if missing_rows:
        raise RuntimeError(f"Missing extracted proposal feature rows: {missing_rows}")
    return torch.tensor(features, dtype=torch.float32)


def build_dataset_specific_material_similarity(
    *,
    cache_dir: str | Path,
    image_dir: str | Path,
    bank_path: str | Path,
    similarity_output: str | Path,
    meta_output: str | Path,
) -> dict[str, Any]:
    cache_dir = Path(cache_dir)
    bank_path = Path(bank_path)
    bank = load_material_bank(bank_path)
    feature_names = list(bank["feature_names"])
    proposal_meta = load_jsonl(cache_dir / "proposal_meta.jsonl")
    category_meta = load_json(cache_dir / "category_meta.json")
    proposal_features = extract_dataset_specific_proposal_features(
        proposal_meta,
        image_dir,
        feature_names,
        progress_prefix=f"dataset_specific:{bank.get('dataset', bank_path.stem)}",
    )
    class_features, class_meta = bank_matrix_for_categories(bank, category_meta)
    material_similarity = cosine_class_similarity(proposal_features, class_features, bank)
    similarity_output = Path(similarity_output)
    similarity_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(material_similarity, similarity_output)
    meta = {
        "cache_dir": str(cache_dir),
        "image_dir": str(image_dir),
        "bank_path": str(bank_path),
        "bank_name": bank.get("bank_name", bank_path.stem),
        "dataset": bank.get("dataset", ""),
        "feature_dim": len(feature_names),
        "feature_names": feature_names,
        "proposal_count": len(proposal_meta),
        "category_count": len(category_meta),
        "proposal_feature_shape": list(proposal_features.shape),
        "material_class_similarity_shape": list(material_similarity.shape),
        "missing_bank_entry_count": sum(1 for row in class_meta if row["missing_bank_entry"]),
        "low_sample_category_count": sum(1 for row in class_meta if row.get("low_sample_count")),
        "leakage_check": "no_full_test_gt_bbox_used_for_bank",
        "bank_leakage_check": bank.get("leakage_check", ""),
        "inference_uses_detector_proposals": True,
        "full_test_gt_bbox_used_for_bank": False,
    }
    save_json(meta_output, meta)
    return meta
