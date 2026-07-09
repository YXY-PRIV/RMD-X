"""Read-only helpers for RAXO prototype files."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import torch


def load_prototypes(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu")


def summarize_prototypes(path: Path) -> dict[str, Any]:
    data = load_prototypes(path)
    supercategory_counts: Counter[str] = Counter()
    shape_counts: Counter[str] = Counter()
    dtype_counts: Counter[str] = Counter()
    negative_count = 0
    feature_dims: set[int] = set()
    fields: Counter[str] = Counter()

    for key, value in data.items():
        if not isinstance(value, dict):
            continue
        fields.update(value.keys())
        supercategory = str(value.get("supercategory"))
        supercategory_counts[supercategory] += 1
        is_negative = "neg_" in key or key == "negative" or supercategory == "negative"
        if is_negative:
            negative_count += 1
        prototype = value.get("prototype")
        if prototype is not None:
            shape = tuple(prototype.shape)
            shape_counts[str(shape)] += 1
            dtype_counts[str(prototype.dtype)] += 1
            if shape:
                feature_dims.add(int(shape[-1]))

    return {
        "path": str(path),
        "top_type": type(data).__name__,
        "prototype_count": len(data),
        "supercategory_count": len(supercategory_counts),
        "feature_dims": sorted(feature_dims),
        "feature_dim": sorted(feature_dims)[0] if len(feature_dims) == 1 else None,
        "negative_prototype_count": negative_count,
        "value_fields": dict(fields),
        "prototype_shapes": dict(shape_counts),
        "prototype_dtypes": dict(dtype_counts),
        "per_supercategory_top20": supercategory_counts.most_common(20),
        "first_keys": list(data.keys())[:20],
    }

