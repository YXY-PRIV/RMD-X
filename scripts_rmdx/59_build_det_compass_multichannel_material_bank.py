#!/usr/bin/env python3
"""Batch27: build DET-COMPASS multi-channel support material banks."""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("/workspace/RAXO")
if not ROOT.exists():
    ROOT = Path("/mnt/data/yxy/program/RAXO")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rmdx.material import FEATURE_NAMES as LEGACY_COLOUR_FEATURE_NAMES


OUT_ROOT = ROOT / "outputs_rmdx" / "material_multichannel_det_compass"
FEATURE_DIR = OUT_ROOT / "features"
BANK_DIR = OUT_ROOT / "banks"
FEATURE_PARQUET = FEATURE_DIR / "support_multichannel_object_features.parquet"
CHANNELS = ["colour", "grey", "high", "low", "density"]
GENERIC_FEATURE_NAMES = [
    "mean",
    "std",
    "min",
    "max",
    "median",
    "q10",
    "q25",
    "q75",
    "q90",
    "entropy",
    "nonzero_ratio",
    "bbox_area_ratio",
    "bbox_aspect_ratio",
    "crop_width",
    "crop_height",
    "edge_density",
    "gradient_mean",
    "gradient_std",
]
CROSS_FEATURE_NAMES = [
    "high_minus_low_mean",
    "high_low_ratio_mean",
    "density_minus_grey_mean",
    "colour_gray_minus_density_mean",
    "channel_mean_colour",
    "channel_mean_grey",
    "channel_mean_high",
    "channel_mean_low",
    "channel_mean_density",
    "channel_std_colour",
    "channel_std_grey",
    "channel_std_high",
    "channel_std_low",
    "channel_std_density",
    "channel_contrast_colour",
    "channel_contrast_grey",
    "channel_contrast_high",
    "channel_contrast_low",
    "channel_contrast_density",
    "high_low_contrast",
]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def clean_vector(values: np.ndarray) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    values = np.where(np.isfinite(values), values, 0.0)
    return [float(v) for v in values.tolist()]


def stats_for_features(arr: np.ndarray) -> dict[str, list[float]]:
    if arr.size == 0:
        return {"mean": [], "std": []}
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    std = np.where(std == 0, 1.0, std)
    return {"mean": clean_vector(mean), "std": clean_vector(std)}


def feature_columns(bank_type: str) -> tuple[list[str], list[str], list[str]]:
    if bank_type == "colour":
        cols = [f"colour__legacy_{name}" for name in LEGACY_COLOUR_FEATURE_NAMES]
        return cols, list(LEGACY_COLOUR_FEATURE_NAMES), ["colour"]
    if bank_type in {"grey", "high", "low", "density"}:
        cols = [f"{bank_type}__{name}" for name in GENERIC_FEATURE_NAMES]
        return cols, list(GENERIC_FEATURE_NAMES), [bank_type]
    if bank_type == "multi_channel_all":
        cols: list[str] = []
        names: list[str] = []
        for channel in CHANNELS:
            cols.extend([f"{channel}__{name}" for name in GENERIC_FEATURE_NAMES])
            names.extend([f"{channel}__{name}" for name in GENERIC_FEATURE_NAMES])
        cols.extend([f"cross__{name}" for name in CROSS_FEATURE_NAMES])
        names.extend([f"cross__{name}" for name in CROSS_FEATURE_NAMES])
        return cols, names, CHANNELS
    raise KeyError(bank_type)


def bank_filename(bank_type: str) -> str:
    if bank_type == "multi_channel_all":
        return "material_bank_multichannel_all_support.json"
    return f"material_bank_{bank_type}_support.json"


def build_bank(df: pd.DataFrame, bank_type: str) -> tuple[dict[str, Any], dict[str, Any]]:
    cols, names, channels = feature_columns(bank_type)
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise KeyError(f"{bank_type} missing feature columns: {missing[:10]}")

    prototypes: dict[str, Any] = {}
    all_means: list[list[float]] = []
    sample_counts: list[int] = []
    low_sample_categories: list[dict[str, Any]] = []
    for category_id, group in df.groupby("category_id", dropna=True):
        category_id_int = int(category_id)
        features = group[cols].to_numpy(dtype=np.float64)
        features = np.where(np.isfinite(features), features, 0.0)
        mean = clean_vector(features.mean(axis=0))
        std = clean_vector(features.std(axis=0))
        median = clean_vector(np.median(features, axis=0))
        count = int(len(group))
        source_rows = sorted({int(v) for v in group["source_row_index"].tolist()})
        category_name = str(group["category_name"].iloc[0])
        prototypes[str(category_id_int)] = {
            "category_id": category_id_int,
            "category_id_for_eval": category_id_int,
            "category_name": category_name,
            "sample_count": count,
            "feature_names": names,
            "mean_feature": mean,
            "std_feature": std,
            "median_feature": median,
            "feature_dim": len(names),
            "source_split": "support/prototype",
            "source_rows": source_rows,
            "leakage_check": "no_full_test_bbox_used",
            "channel_set": channels,
            "low_sample_count": count < 3,
            "notes": "Built from raxo/prototypes parquet-derived objects.xray_bbox and objects.categories.",
        }
        if count < 3:
            low_sample_categories.append({"category_id": category_id_int, "category_name": category_name, "sample_count": count})
        all_means.append(mean)
        sample_counts.append(count)

    all_object_features = df[cols].to_numpy(dtype=np.float64)
    all_object_features = np.where(np.isfinite(all_object_features), all_object_features, 0.0)
    global_prior = {
        "sample_count": int(len(df)),
        "mean_feature": clean_vector(all_object_features.mean(axis=0)) if len(df) else [],
        "std_feature": clean_vector(all_object_features.std(axis=0)) if len(df) else [],
        "is_global_prior": True,
        "use_as_normal_category_bank": False,
    }
    bank_stats = stats_for_features(np.asarray(all_means, dtype=np.float64)) if all_means else {"mean": [], "std": []}
    bank = {
        "version": "rmdx_multichannel_material_bank_v1",
        "bank_name": f"det_compass_{bank_type}_support",
        "mode": "bbox_only",
        "dataset": "DET-COMPASS",
        "feature_names": names,
        "feature_dim": len(names),
        "prototype_count": len(prototypes),
        "prototypes": prototypes,
        "bank_feature_stats": bank_stats,
        "global_prior": global_prior,
        "source_split": "support/prototype",
        "source_note": "Built only from raxo/prototypes support split metadata and multichannel images; no full_test bbox used.",
        "leakage_check": "no_full_test_bbox_used",
        "channel_set": channels,
        "low_sample_count_threshold": 3,
        "low_sample_categories": low_sample_categories,
        "schema_compatibility_note": "Top-level prototypes/feature_names/bank_feature_stats are compatible with existing bank loaders; non-colour banks require Batch28 multi-channel fusion support because feature_dim differs from legacy colour-only 22D.",
    }
    hist = Counter(sample_counts)
    summary = {
        "bank_type": bank_type,
        "path": str(BANK_DIR / bank_filename(bank_type)),
        "category_count": len(prototypes),
        "feature_dim": len(names),
        "channel_set": channels,
        "sample_count_min": min(sample_counts) if sample_counts else None,
        "sample_count_median": float(np.median(sample_counts)) if sample_counts else None,
        "sample_count_max": max(sample_counts) if sample_counts else None,
        "sample_count_histogram": dict(sorted(hist.items())),
        "low_sample_category_count": len(low_sample_categories),
        "leakage_check": "no_full_test_bbox_used",
    }
    return bank, summary


def main() -> int:
    BANK_DIR.mkdir(parents=True, exist_ok=True)
    if not FEATURE_PARQUET.exists():
        raise FileNotFoundError(FEATURE_PARQUET)
    df = pd.read_parquet(FEATURE_PARQUET)
    df = df[df["category_id"].notna() & (df["category_id"] != "")]
    summaries = []
    for bank_type in ["colour", "grey", "high", "low", "density", "multi_channel_all"]:
        bank, summary = build_bank(df, bank_type)
        out_path = BANK_DIR / bank_filename(bank_type)
        write_json(out_path, bank)
        summaries.append(summary)
    write_json(BANK_DIR / "material_bank_build_summary.json", summaries)
    lines = [
        "# DET-COMPASS 多通道 material bank 构建摘要",
        "",
        "| bank_type | category_count | feature_dim | sample_min | sample_median | sample_max | low_sample_categories | leakage_check |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['bank_type']} | {row['category_count']} | {row['feature_dim']} | {row['sample_count_min']} | "
            f"{row['sample_count_median']} | {row['sample_count_max']} | {row['low_sample_category_count']} | {row['leakage_check']} |"
        )
    (BANK_DIR / "material_bank_build_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"banks": len(summaries), "bank_dir": str(BANK_DIR)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
