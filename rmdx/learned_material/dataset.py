from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURE_PARQUET = REPO_ROOT / "outputs_rmdx/material_multichannel_det_compass/features/support_multichannel_object_features.parquet"
DEFAULT_FEATURE_CSV = REPO_ROOT / "outputs_rmdx/material_multichannel_det_compass/features/support_multichannel_object_features.csv"
DEFAULT_BANK = REPO_ROOT / "outputs_rmdx/material_multichannel_det_compass/banks/material_bank_multichannel_all_support.json"


@dataclass
class FeatureBundle:
    df: pd.DataFrame
    feature_names: list[str]


class MaterialObjectDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, object_ids: Iterable[str]):
        self.x = torch.as_tensor(np.array(x, copy=True), dtype=torch.float32)
        self.y = torch.as_tensor(np.array(y, copy=True), dtype=torch.long)
        self.object_ids = list(object_ids)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int):
        return self.x[index], self.y[index], self.object_ids[index]


def read_json(path: str | Path):
    with Path(path).open() as f:
        return json.load(f)


def write_json(path: str | Path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_bank_feature_names(bank_path: str | Path = DEFAULT_BANK) -> list[str]:
    bank = read_json(bank_path)
    feature_names = bank.get("feature_names")
    if not feature_names:
        raise ValueError(f"feature_names missing in bank: {bank_path}")
    return list(feature_names)


def load_feature_table(
    feature_path: str | Path | None = None,
    bank_path: str | Path = DEFAULT_BANK,
) -> FeatureBundle:
    if feature_path is None:
        feature_path = DEFAULT_FEATURE_PARQUET if DEFAULT_FEATURE_PARQUET.exists() else DEFAULT_FEATURE_CSV
    feature_path = Path(feature_path)
    if feature_path.suffix == ".parquet":
        df = pd.read_parquet(feature_path)
    else:
        df = pd.read_csv(feature_path)

    if "source_is_full_test" in df.columns and df["source_is_full_test"].astype(bool).any():
        raise ValueError("full_test rows detected in support/prototype feature table")
    if "source_split" in df.columns:
        bad = ~df["source_split"].astype(str).str.contains("support|prototype", case=False, regex=True)
        if bad.any():
            raise ValueError("non support/prototype rows detected in feature table")

    feature_names = load_bank_feature_names(bank_path)
    missing = [name for name in feature_names if name not in df.columns]
    if missing:
        raise ValueError(f"feature columns missing: {missing[:10]} (total {len(missing)})")

    df = df.copy()
    if "object_id" not in df.columns:
        df["object_id"] = df.apply(
            lambda r: f"row{int(r.get('source_row_index', r.name))}_obj{int(r.get('object_index', 0))}",
            axis=1,
        )
    df["category_id"] = df["category_id"].astype(int)
    df["category_name"] = df["category_name"].astype(str)
    df[feature_names] = df[feature_names].replace([np.inf, -np.inf], np.nan)
    for name in feature_names:
        if df[name].isna().any():
            median = df[name].median()
            if np.isnan(median):
                median = 0.0
            df[name] = df[name].fillna(float(median))
    return FeatureBundle(df=df, feature_names=feature_names)


def build_category_maps(category_ids: Iterable[int]) -> tuple[dict[int, int], dict[int, int]]:
    ids = sorted({int(v) for v in category_ids})
    cat_to_index = {cat_id: i for i, cat_id in enumerate(ids)}
    index_to_cat = {i: cat_id for cat_id, i in cat_to_index.items()}
    return cat_to_index, index_to_cat


def standardize_train_val(
    train_x: np.ndarray,
    val_x: np.ndarray,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return (train_x - mean) / std, (val_x - mean) / std, mean.squeeze(0), std.squeeze(0)


def dataframe_to_arrays(df: pd.DataFrame, feature_names: list[str], cat_to_index: dict[int, int]):
    x = df[feature_names].to_numpy(dtype=np.float32)
    y = df["category_id"].map(cat_to_index).to_numpy(dtype=np.int64)
    object_ids = df["object_id"].astype(str).tolist()
    return x, y, object_ids
