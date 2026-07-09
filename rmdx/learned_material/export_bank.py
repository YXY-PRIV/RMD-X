from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from rmdx.learned_material.dataset import REPO_ROOT, load_feature_table, standardize_train_val, write_json
from rmdx.learned_material.model import build_model


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)


@torch.no_grad()
def export_learned_bank(checkpoint_path: str | Path, split_path: str | Path, output_path: str | Path, metadata_path: str | Path) -> dict:
    ckpt = load_checkpoint(checkpoint_path)
    cfg = ckpt["config"]
    bundle = load_feature_table(feature_path=cfg["feature_path"], bank_path=cfg["bank_path"])
    split_df = pd.read_csv(split_path)
    df = bundle.df.merge(split_df[["object_id", "split", "lowshot_flag"]], on="object_id", how="inner")
    train_df = df[df["split"].astype(str).str.startswith("train")].copy()
    val_counts = df[df["split"].astype(str).eq("val")].groupby("category_id").size().to_dict()
    train_x = train_df[bundle.feature_names].to_numpy(dtype="float32")
    _, train_x_std, mean, std = standardize_train_val(train_x, train_x)
    model = build_model(cfg["config_name"], input_dim=len(bundle.feature_names))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    z = model(torch.as_tensor(train_x_std.copy(), dtype=torch.float32)).cpu()

    train_df = train_df.reset_index(drop=True)
    prototypes = {}
    for category_id, group in train_df.groupby("category_id", sort=True):
        idx = group.index.to_numpy().copy().tolist()
        vec = torch.nn.functional.normalize(z[idx].mean(dim=0, keepdim=True), p=2, dim=-1).squeeze(0).numpy()
        sample_count = int((df["category_id"] == category_id).sum())
        prototypes[str(int(category_id))] = {
            "category_id": int(category_id),
            "category_name": str(group["category_name"].iloc[0]),
            "learned_prototype_vector": vec.astype(float).tolist(),
            "embedding_dim": int(vec.shape[0]),
            "sample_count": sample_count,
            "train_sample_count": int(len(group)),
            "val_sample_count": int(val_counts.get(category_id, 0)),
            "lowshot_flag": bool(sample_count < 3),
            "source_split": "support/prototype",
            "model_config": cfg["config_name"],
            "checkpoint_path": str(checkpoint_path),
            "leakage_check": "no_full_test_bbox_used",
        }
    bank = {
        "version": "batch31_learned_material_adapter_v1",
        "bank_name": Path(output_path).stem,
        "source_split": "support/prototype",
        "leakage_check": "no_full_test_bbox_used",
        "model_config": cfg,
        "feature_names": bundle.feature_names,
        "input_feature_mean": mean.astype(float).tolist(),
        "input_feature_std": std.astype(float).tolist(),
        "embedding_dim": int(next(iter(prototypes.values()))["embedding_dim"]) if prototypes else 0,
        "prototype_count": len(prototypes),
        "prototypes": prototypes,
    }
    write_json(output_path, bank)
    metadata = {
        "checkpoint_path": str(checkpoint_path),
        "output_path": str(output_path),
        "prototype_count": len(prototypes),
        "lowshot_category_count": sum(1 for v in prototypes.values() if v["lowshot_flag"]),
        "source_split": "support/prototype",
        "leakage_check": "no_full_test_bbox_used",
    }
    write_json(metadata_path, metadata)
    return metadata
