from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from rmdx.learned_material.dataset import (
    REPO_ROOT,
    MaterialObjectDataset,
    build_category_maps,
    dataframe_to_arrays,
    load_feature_table,
    standardize_train_val,
    write_json,
)
from rmdx.learned_material.evaluate import evaluate_embeddings
from rmdx.learned_material.losses import compute_class_prototypes, pairwise_ranking_loss
from rmdx.learned_material.model import build_model, model_config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested


def prepare_arrays(split_path: Path, feature_path: Path, bank_path: Path):
    bundle = load_feature_table(feature_path=feature_path, bank_path=bank_path)
    split_df = pd.read_csv(split_path)
    df = bundle.df.merge(split_df[["object_id", "split", "lowshot_flag"]], on="object_id", how="inner")
    if df.empty:
        raise ValueError(f"no rows after joining split: {split_path}")
    cat_to_index, index_to_cat = build_category_maps(df["category_id"].tolist())
    train_df = df[df["split"].astype(str).str.startswith("train")].copy()
    val_df = df[df["split"].astype(str).eq("val")].copy()
    train_x, train_y, train_ids = dataframe_to_arrays(train_df, bundle.feature_names, cat_to_index)
    val_x, val_y, val_ids = dataframe_to_arrays(val_df, bundle.feature_names, cat_to_index)
    train_x, val_x, mean, std = standardize_train_val(train_x, val_x)
    return {
        "feature_names": bundle.feature_names,
        "df": df,
        "train_df": train_df,
        "val_df": val_df,
        "train_x": train_x,
        "train_y": train_y,
        "train_ids": train_ids,
        "val_x": val_x,
        "val_y": val_y,
        "val_ids": val_ids,
        "mean": mean,
        "std": std,
        "cat_to_index": cat_to_index,
        "index_to_cat": index_to_cat,
    }


def train_one(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    device = choose_device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = prepare_arrays(Path(args.split_path), Path(args.feature_path), Path(args.bank_path))

    model = build_model(args.config, input_dim=len(data["feature_names"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ds = MaterialObjectDataset(data["train_x"], data["train_y"], data["train_ids"])
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    num_classes = len(data["cat_to_index"])

    cfg = {
        **model_config(args.config, input_dim=len(data["feature_names"])),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "margin": args.margin,
        "negatives_per_sample": args.negatives_per_sample,
        "device": device,
        "feature_path": str(args.feature_path),
        "bank_path": str(args.bank_path),
        "split_path": str(args.split_path),
        "leakage_check": "no_full_test_bbox_used",
    }
    write_json(out_dir / "config.json", cfg)

    metrics_path = out_dir / "metrics.csv"
    best_metric = -1.0
    best_epoch = -1
    bad_epochs = 0
    rows = []
    started = time.time()
    train_x_t = torch.as_tensor(np.array(data["train_x"], copy=True), dtype=torch.float32, device=device)
    train_y_t = torch.as_tensor(np.array(data["train_y"], copy=True), dtype=torch.long, device=device)

    for epoch in range(1, args.epochs + 1):
        model.eval()
        with torch.no_grad():
            train_z = model(train_x_t)
            prototypes = compute_class_prototypes(train_z, train_y_t, num_classes).detach()
        model.train()
        losses = []
        for xb, yb, _ in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            z = model(xb)
            loss = pairwise_ranking_loss(
                z,
                yb,
                prototypes,
                margin=args.margin,
                negatives_per_sample=args.negatives_per_sample,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch}: {loss.item()}")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        val_metrics = evaluate_embeddings(
            model,
            data["train_x"],
            data["train_y"],
            data["val_x"],
            data["val_y"],
            device=device,
            negatives_per_sample=args.negatives_per_sample,
        )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            **val_metrics,
            "elapsed_sec": round(time.time() - started, 3),
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

        metric = row["val_retrieval_map"]
        checkpoint = {
            "model_state": model.state_dict(),
            "config": cfg,
            "feature_names": data["feature_names"],
            "feature_mean": data["mean"].tolist(),
            "feature_std": data["std"].tolist(),
            "cat_to_index": {str(k): int(v) for k, v in data["cat_to_index"].items()},
            "index_to_cat": {str(k): int(v) for k, v in data["index_to_cat"].items()},
            "epoch": epoch,
            "metrics": row,
        }
        torch.save(checkpoint, out_dir / "last.pt")
        if metric > best_metric:
            best_metric = metric
            best_epoch = epoch
            bad_epochs = 0
            torch.save(checkpoint, out_dir / "best.pt")
        else:
            bad_epochs += 1
        if epoch >= args.min_epochs and bad_epochs >= args.early_stopping_patience:
            break

    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "config_name": args.run_name,
        "model_config": args.config,
        "seed": args.seed,
        "status": "SUCCESS",
        "device": device,
        "best_epoch": best_epoch,
        "best_val_retrieval_map": best_metric,
        "train_object_count": int(len(data["train_y"])),
        "val_object_count": int(len(data["val_y"])),
        "category_count": int(num_classes),
        "checkpoint_dir": str(out_dir),
        "best_checkpoint": str(out_dir / "best.pt"),
        "last_checkpoint": str(out_dir / "last.pt"),
        "metrics_path": str(metrics_path),
        "elapsed_sec": round(time.time() - started, 3),
    }
    write_json(out_dir / "run_summary.json", summary)
    return summary


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    p.add_argument("--config", choices=["mlp64", "mlp128", "colour22_mlp64"], default="mlp64")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--min-epochs", type=int, default=2)
    p.add_argument("--early-stopping-patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--margin", type=float, default=0.1)
    p.add_argument("--negatives-per-sample", type=int, default=16)
    p.add_argument("--device", default="auto")
    p.add_argument("--feature-path", default=str(REPO_ROOT / "outputs_rmdx/material_multichannel_det_compass/features/support_multichannel_object_features.parquet"))
    p.add_argument("--bank-path", default=str(REPO_ROOT / "outputs_rmdx/material_multichannel_det_compass/banks/material_bank_multichannel_all_support.json"))
    p.add_argument("--split-path", default=str(REPO_ROOT / "outputs_rmdx/learned_material_encoder/splits/object_level_split_seed0.csv"))
    p.add_argument("--output-dir", required=True)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    print(json.dumps(train_one(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
