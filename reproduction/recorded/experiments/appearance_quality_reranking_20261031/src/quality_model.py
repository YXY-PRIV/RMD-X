"""Frozen quality-surrogate pilot. No evaluation labels or AP optimization here."""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch import nn

PROTOCOL_SHA = "1efa1c18c607496a786607471efbe8784382b1f9efcb7839336eccb1ddca8643"
HEADS = ["visual_quality", "fixed_fusion_quality"]


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: str | Path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            result.update(block)
    return result.hexdigest()


def write(path: str | Path, value: Any) -> None:
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def save_model(path: Path, value: dict) -> None:
    temp = path.with_name(path.name + ".tmp")
    torch.save(value, temp)
    temp.replace(path)


class QualityMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def configure_cuda(device: str) -> torch.device:
    target = torch.device(device)
    if target.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Actual CUDA is required; CPU fallback is disabled")
    torch.cuda.set_device(target)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    return target


def read_protocol(path: Path) -> dict:
    if sha(path) != PROTOCOL_SHA:
        raise ValueError("Frozen quality protocol SHA mismatch")
    protocol = json.loads(path.read_text())
    if os.environ.get("CUDA_VISIBLE_DEVICES") != protocol["resources"]["GPU_training"]:
        raise ValueError("Physical GPU1 UUID must be the sole CUDA_VISIBLE_DEVICES entry")
    return protocol


def read_data(path: Path, protocol: dict) -> tuple[dict, dict]:
    arrays = {}
    with np.load(path, allow_pickle=False) as z:
        needed = {f"{prefix}_{split}" for prefix in ["X", "y", "mask", "image_ids"] for split in ["train", "val"]}
        needed |= {"feature_names", "metadata_json"}
        if set(z.files) != needed:
            raise ValueError(f"Training NPZ must contain only fit train/val fields: {sorted(set(z.files) ^ needed)}")
        names = z["feature_names"].astype(str).tolist()
        if names != protocol["feature_schema"]["names"] or len(names) != 16:
            raise ValueError("Training feature names/order differ from frozen16 schema")
        meta = json.loads(str(z["metadata_json"].item()))
        split_ref = protocol["training"]["image_split"]
        if sha(split_ref["path"]) != split_ref["sha256"]:
            raise ValueError("Frozen image split source changed")
        split_record = json.loads(Path(split_ref["path"]).read_text())
        for split, expected_rows in [("train", 5874), ("val", 1408)]:
            x = np.asarray(z[f"X_{split}"])
            y = np.asarray(z[f"y_{split}"])
            mask = np.asarray(z[f"mask_{split}"])
            ids = z[f"image_ids_{split}"]
            if x.dtype != np.float32 or x.shape != (expected_rows, 16) or not np.isfinite(x).all():
                raise ValueError(f"Malformed {split} input features")
            if y.dtype != np.float32 or y.shape != (expected_rows, 2) or not np.isfinite(y).all() or np.any((y < 0) | (y > 1)):
                raise ValueError(f"Malformed {split} soft targets")
            if mask.dtype != np.float32 or mask.shape != y.shape or not np.isin(mask, [0, 1]).all():
                raise ValueError(f"Malformed {split} retained mask")
            if np.any(y[mask == 0] != 0) or np.any(mask.sum(axis=0) == 0):
                raise ValueError("Inactive targets must be zero; both heads need retained rows")
            if ids.shape != (expected_rows,) or set(ids.astype(int)) != set(split_record[f"{split}_original_image_ids"]):
                raise ValueError(f"{split} image identities differ from frozen full-image split")
            arrays.update({f"X_{split}": x, f"y_{split}": y, f"mask_{split}": mask, f"image_ids_{split}": ids})
        if set(arrays["image_ids_train"]) & set(arrays["image_ids_val"]):
            raise ValueError("Train and early-stop images overlap")
    return arrays, {"feature_names": names, "producer_metadata": meta,
                    "train_images": len(set(arrays["image_ids_train"])),
                    "val_images": len(set(arrays["image_ids_val"])),
                    "rows": {s: len(arrays[f"X_{s}"]) for s in ["train", "val"]},
                    "retained_per_head": {s: arrays[f"mask_{s}"].sum(axis=0).astype(int).tolist() for s in ["train", "val"]}}


def masked_bce(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    element = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return ((element * mask).sum(dim=0) / mask.sum(dim=0).clamp_min(1)).mean()


def standardize(raw: np.ndarray, stats: dict, active: list[int]) -> np.ndarray:
    result = ((np.asarray(raw, dtype=np.float64) - np.asarray(stats["mean"])) / np.asarray(stats["std"])).astype(np.float32)
    inactive = sorted(set(range(16)) - set(active))
    result[:, inactive] = 0.0
    if not np.isfinite(result).all():
        raise ValueError("Non-finite standardized feature")
    return result


@torch.no_grad()
def predict(model: nn.Module, x: torch.Tensor, batch_size: int = 512) -> np.ndarray:
    model.eval()
    if len(x) == 0:
        return np.empty((0, 2), np.float32)
    return torch.cat([model(batch).sigmoid() for batch in x.split(batch_size)]).cpu().numpy().astype(np.float32)


def quality_metrics(q: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict:
    result = {}
    for h, name in enumerate(HEADS):
        selected = mask[:, h] == 1
        pred, truth = q[selected, h].astype(np.float64), target[selected, h].astype(np.float64)
        safe = np.clip(pred, 1e-7, 1 - 1e-7)
        result[name] = {"masked_rows": int(selected.sum()), "target_mean": float(truth.mean()),
                        "predicted_mean": float(pred.mean()), "soft_bce": float(-(truth*np.log(safe)+(1-truth)*np.log(1-safe)).mean()),
                        "mse": float(np.square(pred-truth).mean()), "mae": float(np.abs(pred-truth).mean())}
    return result


def train(args: Any) -> dict:
    protocol_path, data_path, output = Path(args.protocol).resolve(), Path(args.input).resolve(), Path(args.output_dir).resolve()
    protocol = read_protocol(protocol_path)
    training = protocol["training"]
    if args.variant not in training["feature_variants"] or args.seed not in training["seeds"]:
        raise ValueError("Variant or seed outside frozen protocol")
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f"New empty output or explicit resume required: {output}")
    output.mkdir(parents=True, exist_ok=True)
    raw, data_info = read_data(data_path, protocol)
    device = configure_cuda(args.device)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    union = raw["mask_train"].any(axis=1)
    train_fit = raw["X_train"][union].astype(np.float64)
    mean, std = train_fit.mean(axis=0), train_fit.std(axis=0, ddof=0)
    zero = std == 0
    std[zero] = 1.0
    stats = {"fit_role": "train union of native-retained heads only", "rows": int(union.sum()), "ddof": 0,
             "mean": mean.tolist(), "std": std.tolist(), "zero_std_coordinates": np.flatnonzero(zero).tolist()}
    active = training["feature_variants"][args.variant]
    x_full = {s: torch.from_numpy(standardize(raw[f"X_{s}"], stats, active)).to(device) for s in ["train", "val"]}
    target = {s: torch.from_numpy(raw[f"y_{s}"]).to(device) for s in ["train", "val"]}
    masks = {s: torch.from_numpy(raw[f"mask_{s}"]).to(device) for s in ["train", "val"]}
    # Only optimization drops neither-head-retained rows; saved output retains original row order.
    use = torch.from_numpy(union).to(device)
    x_train, y_train, mask_train = x_full["train"][use], target["train"][use], masks["train"][use]
    model = QualityMLP().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["lr"], weight_decay=training["weight_decay"])
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    config = {"protocol": {"path": str(protocol_path), "sha256": sha(protocol_path)}, "input_sha256": sha(data_path),
              "variant": args.variant, "seed": args.seed, "feature_names": data_info["feature_names"],
              "active_features": active, "inactive_standardized_columns_exact_zero": sorted(set(range(16))-set(active)),
              "standardization": stats, "training": training, "architecture": [16, 32, 16, 2],
              "heads": HEADS, "loss": "mean of two native-kept-masked unweighted soft-target BCE; no pos_weight"}
    best_state, best_loss, best_epoch, stale, start_epoch, elapsed_prior = None, float("inf"), -1, 0, 0, 0.0
    history = []
    if args.resume:
        old = torch.load(args.resume, map_location=device, weights_only=True)
        if old["config"] != config:
            raise ValueError("Resume input/config changed")
        model.load_state_dict(old["state_dict"]); optimizer.load_state_dict(old["optimizer_state"])
        generator.set_state(old["shuffle_state"].cpu())
        best_state, best_loss, best_epoch, stale = old["best_state"], old["best_loss"], old["best_epoch"], old["stale"]
        start_epoch, history, elapsed_prior = old["epoch"] + 1, old["history"], old["elapsed_seconds"]
        if best_state is not None and not (output / "model.pt").exists():
            save_model(output / "model.pt", {"format": "appearance_quality_v1", "state_dict": best_state,
                                             "config": config, "best_epoch": best_epoch, "best_val_bce": best_loss})
    environment = {"PID": os.getpid(), "PPID": os.getppid(), "argv": sys.argv, "started_utc": utc(),
                   "device": str(device), "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                   "GPU_name": torch.cuda.get_device_name(device), "torch_version": str(torch.__version__),
                   "torch_cuda_version": torch.version.cuda, "deterministic_algorithms": True, "TF32_enabled": False}
    write(output / "run_config.json", {"config": config, "data": data_info, "environment": environment})
    write(output / "feature_standardization.json", stats)
    write(output / "run_status.json", {"status": "RUNNING_CUDA", "variant": args.variant, "seed": args.seed, "PID": os.getpid()})
    print(json.dumps({"status": "ACTUAL_CUDA_TRAINING", "variant": args.variant, "seed": args.seed,
                      "PID": os.getpid(), "GPU": environment["GPU_name"], "GPU_UUID": environment["CUDA_VISIBLE_DEVICES"],
                      "train_original_rows": len(raw["X_train"]), "train_union_rows": len(x_train),
                      "train_head_masks": data_info["retained_per_head"]["train"], "val_head_masks": data_info["retained_per_head"]["val"]}), flush=True)
    torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device)
    started = time.perf_counter()
    for epoch in range(start_epoch, training["max_epochs"]):
        if stale >= training["early_stop_patience"]:
            break
        model.train()
        order = torch.randperm(len(x_train), generator=generator).to(device)
        for indices in order.split(training["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            loss = masked_bce(model(x_train[indices]), y_train[indices], mask_train[indices])
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            train_loss = float(masked_bce(model(x_train), y_train, mask_train).item())
            val_loss = float(masked_bce(model(x_full["val"]), target["val"], masks["val"]).item())
        if not np.isfinite(val_loss):
            raise FloatingPointError("Non-finite masked validation loss")
        if val_loss < best_loss - training["early_stop_min_delta"]:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            save_model(output / "model.pt", {"format": "appearance_quality_v1", "state_dict": best_state,
                                             "config": config, "best_epoch": epoch, "best_val_bce": best_loss})
        else:
            stale += 1
        torch.cuda.synchronize(device)
        elapsed = elapsed_prior + time.perf_counter() - started
        history.append({"epoch": epoch, "train_masked_soft_bce": train_loss, "val_masked_soft_bce": val_loss,
                        "best_epoch": best_epoch, "stale_epochs": stale, "elapsed_seconds": elapsed})
        write(output / "train_history.json", history)
        save_model(output / "last.pt", {"config": config, "epoch": epoch, "state_dict": model.state_dict(),
                                        "optimizer_state": optimizer.state_dict(), "best_state": best_state,
                                        "best_loss": best_loss, "best_epoch": best_epoch, "stale": stale,
                                        "history": history, "shuffle_state": generator.get_state(), "elapsed_seconds": elapsed})
        print(json.dumps(history[-1]), flush=True)
    if best_state is None:
        raise RuntimeError("No valid quality checkpoint")
    model.load_state_dict(best_state)
    metrics = {}
    for split in ["train", "val"]:
        quality = predict(model, x_full[split])
        np.savez_compressed(output / f"{split}_quality.npz", quality=quality,
                            q_visual=quality[:, 0], q_fused=quality[:, 1])
        metrics[split] = quality_metrics(quality, raw[f"y_{split}"], raw[f"mask_{split}"])
    torch.cuda.synchronize(device)
    result = {"status": "COMPLETE", "variant": args.variant, "seed": args.seed, "best_epoch": best_epoch,
              "epochs_completed": len(history), "best_val_masked_soft_bce": best_loss,
              "early_stopped": stale >= training["early_stop_patience"],
              "elapsed_seconds": elapsed_prior + time.perf_counter() - started,
              "timing_scope": "train loop plus validation/checkpoint/export wall; not pure GPU kernel timing",
              "peak_CUDA_allocated_bytes": torch.cuda.max_memory_allocated(device),
              "peak_CUDA_reserved_bytes": torch.cuda.max_memory_reserved(device),
              "checkpoint_sha256": sha(output / "model.pt"), "environment": environment, "metrics": metrics,
              "AP_computed": False, "alpha_selected": False, "evaluation_labels_used": False,
              "quality_interpretation": "IoU-times-original-binary-correct surrogate; not COCO TP or certified probability calibration"}
    write(output / "metrics.json", result)
    write(output / "run_status.json", {key: result[key] for key in ["status", "variant", "seed", "best_epoch", "epochs_completed", "checkpoint_sha256", "AP_computed", "alpha_selected", "evaluation_labels_used"]})
    return result


def predict_checkpoint(checkpoint_path: Path, raw: np.ndarray, feature_names: list[str], device: str, batch_size: int = 512) -> tuple[np.ndarray, dict]:
    cuda = configure_cuda(device)
    checkpoint = torch.load(checkpoint_path, map_location=cuda, weights_only=True)
    if checkpoint.get("format") != "appearance_quality_v1":
        raise ValueError("Not a quality checkpoint")
    config = checkpoint["config"]
    if feature_names != config["feature_names"] or np.asarray(raw).ndim != 2 or raw.shape[1] != 16:
        raise ValueError("Input feature order/shape differs from frozen schema")
    model = QualityMLP().to(cuda); model.load_state_dict(checkpoint["state_dict"])
    standardized = standardize(raw, config["standardization"], config["active_features"])
    return predict(model, torch.from_numpy(standardized).to(cuda), batch_size), config
