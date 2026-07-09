"""JCL-RMD-X reranker policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class PrototypePolicy:
    category_ids: list[int]
    prototypes: np.ndarray
    mean: np.ndarray
    std: np.ndarray


def standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / np.where(std < 1e-6, 1.0, std)


def build_prototype_policy(features: np.ndarray, labels: np.ndarray) -> PrototypePolicy:
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std < 1e-6] = 1.0
    z = standardize(features, mean, std)
    category_ids = sorted({int(v) for v in labels.tolist()})
    protos = []
    for cid in category_ids:
        proto = z[labels == cid].mean(axis=0)
        norm = np.linalg.norm(proto)
        protos.append(proto / max(norm, 1e-12))
    return PrototypePolicy(category_ids, np.asarray(protos, dtype=np.float32), mean.astype(np.float32), std.astype(np.float32))


def prototype_scores(policy: dict[str, Any], features: np.ndarray) -> dict[int, np.ndarray]:
    mean = np.asarray(policy["mean"], dtype=np.float32)
    std = np.asarray(policy["std"], dtype=np.float32)
    protos = np.asarray(policy["prototypes"], dtype=np.float32)
    z = standardize(features.astype(np.float32), mean, std)
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    z = z / np.maximum(norms, 1e-12)
    scores = z @ protos.T
    return {int(cid): scores[:, idx] for idx, cid in enumerate(policy["category_ids"])}


class LinearHead(torch.nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def train_linear_head(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    *,
    epochs: int = 40,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
    device: str = "cpu",
) -> tuple[dict[str, Any], list[dict[str, float]]]:
    category_ids = sorted({int(v) for v in train_y.tolist()})
    id_to_idx = {cid: idx for idx, cid in enumerate(category_ids)}
    y = torch.tensor([id_to_idx[int(v)] for v in train_y], dtype=torch.long, device=device)
    val_mask = np.asarray([int(v) in id_to_idx for v in val_y], dtype=bool)
    vy = torch.tensor([id_to_idx[int(v)] for v in val_y[val_mask]], dtype=torch.long, device=device)
    mean = train_x.mean(axis=0).astype(np.float32)
    std = train_x.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    x = torch.tensor(standardize(train_x.astype(np.float32), mean, std), dtype=torch.float32, device=device)
    vx = torch.tensor(standardize(val_x[val_mask].astype(np.float32), mean, std), dtype=torch.float32, device=device)
    counts = np.bincount([id_to_idx[int(v)] for v in train_y], minlength=len(category_ids)).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    model = LinearHead(train_x.shape[1], len(category_ids)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    best_state: dict[str, torch.Tensor] | None = None
    best_acc = -1.0
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            if vx.numel() and vy.numel():
                val_logits = model(vx)
                pred = val_logits.argmax(dim=1)
                acc = float((pred == vy).float().mean().item())
                topk = min(5, val_logits.shape[1])
                top5 = float((val_logits.topk(topk, dim=1).indices == vy.view(-1, 1)).any(dim=1).float().mean().item())
            else:
                acc = 0.0
                top5 = 0.0
        history.append({"epoch": float(epoch), "loss": float(loss.item()), "val_acc": acc, "val_top5": top5})
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    state = {
        "category_ids": category_ids,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "input_dim": int(train_x.shape[1]),
        "num_classes": len(category_ids),
    }
    return state, history


def linear_scores(policy: dict[str, Any], features: np.ndarray) -> dict[int, np.ndarray]:
    mean = np.asarray(policy["mean"], dtype=np.float32)
    std = np.asarray(policy["std"], dtype=np.float32)
    x = torch.tensor(standardize(features.astype(np.float32), mean, std), dtype=torch.float32)
    model = LinearHead(int(policy["input_dim"]), int(policy["num_classes"]))
    model.load_state_dict(policy["state_dict"])
    model.eval()
    with torch.no_grad():
        logits = model(x).float().numpy()
    return {int(cid): logits[:, idx] for idx, cid in enumerate(policy["category_ids"])}
