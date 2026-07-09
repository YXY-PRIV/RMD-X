"""Training helpers for Batch53 multi-dataset XCL adapter."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from rmdx.mxcl_adapter_model import make_adapter


@dataclass
class MXCLTrainResult:
    policy_id: str
    policy_type: str
    best_epoch: int
    train_top1: float
    val_top1: float
    val_top5: float
    val_macro_top1: float
    per_dataset_val_top1: dict[str, float]
    per_dataset_val_top5: dict[str, float]
    risk: str
    state_dict: dict


def topk_metrics(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> tuple[float, float, float]:
    if labels.numel() == 0:
        return 0.0, 0.0, 0.0
    pred1 = torch.argmax(logits, dim=1)
    top1 = (pred1 == labels).float().mean().item()
    top5 = (torch.topk(logits, k=min(5, logits.shape[1]), dim=1).indices == labels.view(-1, 1)).any(dim=1).float().mean().item()
    per = []
    for c in range(num_classes):
        mask = labels == c
        if mask.any():
            per.append((pred1[mask] == labels[mask]).float().mean().item())
    macro = float(sum(per) / len(per)) if per else 0.0
    return top1, top5, macro


def logits_for(model: nn.Module, images: torch.Tensor, text: torch.Tensor, batch_size: int = 2048) -> torch.Tensor:
    outs: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        adapted_text = model.forward_text(text)
        for i in range(0, images.shape[0], batch_size):
            outs.append(model.forward_image(images[i : i + batch_size]) @ adapted_text.T * 20.0)
    return torch.cat(outs, dim=0) if outs else torch.empty((0, text.shape[0]), device=text.device)


def evaluate(model: nn.Module, images: torch.Tensor, labels: torch.Tensor, text: torch.Tensor, batch_size: int = 2048) -> tuple[float, float, float]:
    return topk_metrics(logits_for(model, images, text, batch_size).cpu(), labels.cpu(), text.shape[0])


def per_dataset_metrics(model: nn.Module, images: torch.Tensor, labels: torch.Tensor, datasets: list[str], text: torch.Tensor, batch_size: int = 2048) -> tuple[dict[str, float], dict[str, float]]:
    logits = logits_for(model, images, text, batch_size).cpu()
    labels_cpu = labels.cpu()
    top1: dict[str, float] = {}
    top5: dict[str, float] = {}
    for dataset in sorted(set(datasets)):
        idx = torch.tensor([i for i, d in enumerate(datasets) if d == dataset], dtype=torch.long)
        if idx.numel() == 0:
            continue
        t1, t5, _ = topk_metrics(logits[idx], labels_cpu[idx], text.shape[0])
        top1[dataset] = t1
        top5[dataset] = t5
    return top1, top5


def train_adapter(
    policy_id: str,
    policy_type: str,
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    val_images: torch.Tensor,
    val_labels: torch.Tensor,
    val_datasets: list[str],
    text_embeddings: torch.Tensor,
    *,
    epochs: int = 40,
    lr: float = 1e-3,
    batch_size: int = 512,
    device: str = "cuda:0",
) -> MXCLTrainResult:
    device = device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
    model = make_adapter(policy_type, dim=train_images.shape[1]).to(device)
    text = text_embeddings.to(device).float()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(train_images.float(), train_labels.long()), batch_size=batch_size, shuffle=True, num_workers=0)
    best_state = None
    best_score = -1.0
    best_epoch = 0
    patience = 8
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for image, label in loader:
            image = image.to(device)
            label = label.to(device)
            adapted_text = model.forward_text(text)
            logits = model.forward_image(image) @ adapted_text.T * 20.0
            loss = loss_fn(logits, label)
            opt.zero_grad()
            loss.backward()
            opt.step()
        val_top1, val_top5, val_macro = evaluate(model, val_images.to(device), val_labels.to(device), text, batch_size)
        score = val_top1 + 0.2 * val_macro + 0.05 * val_top5
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    train_top1, _train_top5, _train_macro = evaluate(model, train_images.to(device), train_labels.to(device), text, batch_size)
    val_top1, val_top5, val_macro = evaluate(model, val_images.to(device), val_labels.to(device), text, batch_size)
    ds_top1, ds_top5 = per_dataset_metrics(model, val_images.to(device), val_labels.to(device), val_datasets, text, batch_size)
    risk = "READY"
    if train_top1 - val_top1 > 0.25:
        risk = "READY_WITH_OVERFIT_RISK"
    if val_top1 < 0.02:
        risk = "LOW_VALIDATION_SIGNAL"
    return MXCLTrainResult(policy_id, policy_type, best_epoch, train_top1, val_top1, val_top5, val_macro, ds_top1, ds_top5, risk, best_state or model.state_dict())
