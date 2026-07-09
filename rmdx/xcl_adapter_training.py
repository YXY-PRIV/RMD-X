"""Training helpers for Batch52 XCL-Adapter."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from rmdx.xcl_adapter_model import make_adapter


@dataclass
class TrainResult:
    policy_id: str
    policy_type: str
    best_epoch: int
    train_top1: float
    val_top1: float
    val_top5: float
    val_macro_top1: float
    risk: str
    state_dict: dict


def topk_metrics(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> tuple[float, float, float]:
    pred1 = torch.argmax(logits, dim=1)
    top1 = (pred1 == labels).float().mean().item() if labels.numel() else 0.0
    top5 = (torch.topk(logits, k=min(5, logits.shape[1]), dim=1).indices == labels.view(-1, 1)).any(dim=1).float().mean().item() if labels.numel() else 0.0
    per = []
    for c in range(num_classes):
        mask = labels == c
        if mask.any():
            per.append((pred1[mask] == labels[mask]).float().mean().item())
    macro = float(sum(per) / len(per)) if per else 0.0
    return top1, top5, macro


def evaluate(model: nn.Module, images: torch.Tensor, labels: torch.Tensor, text: torch.Tensor, batch_size: int = 2048) -> tuple[float, float, float]:
    model.eval()
    outs = []
    with torch.no_grad():
        t = model.forward_text(text)
        for i in range(0, images.shape[0], batch_size):
            im = model.forward_image(images[i : i + batch_size])
            outs.append(im @ t.T * 20.0)
    logits = torch.cat(outs, dim=0)
    return topk_metrics(logits.cpu(), labels.cpu(), text.shape[0])


def train_adapter(
    policy_id: str,
    policy_type: str,
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    val_images: torch.Tensor,
    val_labels: torch.Tensor,
    text_embeddings: torch.Tensor,
    *,
    epochs: int = 35,
    lr: float = 1e-3,
    batch_size: int = 512,
    device: str = "cuda:0",
) -> TrainResult:
    device = device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
    model = make_adapter(policy_type, dim=train_images.shape[1]).to(device)
    text = text_embeddings.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    ds = TensorDataset(train_images.float(), train_labels.long())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    best = None
    best_score = -1.0
    best_epoch = 0
    patience = 8
    since = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for image, label in loader:
            image = image.to(device)
            label = label.to(device)
            t = model.forward_text(text)
            logits = model.forward_image(image) @ t.T * 20.0
            loss = loss_fn(logits, label)
            opt.zero_grad()
            loss.backward()
            opt.step()
        val_top1, val_top5, val_macro = evaluate(model, val_images.to(device), val_labels.to(device), text, batch_size=batch_size)
        score = val_top1 + 0.2 * val_macro
        if score > best_score:
            best_score = score
            best = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_epoch = epoch
            since = 0
        else:
            since += 1
        if since >= patience:
            break
    if best is not None:
        model.load_state_dict(best)
    train_top1, _train_top5, _train_macro = evaluate(model, train_images.to(device), train_labels.to(device), text, batch_size=batch_size)
    val_top1, val_top5, val_macro = evaluate(model, val_images.to(device), val_labels.to(device), text, batch_size=batch_size)
    risk = "READY"
    if train_top1 - val_top1 > 0.25:
        risk = "READY_WITH_OVERFIT_RISK"
    if val_top1 < 0.02:
        risk = "LOW_VALIDATION_SIGNAL"
    return TrainResult(policy_id, policy_type, best_epoch, train_top1, val_top1, val_top5, val_macro, risk, best or model.state_dict())
