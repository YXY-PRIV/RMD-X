from __future__ import annotations

import numpy as np
import torch

from rmdx.learned_material.losses import compute_class_prototypes


@torch.no_grad()
def evaluate_embeddings(
    model: torch.nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    device: str,
    negatives_per_sample: int = 32,
) -> dict:
    model.eval()
    train_x_t = torch.as_tensor(np.array(train_x, copy=True), dtype=torch.float32, device=device)
    train_y_t = torch.as_tensor(np.array(train_y, copy=True), dtype=torch.long, device=device)
    val_x_t = torch.as_tensor(np.array(val_x, copy=True), dtype=torch.float32, device=device)
    val_y_t = torch.as_tensor(np.array(val_y, copy=True), dtype=torch.long, device=device)

    train_z = model(train_x_t)
    val_z = model(val_x_t)
    num_classes = int(max(train_y.max(), val_y.max()) + 1)
    prototypes = compute_class_prototypes(train_z, train_y_t, num_classes)
    sims = val_z @ prototypes.T
    pos = sims[torch.arange(val_y_t.numel(), device=device), val_y_t]
    masked = sims.clone()
    masked[torch.arange(val_y_t.numel(), device=device), val_y_t] = -1e9
    max_neg = masked.max(dim=1).values
    ranks = (sims > pos[:, None]).sum(dim=1) + 1
    top1 = (ranks == 1).float().mean().item()
    top5 = (ranks <= 5).float().mean().item()
    mrr = (1.0 / ranks.float()).mean().item()
    margin = (pos - max_neg).mean().item()
    ranking_acc = (pos > max_neg).float().mean().item()
    emb_norm = val_z.norm(dim=1)

    proto_sims = prototypes @ prototypes.T
    eye = torch.eye(proto_sims.shape[0], dtype=torch.bool, device=device)
    inter = proto_sims[~eye].mean().item()

    return {
        "val_count": int(val_y_t.numel()),
        "val_top1": float(top1),
        "val_top5": float(top5),
        "val_mrr": float(mrr),
        "val_retrieval_map": float(mrr),
        "val_ranking_accuracy": float(ranking_acc),
        "val_positive_vs_max_negative_margin": float(margin),
        "embedding_norm_mean": float(emb_norm.mean().item()),
        "embedding_norm_std": float(emb_norm.std().item()),
        "prototype_inter_class_similarity_mean": float(inter),
    }


def per_category_accuracy(model, train_x, train_y, val_x, val_y, index_to_cat, device: str) -> list[dict]:
    with torch.no_grad():
        model.eval()
        train_x_t = torch.as_tensor(np.array(train_x, copy=True), dtype=torch.float32, device=device)
        train_y_t = torch.as_tensor(np.array(train_y, copy=True), dtype=torch.long, device=device)
        val_x_t = torch.as_tensor(np.array(val_x, copy=True), dtype=torch.float32, device=device)
        val_y_t = torch.as_tensor(np.array(val_y, copy=True), dtype=torch.long, device=device)
        train_z = model(train_x_t)
        val_z = model(val_x_t)
        prototypes = compute_class_prototypes(train_z, train_y_t, len(index_to_cat))
        pred = (val_z @ prototypes.T).argmax(dim=1).cpu().numpy()
    rows = []
    for cls in sorted(set(val_y.tolist())):
        mask = val_y == cls
        rows.append({
            "category_id": int(index_to_cat[int(cls)]),
            "val_count": int(mask.sum()),
            "top1_accuracy": float((pred[mask] == cls).mean()) if mask.sum() else 0.0,
        })
    return rows
