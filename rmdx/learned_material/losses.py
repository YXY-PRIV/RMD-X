from __future__ import annotations

import torch
from torch.nn import functional as F


def compute_class_prototypes(embeddings: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Compute L2-normalized class prototypes from object embeddings."""
    dim = embeddings.shape[1]
    prototypes = embeddings.new_zeros((num_classes, dim))
    counts = embeddings.new_zeros((num_classes, 1))
    prototypes.index_add_(0, labels, embeddings)
    counts.index_add_(0, labels, torch.ones((labels.shape[0], 1), device=embeddings.device, dtype=embeddings.dtype))
    prototypes = prototypes / counts.clamp_min(1.0)
    return F.normalize(prototypes, p=2, dim=-1)


def sample_negative_indices(labels: torch.Tensor, num_classes: int, negatives_per_sample: int) -> torch.Tensor:
    """Sample negative class indices that differ from each sample label."""
    device = labels.device
    neg = torch.randint(0, num_classes - 1, (labels.shape[0], negatives_per_sample), device=device)
    return neg + (neg >= labels[:, None]).long()


def pairwise_ranking_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    margin: float = 0.1,
    negatives_per_sample: int = 16,
) -> torch.Tensor:
    """Margin ranking loss: score(x, pos) should exceed sampled negatives."""
    pos_scores = (embeddings * prototypes[labels]).sum(dim=-1, keepdim=True)
    neg_idx = sample_negative_indices(labels, prototypes.shape[0], negatives_per_sample)
    neg_scores = torch.einsum("bd,bnd->bn", embeddings, prototypes[neg_idx])
    loss = torch.clamp(float(margin) - pos_scores + neg_scores, min=0.0)
    return loss.mean()

