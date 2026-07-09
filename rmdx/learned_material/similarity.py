from __future__ import annotations

import torch
from torch.nn import functional as F


@torch.no_grad()
def cosine_class_similarity(
    proposal_embeddings: torch.Tensor,
    category_prototypes: torch.Tensor,
    *,
    chunk_size: int = 8192,
) -> torch.Tensor:
    """Compute proposal x category cosine similarity in chunks."""
    proposal_embeddings = F.normalize(proposal_embeddings.float(), p=2, dim=-1)
    category_prototypes = F.normalize(category_prototypes.float(), p=2, dim=-1)
    rows = []
    for start in range(0, proposal_embeddings.shape[0], chunk_size):
        rows.append(proposal_embeddings[start : start + chunk_size] @ category_prototypes.T)
    return torch.cat(rows, dim=0).float()

