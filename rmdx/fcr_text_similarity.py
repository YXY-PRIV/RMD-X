"""Utilities for FCR crop-text embeddings and similarity."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_rows(tensor: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return tensor.float() / tensor.float().norm(dim=1, keepdim=True).clamp_min(eps)


def compute_crop_text_similarity(crop_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
    crops = normalize_rows(crop_embeddings)
    texts = normalize_rows(text_embeddings)
    sim = crops @ texts.T
    if torch.isnan(sim).any():
        raise RuntimeError("crop-text similarity contains NaN")
    if torch.isinf(sim).any():
        raise RuntimeError("crop-text similarity contains Inf")
    if sim.numel() > 1 and float(sim.std().item()) <= 1e-12:
        raise RuntimeError("crop-text similarity is constant")
    return sim.float()

