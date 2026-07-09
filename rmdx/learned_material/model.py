from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


MODEL_CONFIGS = {
    "mlp64": {"input_dim": 110, "hidden_dim": 128, "embedding_dim": 64, "dropout": 0.1},
    "mlp128": {"input_dim": 110, "hidden_dim": 256, "embedding_dim": 128, "dropout": 0.1},
    "colour22_mlp64": {"input_dim": 22, "hidden_dim": 64, "embedding_dim": 64, "dropout": 0.1},
}


class MaterialPrototypeAdapter(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int, dropout: float = 0.1):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.embedding_dim = int(embedding_dim)
        self.dropout = float(dropout)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), p=2, dim=-1)


def build_model(config_name: str, input_dim: int | None = None) -> MaterialPrototypeAdapter:
    if config_name not in MODEL_CONFIGS:
        raise ValueError(f"unknown config_name: {config_name}")
    cfg = dict(MODEL_CONFIGS[config_name])
    if input_dim is not None:
        cfg["input_dim"] = int(input_dim)
    return MaterialPrototypeAdapter(**cfg)


def model_config(config_name: str, input_dim: int | None = None) -> dict:
    cfg = dict(MODEL_CONFIGS[config_name])
    if input_dim is not None:
        cfg["input_dim"] = int(input_dim)
    cfg["config_name"] = config_name
    return cfg

