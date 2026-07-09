"""Lightweight XCL adapter models."""

from __future__ import annotations

import torch
from torch import nn


class LinearImageAdapter(nn.Module):
    def __init__(self, dim: int = 512) -> None:
        super().__init__()
        self.image = nn.Linear(dim, dim, bias=False)
        nn.init.eye_(self.image.weight)

    def forward_image(self, image: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(self.image(image.float()), dim=-1)

    def forward_text(self, text: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(text.float(), dim=-1)


class DualLinearAdapter(nn.Module):
    def __init__(self, dim: int = 512) -> None:
        super().__init__()
        self.image = nn.Linear(dim, dim, bias=False)
        self.text = nn.Linear(dim, dim, bias=False)
        nn.init.eye_(self.image.weight)
        nn.init.eye_(self.text.weight)

    def forward_image(self, image: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(self.image(image.float()), dim=-1)

    def forward_text(self, text: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(self.text(text.float()), dim=-1)


def make_adapter(policy_type: str, dim: int = 512) -> nn.Module:
    if policy_type == "linear_image_adapter":
        return LinearImageAdapter(dim)
    if policy_type == "dual_linear_adapter":
        return DualLinearAdapter(dim)
    raise ValueError(f"Unsupported adapter policy: {policy_type}")

