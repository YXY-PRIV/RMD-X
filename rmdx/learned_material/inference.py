from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from rmdx.learned_material.model import build_model


def load_json(path: str | Path) -> Any:
    with Path(path).open() as f:
        return json.load(f)


def save_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_adapter(checkpoint_path: str | Path, device: str = "cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = build_model(config["config_name"], input_dim=len(checkpoint["feature_names"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint


@torch.no_grad()
def embed_proposal_features(
    proposal_features: torch.Tensor,
    checkpoint: dict,
    model: torch.nn.Module,
    *,
    device: str = "cpu",
    batch_size: int = 8192,
) -> torch.Tensor:
    mean = torch.tensor(checkpoint["feature_mean"], dtype=torch.float32)
    std = torch.tensor(checkpoint["feature_std"], dtype=torch.float32).clamp_min(1e-6)
    if proposal_features.shape[1] != len(checkpoint["feature_names"]):
        raise ValueError(
            f"proposal feature dim {proposal_features.shape[1]} != adapter input dim {len(checkpoint['feature_names'])}"
        )
    outputs = []
    for start in range(0, proposal_features.shape[0], batch_size):
        x = proposal_features[start : start + batch_size].float()
        x = (x - mean) / std
        z = model(x.to(device)).cpu()
        outputs.append(z)
    return torch.cat(outputs, dim=0)


def load_learned_bank_prototypes(
    bank_path: str | Path,
    category_meta: list[dict[str, Any]],
) -> tuple[torch.Tensor, list[int]]:
    bank = load_json(bank_path)
    prototypes_by_cat = bank.get("prototypes", {})
    dim = int(bank.get("embedding_dim", 0))
    rows = []
    missing = []
    for cat in category_meta:
        category_id = int(cat["category_id"])
        item = prototypes_by_cat.get(str(category_id))
        if item is None:
            rows.append(torch.zeros(dim, dtype=torch.float32))
            missing.append(category_id)
        else:
            rows.append(torch.tensor(item["learned_prototype_vector"], dtype=torch.float32))
    proto = F.normalize(torch.stack(rows, dim=0), p=2, dim=-1)
    return proto, missing

