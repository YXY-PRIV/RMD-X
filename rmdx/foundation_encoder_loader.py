"""Load local crop-text foundation encoders registered by Batch50."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class LoadedFoundationEncoder:
    encoder_id: str
    family: str
    model: Any
    processor: Any
    device: str
    tokenizer: Any = None


def load_registry_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def load_foundation_encoder(config_path: str | Path) -> LoadedFoundationEncoder:
    """Load a ready local text+image encoder.

    Only local paths from the Batch50 registry are accepted. This function
    deliberately does not download remote weights.
    """
    cfg = load_registry_config(config_path)
    if not cfg.get("ready_for_fcr"):
        raise RuntimeError(f"No ready FCR encoder in registry: {cfg.get('reason', 'unknown')}")

    family = cfg.get("encoder_family", "")
    device = cfg.get("device", "cpu")
    if family in {"transformers_clip", "transformers_siglip"}:
        from transformers import AutoModel, AutoProcessor  # type: ignore

        model_dir = cfg["model_name"]
        processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
        model = AutoModel.from_pretrained(model_dir, local_files_only=True).to(device)
        model.eval()
        return LoadedFoundationEncoder(cfg["encoder_id"], family, model, processor, device)

    if family == "open_clip":
        import torch
        import open_clip  # type: ignore

        model_name = cfg.get("model_name") or "ViT-B-32"
        weight_path = cfg.get("weight_path")
        if not weight_path or not Path(weight_path).exists():
            raise RuntimeError(f"OpenCLIP weight path missing: {weight_path}")
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=None,
            device=device,
        )
        state = torch.load(weight_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        if len(missing) > 0 or len(unexpected) > 0:
            # OpenCLIP checkpoints can have minor metadata differences, but a
            # large mismatch is a real smoke blocker.
            total = len(state) if hasattr(state, "__len__") else 0
            if len(missing) > max(20, int(total * 0.05)):
                raise RuntimeError(f"OpenCLIP state_dict mismatch; missing={len(missing)} unexpected={len(unexpected)}")
        model.to(device)
        model.eval()
        tokenizer = open_clip.get_tokenizer(model_name)
        return LoadedFoundationEncoder(cfg["encoder_id"], family, model, preprocess, device, tokenizer=tokenizer)

    raise RuntimeError(f"Unsupported local encoder family for smoke: {family}")
