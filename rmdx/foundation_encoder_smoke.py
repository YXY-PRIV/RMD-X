"""Smoke utilities for local crop-text foundation encoders."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from rmdx.foundation_encoder_loader import LoadedFoundationEncoder


def read_first_proposals(cache_dir: Path, image_root: Path, limit: int = 16) -> list[dict[str, Any]]:
    proposals: list[dict[str, Any]] = []
    with (cache_dir / "proposal_meta.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            path = image_root / row["file_name"]
            if path.exists():
                row["image_path"] = str(path)
                proposals.append(row)
            if len(proposals) >= limit:
                break
    return proposals


def crop_proposals(proposals: list[dict[str, Any]]) -> list[Image.Image]:
    crops: list[Image.Image] = []
    for row in proposals:
        img = Image.open(row["image_path"]).convert("RGB")
        x, y, w, h = [float(v) for v in row["bbox"]]
        x1 = max(0, int(round(x)))
        y1 = max(0, int(round(y)))
        x2 = min(img.width, int(round(x + w)))
        y2 = min(img.height, int(round(y + h)))
        if x2 <= x1 or y2 <= y1:
            x1, y1, x2, y2 = 0, 0, img.width, img.height
        crops.append(img.crop((x1, y1, x2, y2)))
    return crops


def read_prompts(prompt_path: Path, limit_categories: int = 32) -> list[str]:
    data = json.loads(prompt_path.read_text())
    prompts: list[str] = []
    for cat in data.get("categories", [])[:limit_categories]:
        vals = cat.get("prompts") or []
        if vals:
            prompts.append(str(vals[0]))
    return prompts


def run_transformers_smoke(
    encoder: LoadedFoundationEncoder,
    crops: list[Image.Image],
    prompts: list[str],
) -> dict[str, Any]:
    start = time.time()
    processor = encoder.processor
    model = encoder.model
    device = encoder.device
    with torch.no_grad():
        image_inputs = processor(images=crops, return_tensors="pt").to(device)
        text_inputs = processor(text=prompts, padding=True, return_tensors="pt").to(device)
        image_features = model.get_image_features(**image_inputs)
        text_features = model.get_text_features(**text_inputs)
        image_features = torch.nn.functional.normalize(image_features.float(), dim=-1)
        text_features = torch.nn.functional.normalize(text_features.float(), dim=-1)
        similarity = image_features @ text_features.T
    return {
        "image_shape": list(image_features.shape),
        "text_shape": list(text_features.shape),
        "similarity_shape": list(similarity.shape),
        "nan_count": int(torch.isnan(similarity).sum().item()),
        "inf_count": int(torch.isinf(similarity).sum().item()),
        "runtime_sec": round(time.time() - start, 3),
        "similarity": similarity.detach().cpu(),
    }


def run_open_clip_smoke(
    encoder: LoadedFoundationEncoder,
    crops: list[Image.Image],
    prompts: list[str],
) -> dict[str, Any]:
    start = time.time()
    model = encoder.model
    preprocess = encoder.processor
    tokenizer = encoder.tokenizer
    device = encoder.device
    with torch.no_grad():
        image_inputs = torch.stack([preprocess(img) for img in crops]).to(device)
        text_inputs = tokenizer(prompts).to(device)
        image_features = model.encode_image(image_inputs)
        text_features = model.encode_text(text_inputs)
        image_features = torch.nn.functional.normalize(image_features.float(), dim=-1)
        text_features = torch.nn.functional.normalize(text_features.float(), dim=-1)
        similarity = image_features @ text_features.T
    return {
        "image_shape": list(image_features.shape),
        "text_shape": list(text_features.shape),
        "similarity_shape": list(similarity.shape),
        "nan_count": int(torch.isnan(similarity).sum().item()),
        "inf_count": int(torch.isinf(similarity).sum().item()),
        "std": float(similarity.std().item()) if similarity.numel() > 1 else 0.0,
        "runtime_sec": round(time.time() - start, 3),
        "similarity": similarity.detach().cpu(),
    }
