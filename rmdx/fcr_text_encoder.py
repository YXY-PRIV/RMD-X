"""Offline OpenCLIP wrapper for FCR-RMD-X Batch51."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

ENCODER_ROOT = Path("/mnt/data/yxy/models/foundation_encoder/openclip_vit_b32")
MANIFEST = ENCODER_ROOT / "encoder_manifest.json"


@dataclass
class EncodeResult:
    embeddings: torch.Tensor
    runtime_sec: float
    embedding_dim: int
    device: str


class OpenCLIPTextImageEncoder:
    def __init__(self, device: str = "cuda:0") -> None:
        self.device = device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
        try:
            import open_clip  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"open_clip import failed: {exc}") from exc
        self.open_clip = open_clip
        self.manifest = json.loads(MANIFEST.read_text())
        self.model_name = self.manifest.get("model_name", "ViT-B-32")
        self.weight_path = Path(self.manifest["weight_path"])
        if not self.weight_path.exists():
            raise RuntimeError(f"OpenCLIP weight missing: {self.weight_path}")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=None,
            device=self.device,
        )
        state = torch.load(self.weight_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing or unexpected:
            total = len(state) if hasattr(state, "__len__") else 0
            if len(missing) > max(20, int(total * 0.05)):
                raise RuntimeError(f"OpenCLIP state_dict mismatch: missing={len(missing)} unexpected={len(unexpected)}")
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(self.model_name)
        self.embedding_dim = int(getattr(self.model, "text_projection").shape[1])

    def _check(self, tensor: torch.Tensor, name: str) -> torch.Tensor:
        if torch.isnan(tensor).any():
            raise RuntimeError(f"{name} contains NaN")
        if torch.isinf(tensor).any():
            raise RuntimeError(f"{name} contains Inf")
        return tensor

    def encode_texts(self, texts: list[str], batch_size: int = 128) -> EncodeResult:
        start = time.time()
        outputs: list[torch.Tensor] = []
        with torch.no_grad():
            for offset in range(0, len(texts), batch_size):
                batch = texts[offset : offset + batch_size]
                tokens = self.tokenizer(batch).to(self.device)
                emb = self.model.encode_text(tokens).float()
                emb = torch.nn.functional.normalize(emb, dim=-1)
                outputs.append(self._check(emb.detach().cpu(), "text_embeddings"))
        tensor = torch.cat(outputs, dim=0) if outputs else torch.empty((0, self.embedding_dim), dtype=torch.float32)
        return EncodeResult(tensor, round(time.time() - start, 3), int(tensor.shape[1]) if tensor.ndim == 2 and tensor.shape[0] else self.embedding_dim, self.device)

    def encode_images(self, images: list[Image.Image], batch_size: int = 128) -> EncodeResult:
        start = time.time()
        outputs: list[torch.Tensor] = []
        with torch.no_grad():
            for offset in range(0, len(images), batch_size):
                batch = images[offset : offset + batch_size]
                inputs = torch.stack([self.preprocess(img.convert("RGB")) for img in batch]).to(self.device)
                emb = self.model.encode_image(inputs).float()
                emb = torch.nn.functional.normalize(emb, dim=-1)
                outputs.append(self._check(emb.detach().cpu(), "image_embeddings"))
        tensor = torch.cat(outputs, dim=0) if outputs else torch.empty((0, self.embedding_dim), dtype=torch.float32)
        return EncodeResult(tensor, round(time.time() - start, 3), int(tensor.shape[1]) if tensor.ndim == 2 and tensor.shape[0] else self.embedding_dim, self.device)


def load_model(device: str = "cuda:0") -> OpenCLIPTextImageEncoder:
    return OpenCLIPTextImageEncoder(device=device)

