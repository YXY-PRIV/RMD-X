"""FCR encoder asset utilities.

The current implementation never downloads weights.  It can reuse existing
RMD-X DINOv2 proposal embeddings and visual similarities as an image-only
prototype fallback when no local text encoder is available.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


def module_available(name: str) -> tuple[bool, str]:
    spec = importlib.util.find_spec(name)
    return bool(spec), "" if spec is None else str(spec.origin)


def find_local_foundation_assets(paths: list[Path]) -> list[str]:
    patterns = ("*clip*", "*siglip*", "*openclip*", "*dinov2*", "*eva*", "*convnext*", "*.safetensors", "*.bin", "*.pth", "*.pt")
    hits: list[str] = []
    for root in paths:
        if not root.exists():
            continue
        for pattern in patterns:
            for path in root.rglob(pattern):
                if path.is_file() or path.is_dir():
                    hits.append(str(path))
                    if len(hits) >= 300:
                        return sorted(set(hits))
    return sorted(set(hits))


def detect_encoder_status(repo_root: Path) -> dict[str, Any]:
    modules = {}
    for name in ["open_clip", "clip", "transformers", "timm", "torchvision", "torch", "PIL", "safetensors"]:
        ok, origin = module_available(name)
        modules[name] = {"ok": ok, "origin": origin}

    asset_roots = [
        Path("/mnt/data/raxo/torch_cache/hub"),
        Path("/home/xinxueshi/.cache/torch/hub"),
        Path("/home/xinxueshi/.cache/huggingface"),
        repo_root / "outputs_rmdx",
    ]
    assets = find_local_foundation_assets(asset_roots)
    dinov2_repo = Path("/mnt/data/raxo/torch_cache/hub/facebookresearch_dinov2_main")
    dinov2_weights = [
        Path("/mnt/data/raxo/torch_cache/hub/checkpoints/dinov2_vitb14_pretrain.pth"),
        Path("/mnt/data/raxo/torch_cache/hub/checkpoints/dinov2_vitb14_reg4_pretrain.pth"),
    ]
    has_dinov2 = dinov2_repo.exists() and any(p.exists() for p in dinov2_weights)
    has_text_encoder = bool(modules["open_clip"]["ok"] or modules["clip"]["ok"])
    has_transformers_local = modules["transformers"]["ok"] and any(("clip" in p.lower() or "siglip" in p.lower()) and (p.endswith(".bin") or p.endswith(".safetensors") or p.endswith("config.json")) for p in assets)
    if has_text_encoder or has_transformers_local:
        final_status = "READY_TEXT_ENCODER_LOCAL"
        selected_encoder = "local_crop_text_encoder"
    elif has_dinov2:
        final_status = "READY_IMAGE_ONLY_DINOV2_LOCAL"
        selected_encoder = "dinov2_local_image_only_proto_fallback"
    else:
        final_status = "BLOCKED_FOUNDATION_ENCODER_UNAVAILABLE"
        selected_encoder = "none"
    return {
        "final_status": final_status,
        "selected_encoder": selected_encoder,
        "text_encoder_available": bool(has_text_encoder or has_transformers_local),
        "image_encoder_available": bool(has_dinov2),
        "dinov2_repo": str(dinov2_repo) if dinov2_repo.exists() else "",
        "dinov2_weight_paths": [str(p) for p in dinov2_weights if p.exists()],
        "modules": modules,
        "asset_hits": assets[:200],
        "downloads_performed": False,
    }

