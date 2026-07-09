"""Local foundation encoder registry utilities for Batch50.

This module only audits and records local assets. It does not download,
install, move, or modify model files.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import os
from pathlib import Path
from typing import Any


REPO_ROOT = Path("/mnt/data/yxy/program/RAXO")
OUT_ROOT = REPO_ROOT / "outputs_rmdx" / "foundation_encoder"

SEARCH_ROOTS = [
    Path("/mnt/data"),
    Path("/mnt/data/yxy"),
    Path("/mnt/data/yxy/program"),
    Path("/home/xinxueshi"),
    Path("/home/xinxueshi/yxy"),
    Path("/home/xinxueshi/.cache"),
    REPO_ROOT / "models",
    REPO_ROOT / "checkpoints",
    REPO_ROOT / "weights",
    REPO_ROOT / "pretrained",
]

WEIGHT_SUFFIXES = {".pt", ".pth", ".bin", ".safetensors", ".ckpt"}
CONFIG_NAMES = {"config.json", "open_clip_config.json", "preprocessor_config.json", "tokenizer_config.json"}
KEYWORDS = (
    "clip",
    "open_clip",
    "siglip",
    "laion",
    "vit",
    "vit-b",
    "vit-l",
    "vit-h",
    "dinov2",
    "eva",
    "convnext",
    "resnet",
    "bert",
)


def module_available(name: str) -> dict[str, Any]:
    spec = importlib.util.find_spec(name)
    return {"module": name, "available": spec is not None, "origin": getattr(spec, "origin", "") if spec else ""}


def ensure_dirs() -> None:
    for path in [
        OUT_ROOT / "audit",
        OUT_ROOT / "install_plan",
        OUT_ROOT / "registry",
        OUT_ROOT / "smoke",
        REPO_ROOT / "outputs_rmdx" / "rmdx_results",
        REPO_ROOT / "docs_rmdx",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def safe_rel(path: Path) -> str:
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


def classify_path(path: Path) -> str:
    lower = str(path).lower()
    if "siglip" in lower:
        return "siglip"
    if "open_clip" in lower or "laion" in lower:
        return "open_clip"
    if "clip" in lower:
        return "clip"
    if "dinov2" in lower:
        return "dinov2"
    if "resnet" in lower:
        return "resnet"
    if "bert" in lower:
        return "text_only_bert"
    if "vit" in lower:
        return "vit_unknown"
    return "unknown"


def scan_local_assets(max_files: int = 20000) -> list[dict[str, Any]]:
    """Find local candidate weights/configs without changing the filesystem."""
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    count = 0
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            # Keep the audit responsive. Cache/package internals can be huge.
            dirnames[:] = [
                d
                for d in dirnames
                if d not in {".git", "__pycache__", "node_modules", ".venv", "env", "site-packages"}
            ]
            pdir = Path(dirpath)
            lower_dir = str(pdir).lower()
            interesting_dir = any(k in lower_dir for k in KEYWORDS)
            for name in filenames:
                path = pdir / name
                suffix = path.suffix.lower()
                lower = name.lower()
                is_weight = suffix in WEIGHT_SUFFIXES
                is_config = lower in CONFIG_NAMES
                keyword_hit = any(k in lower or k in lower_dir for k in KEYWORDS)
                if not ((is_weight and keyword_hit) or (is_config and (keyword_hit or interesting_dir))):
                    continue
                resolved = safe_rel(path)
                if resolved in seen:
                    continue
                seen.add(resolved)
                try:
                    size = path.stat().st_size
                except OSError:
                    size = ""
                rows.append(
                    {
                        "path": resolved,
                        "asset_type": "weight" if is_weight else "config",
                        "size_bytes": size,
                        "encoder_family_guess": classify_path(path),
                    }
                )
                count += 1
                if count >= max_files:
                    rows.append(
                        {
                            "path": "",
                            "asset_type": "scan_truncated",
                            "size_bytes": "",
                            "encoder_family_guess": "audit_limit_reached",
                        }
                    )
                    return rows
    return rows


def find_transformers_model_dirs(asset_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Inspect local config.json files for CLIP/SigLIP-style transformers models."""
    out: list[dict[str, Any]] = []
    config_paths = [Path(r["path"]) for r in asset_rows if r.get("asset_type") == "config" and Path(r.get("path", "")).name == "config.json"]
    for config_path in config_paths:
        try:
            data = json.loads(config_path.read_text())
        except Exception as exc:  # noqa: BLE001
            out.append({"model_dir": str(config_path.parent), "readable": False, "error": str(exc)})
            continue
        text = json.dumps(data).lower()
        family = "unknown"
        if "siglip" in text:
            family = "siglip"
        elif "clip" in text:
            family = "clip"
        elif "bert" in text:
            family = "text_only_bert"
        processor = any((config_path.parent / n).exists() for n in ["preprocessor_config.json", "processor_config.json"])
        tokenizer = any((config_path.parent / n).exists() for n in ["tokenizer.json", "tokenizer_config.json", "vocab.txt", "merges.txt"])
        weight = any(p.suffix.lower() in WEIGHT_SUFFIXES for p in config_path.parent.glob("*"))
        supports_both = family in {"clip", "siglip"} and processor and tokenizer and weight
        out.append(
            {
                "model_dir": str(config_path.parent),
                "readable": True,
                "family": family,
                "processor_or_preprocessor": processor,
                "tokenizer": tokenizer,
                "weight_file_in_dir": weight,
                "supports_crop_text_similarity": supports_both,
                "config_model_type": data.get("model_type", ""),
                "architectures": ";".join(data.get("architectures", []) or []),
            }
        )
    return out


def audit_environment() -> dict[str, Any]:
    imports = [module_available(m) for m in ["open_clip", "clip", "transformers", "timm", "torch", "torchvision", "PIL", "cv2", "safetensors", "sentencepiece"]]
    import_map = {r["module"]: r for r in imports}
    assets = scan_local_assets()
    transformers_dirs = find_transformers_model_dirs(assets)

    ready_transformers = [d for d in transformers_dirs if d.get("supports_crop_text_similarity") and import_map["transformers"]["available"]]
    open_clip_weights = [r for r in assets if r.get("asset_type") == "weight" and r.get("encoder_family_guess") == "open_clip"]
    clip_weights = [r for r in assets if r.get("asset_type") == "weight" and r.get("encoder_family_guess") == "clip"]
    dinov2_weights = [r for r in assets if r.get("asset_type") == "weight" and r.get("encoder_family_guess") == "dinov2"]
    dinov2_weights = sorted(dinov2_weights, key=lambda r: ("linear" in str(r.get("path", "")).lower(), str(r.get("path", ""))))

    if ready_transformers:
        final_status = "READY_TRANSFORMERS_LOCAL"
        selected = ready_transformers[0]["model_dir"]
    elif import_map["open_clip"]["available"] and open_clip_weights:
        final_status = "READY_OPENCLIP_LOCAL"
        selected = open_clip_weights[0]["path"]
    elif import_map["clip"]["available"] and clip_weights:
        final_status = "READY_CLIP_LOCAL"
        selected = clip_weights[0]["path"]
    elif dinov2_weights:
        final_status = "READY_IMAGE_ONLY_LOCAL"
        selected = dinov2_weights[0]["path"]
    elif any(import_map[m]["available"] for m in ["open_clip", "clip", "transformers"]):
        final_status = "BLOCKED_WEIGHT_MISSING"
        selected = ""
    else:
        final_status = "BLOCKED_DEPENDENCY_MISSING"
        selected = ""

    text_ready = final_status in {"READY_TRANSFORMERS_LOCAL", "READY_OPENCLIP_LOCAL", "READY_CLIP_LOCAL", "READY_SIGLIP_LOCAL"}
    if not text_ready and final_status == "READY_IMAGE_ONLY_LOCAL":
        blocker = "text_encoder_missing; image-only encoder cannot unlock full FCR"
    elif not text_ready:
        blocker = final_status
    else:
        blocker = ""

    return {
        "imports": imports,
        "assets": assets,
        "transformers_model_dirs": transformers_dirs,
        "final_status": final_status,
        "selected": selected,
        "text_encoder_available": text_ready,
        "image_encoder_available": bool(text_ready or dinov2_weights),
        "blocker": blocker,
        "downloads_performed": False,
        "installs_performed": False,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
