"""Schema checks for non-test calibration caches."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


REQUIRED_CACHE_FILES = [
    "proposal_meta.jsonl",
    "visual_similarity.pt",
    "material_class_similarity.pt",
    "category_meta.json",
    "prototype_meta.json",
    "cache_summary.json",
]


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if limit is not None and idx >= limit:
                break
            rows.append(json.loads(line))
    return rows


def tensor_info(path: str | Path) -> dict[str, Any]:
    tensor = torch.load(path, map_location="cpu")
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}


def cache_file_status(cache_dir: str | Path) -> dict[str, Any]:
    cache_dir = Path(cache_dir)
    files = {name: (cache_dir / name).exists() for name in REQUIRED_CACHE_FILES}
    return {
        "cache_dir": str(cache_dir),
        "cache_dir_exists": cache_dir.exists(),
        "required_files_present": all(files.values()),
        "missing_files": [name for name, ok in files.items() if not ok],
        "files": files,
    }
