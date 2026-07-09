#!/usr/bin/env python3
"""Export Batch18 cache manifest and checksums without rebuilding cache."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch


CACHE_DIR = Path("/workspace/RAXO/outputs_rmdx/cache/det_compass_groundingDINO_100_0_full")
OUT_DIR = Path("/workspace/RAXO/outputs_rmdx/cache_manifest")
MANIFEST = OUT_DIR / "det_compass_groundingDINO_100_0_full_manifest.json"
CHECKSUMS = OUT_DIR / "det_compass_groundingDINO_100_0_full_file_checksums.json"
REUSE_NOTES = OUT_DIR / "det_compass_groundingDINO_100_0_full_reuse_notes.md"

REQUIRED = {
    "proposal_meta": "proposal_meta.jsonl",
    "proposal_features": "proposal_features.pt",
    "visual_similarity": "visual_similarity.pt",
    "material_similarity": "material_class_similarity.pt",
    "prototype_meta": "prototype_meta.json",
    "category_meta": "category_meta.json",
    "summary": "cache_summary.json",
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024 * 16) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_info(path: Path) -> dict[str, Any] | None:
    if path.suffix != ".pt":
        return None
    tensor = torch.load(path, map_location="cpu")
    if not torch.is_tensor(tensor):
        return {"type": str(type(tensor))}
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
    }


def file_info(path: Path) -> dict[str, Any]:
    stat = path.stat()
    info = {
        "path": str(path),
        "name": path.name,
        "size_bytes": stat.st_size,
        "modified_time_unix": stat.st_mtime,
    }
    tinfo = tensor_info(path)
    if tinfo is not None:
        info["tensor"] = tinfo
    return info


def write_reuse_notes(summary: dict[str, Any]) -> None:
    lines = [
        "# DET-COMPASS groundingDINO 100/0 Full Cache Reuse Notes",
        "",
        "## Applies To",
        "",
        "- dataset = DET-COMPASS",
        "- detector = groundingDINO",
        "- setting = 100/0",
        "- nms = 0.8",
        "- image dir = `/datasets/xray-datasets/DET-COMPASS/test`",
        "- prototypes = `/models/ICCV25_experimentation/known_prototypes_sam2.pt`",
        "- support material bank = `/workspace/RAXO/outputs_rmdx/material/material_colour_bank_support_full_bbox_only.json`",
        "- with_masks input = `/models/ICCV25_experimentation/initial_detections/groundingDINO/coco_results.bbox_DET-COMPASS_with_masks.json`",
        "",
        "## Does Not Apply To",
        "",
        "- other detectors",
        "- other datasets",
        "- other descriptor settings",
        "- changed NMS threshold",
        "- changed prototype file",
        "- changed with_masks input",
        "- changed support material bank",
        "",
        "## Future Batches",
        "",
        "- Batch19 multi-detector: each detector needs a separate cache.",
        "- Batch22 multi-dataset: each dataset needs a separate cache.",
        "- Batch27-29 multi-channel material:",
        "  - `visual_similarity.pt` can be reused.",
        "  - `proposal_features.pt` can be reused.",
        "  - `proposal_meta.jsonl` can be reused.",
        "  - `material_class_similarity.pt` must be rebuilt.",
        "  - manifest and checksum validation must run before reuse.",
        "",
        "## Summary",
        "",
        f"- proposal_count: `{summary.get('proposal_count')}`",
        f"- image_count: `{summary.get('image_count')}`",
        f"- prototype_count: `{summary.get('prototype_count')}`",
        f"- category_count: `{summary.get('category_count')}`",
        f"- feature_dim: `{summary.get('feature_dim')}`",
        f"- build_time_seconds: `{summary.get('build_time_seconds')}`",
    ]
    REUSE_NOTES.write_text("\n".join(lines) + "\n")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    missing = [name for name in REQUIRED.values() if not (CACHE_DIR / name).exists()]
    if missing:
        raise SystemExit(f"missing cache files: {missing}")

    summary = json.load((CACHE_DIR / "cache_summary.json").open())
    file_list = []
    for path in sorted(CACHE_DIR.iterdir()):
        if path.is_file():
            file_list.append(file_info(path))

    manifest = {
        "cache_dir": str(CACHE_DIR),
        "required_files": {key: str(CACHE_DIR / value) for key, value in REQUIRED.items()},
        "files": file_list,
        "proposal_count": summary.get("proposal_count"),
        "image_count": summary.get("image_count"),
        "prototype_count": summary.get("prototype_count"),
        "category_count": summary.get("category_count"),
        "feature_dim": summary.get("feature_dim"),
        "dtype": summary.get("dtype"),
        "source_with_masks_path": summary.get("input_paths", {}).get("json_res"),
        "source_prototype_path": summary.get("input_paths", {}).get("prototypes"),
        "source_material_bank_path": summary.get("input_paths", {}).get("material_bank"),
        "build_script": summary.get("script_name"),
        "build_time_seconds": summary.get("build_time_seconds"),
        "nms": summary.get("nms"),
        "max_images": summary.get("max_images"),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2))

    checksum_rows = {}
    for name in REQUIRED.values():
        path = CACHE_DIR / name
        checksum_rows[name] = {
            "path": str(path),
            "algorithm": "sha256",
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    CHECKSUMS.write_text(json.dumps(checksum_rows, indent=2))
    write_reuse_notes(summary)

    print(json.dumps({"manifest": str(MANIFEST), "checksums": str(CHECKSUMS), "reuse_notes": str(REUSE_NOTES)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
