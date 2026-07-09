"""Build reusable RMD-X pre-argmax feature caches.

The cache stores proposal features and similarity tensors that are independent
of lambda sweeps. It does not use GT matching information and does not modify
RAXO baseline files.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from pycocotools.coco import COCO
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
RAXO_DIR = REPO_ROOT / "raxo"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(RAXO_DIR) not in sys.path:
    sys.path.insert(0, str(RAXO_DIR))

from dataset import CocoPrototypes_with_masks
from rmdx.preargmax_fusion import (
    build_category_material_matrix,
    compute_material_similarity_for_proposals,
    load_material_bank,
)
from rmdx.rmdx_main_with_masks_prop import (
    assemble_prototypes,
    attach_file_names,
    build_model,
    custom_collate_fn,
    limit_coco_images,
    apply_nms,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build RMD-X pre-argmax cache")
    parser.add_argument("--json-res", required=True)
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--prototypes", required=True)
    parser.add_argument("--material-bank", required=True)
    parser.add_argument("--output-cache-dir", required=True)
    parser.add_argument("--nms", type=float, default=0.8)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--shard-size", type=int, default=None)
    return parser.parse_args()


def tensor_dtype(name: str) -> torch.dtype:
    return torch.float16 if name == "float16" else torch.float32


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def build_category_meta(categories: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows = []
    name_to_index = {}
    for idx, cat in enumerate(categories):
        name = str(cat["name"])
        name_to_index[name] = idx
        rows.append(
            {
                "class_index": idx,
                "category_id": int(cat["id"]),
                "category_name": name,
            }
        )
    return rows, name_to_index


def build_prototype_meta(
    cat_names: list[str],
    prototype_keys: list[str],
    category_name_to_id: dict[str, int],
    category_name_to_class_index: dict[str, int],
) -> list[dict[str, Any]]:
    original_mapper = {key: idx for idx, key in enumerate(prototype_keys)}
    rows = []
    for idx, (category_name, prototype_key) in enumerate(zip(cat_names, prototype_keys)):
        is_negative = category_name == "negative"
        if category_name in original_mapper:
            canonical_index = original_mapper[category_name]
        else:
            canonical_index = next(
                (candidate_idx for candidate_idx, name in enumerate(cat_names) if name == category_name),
                idx,
            )
        rows.append(
            {
                "prototype_index": idx,
                "prototype_key": prototype_key,
                "supercategory": category_name,
                "category_id": None if is_negative else category_name_to_id.get(category_name),
                "category_name": None if is_negative else category_name,
                "is_negative": is_negative,
                "canonical_index": canonical_index,
                "material_class_index": None
                if is_negative
                else category_name_to_class_index.get(category_name),
            }
        )
    return rows


def proposal_meta_from_annotation(
    ann: dict[str, Any],
    cache_index: int,
    images_by_id: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    image_id = int(ann["image_id"])
    image = images_by_id[image_id]
    return {
        "cache_index": cache_index,
        "image_id": image_id,
        "annotation_id": ann.get("id"),
        "file_name": image.get("file_name"),
        "bbox": [float(v) for v in ann["bbox"]],
        "detector_score": float(ann.get("score", 0.0)),
        "detector_category_id": int(ann["category_id"]) if ann.get("category_id") is not None else None,
        "width": int(image.get("width", 0)),
        "height": int(image.get("height", 0)),
    }


def build_cache(args: argparse.Namespace) -> dict[str, Any]:
    start = time.time()
    out_dir = Path(args.output_cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = tensor_dtype(args.dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[cache_builder] device={device}", flush=True)
    if device != "cuda":
        raise RuntimeError("cache_builder requires CUDA for DINOv2 feature extraction")

    coco = COCO(args.json_res)
    limit_coco_images(coco, args.max_images)
    categories = list(coco.dataset["categories"])
    category_name_to_id = {str(c["name"]): int(c["id"]) for c in categories}
    category_meta, category_name_to_class_index = build_category_meta(categories)
    vocab = list(category_name_to_id.keys())

    print("[cache_builder] assembling prototypes", flush=True)
    prototypes, cat_names, prototype_keys = assemble_prototypes(args.prototypes, vocab)
    prototypes = prototypes.cuda()
    prototype_meta = build_prototype_meta(
        cat_names,
        prototype_keys,
        category_name_to_id,
        category_name_to_class_index,
    )

    print("[cache_builder] preparing material lookup", flush=True)
    material_lookup = build_category_material_matrix(
        vocab,
        vocab,
        load_material_bank(args.material_bank),
    )

    print("[cache_builder] applying NMS", flush=True)
    nms_annotations = apply_nms(coco, float(args.nms))
    print(f"[cache_builder] proposals after NMS={len(nms_annotations)}", flush=True)
    coco.dataset["annotations"] = nms_annotations
    attach_file_names(coco)
    coco.createIndex()

    print("[cache_builder] building DINO model", flush=True)
    transforms, model = build_model()
    data = CocoPrototypes_with_masks(coco, args.image_path, transforms)
    dataloader = DataLoader(
        data,
        batch_size=256,
        shuffle=False,
        num_workers=8,
        collate_fn=custom_collate_fn,
    )
    print(f"[cache_builder] proposals after sanity={len(data)}", flush=True)

    proposal_rows: list[dict[str, Any]] = []
    feature_chunks: list[torch.Tensor] = []
    visual_chunks: list[torch.Tensor] = []
    material_chunks: list[torch.Tensor] = []
    images_by_id = {int(img["id"]): img for img in coco.dataset["images"]}
    cache_index = 0

    for crops, annotations, masks in tqdm(dataloader, desc="[cache_builder] features"):
        crops = crops.cuda()
        masks = masks.cuda()
        with torch.no_grad():
            outputs = model.get_intermediate_layers(crops)[0]
            masks_expanded = masks.unsqueeze(-1)
            masked_t = outputs * masks_expanded
            sum_masked = masked_t.sum(dim=1, keepdim=True)
            count_masked = masks_expanded.sum(dim=1, keepdim=True).clamp(min=1)
            features = (sum_masked / count_masked).squeeze(1)
            visual_similarity = torch.nn.functional.cosine_similarity(
                features.unsqueeze(1),
                prototypes.unsqueeze(0),
                dim=-1,
            )
            material_similarity = compute_material_similarity_for_proposals(
                annotations,
                args.image_path,
                material_lookup,
                device=visual_similarity.device,
            )

        for ann in annotations:
            proposal_rows.append(proposal_meta_from_annotation(ann, cache_index, images_by_id))
            cache_index += 1
        feature_chunks.append(features.detach().cpu().to(dtype))
        visual_chunks.append(visual_similarity.detach().cpu().to(dtype))
        material_chunks.append(material_similarity.detach().cpu().to(dtype))

    proposal_features = torch.cat(feature_chunks, dim=0) if feature_chunks else torch.empty((0, 768), dtype=dtype)
    visual_similarity = torch.cat(visual_chunks, dim=0) if visual_chunks else torch.empty((0, len(prototype_meta)), dtype=dtype)
    material_similarity = torch.cat(material_chunks, dim=0) if material_chunks else torch.empty((0, len(category_meta)), dtype=dtype)

    print(f"[cache_builder] proposal_features={tuple(proposal_features.shape)}", flush=True)
    print(f"[cache_builder] visual_similarity={tuple(visual_similarity.shape)}", flush=True)
    print(f"[cache_builder] material_class_similarity={tuple(material_similarity.shape)}", flush=True)

    torch.save(proposal_features, out_dir / "proposal_features.pt")
    torch.save(visual_similarity, out_dir / "visual_similarity.pt")
    torch.save(material_similarity, out_dir / "material_class_similarity.pt")
    write_jsonl(out_dir / "proposal_meta.jsonl", proposal_rows)
    write_json(out_dir / "prototype_meta.json", prototype_meta)
    write_json(out_dir / "category_meta.json", category_meta)
    summary = {
        "proposal_count": len(proposal_rows),
        "image_count": len(coco.dataset["images"]),
        "prototype_count": len(prototype_meta),
        "category_count": len(category_meta),
        "feature_dim": int(proposal_features.shape[1]) if proposal_features.ndim == 2 else None,
        "dtype": args.dtype,
        "build_time_seconds": time.time() - start,
        "input_paths": {
            "json_res": args.json_res,
            "image_path": args.image_path,
            "prototypes": args.prototypes,
            "material_bank": args.material_bank,
        },
        "script_name": "rmdx/cache_builder.py",
        "nms": float(args.nms),
        "max_images": args.max_images,
        "dino_source": "local cache preferred by rmdx_main_with_masks_prop.build_model",
        "notes": [
            "No GT matching information is stored.",
            "Material similarity uses support-set colour material bank only.",
        ],
    }
    write_json(out_dir / "cache_summary.json", summary)
    print(f"[cache_builder] cache output={out_dir}", flush=True)
    return summary


def main() -> int:
    summary = build_cache(parse_args())
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
