"""RMD-X pre-argmax variant of RAXO main_with_masks_prop.py.

Copied from RAXO `raxo/main_with_masks_prop.py` and modified for RMD-X
pre-argmax material/reliability fusion. The original RAXO file is untouched.
This smoke-focused wrapper writes only to an explicit --output-path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from pycocotools.coco import COCO
from torch.utils.data import DataLoader
from torchvision.ops import nms
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
RAXO_DIR = REPO_ROOT / "raxo"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(RAXO_DIR) not in sys.path:
    sys.path.insert(0, str(RAXO_DIR))

from dataset import CocoPrototypes_with_masks  # noqa: E402
from rmdx.preargmax_fusion import (  # noqa: E402
    build_category_material_matrix,
    compute_material_similarity_for_proposals,
    compute_reliability_gate,
    compute_visual_margin,
    fuse_similarity_material,
    fuse_similarity_material_reliability,
    fuse_similarity_visual_only,
    load_material_bank,
)


DINO_PATH_SIZE = 14
device = "cuda" if torch.cuda.is_available() else "cpu"
assert device == "cuda", "This script requires a GPU to run."


parser = argparse.ArgumentParser(description="RMD-X pre-argmax fusion smoke")
parser.add_argument("--json_res", required=True, help="COCO with_masks input")
parser.add_argument("--image_path", required=True, help="Image directory")
parser.add_argument("--prototypes", required=True, help="Path to known prototypes .pt")
parser.add_argument("--nms", required=True, help="NMS threshold")
parser.add_argument("--name", required=False, default="", help="Name to append")
parser.add_argument("--branch", required=True, choices=["known", "unknown"], default="known")
parser.add_argument(
    "--rmdx-mode",
    choices=["visual_only", "material_preargmax", "material_reliability_preargmax"],
    default="visual_only",
)
parser.add_argument(
    "--material-bank",
    default="/workspace/RAXO/outputs_rmdx/material/material_colour_bank_support_full_bbox_only.json",
)
parser.add_argument("--lambda-material", type=float, default=0.0)
parser.add_argument("--lambda-reliability", type=float, default=0.0)
parser.add_argument(
    "--reliability-gate",
    choices=["none", "multiply_material"],
    default="multiply_material",
)
parser.add_argument("--output-path", required=True)
parser.add_argument("--max-images", type=int, default=None)
parser.add_argument("--debug-limit-proposals", type=int, default=None)
parser.add_argument("--replace-score-with-fused", action="store_true")


def custom_collate_fn(batch):
    images = [item[0] for item in batch]
    dicts = [item[1] for item in batch]
    masks = [item[2] for item in batch]
    images = torch.stack(images, dim=0)
    masks = torch.stack(masks, dim=0)
    return images, dicts, masks


def limit_coco_images(coco: COCO, max_images: int | None) -> None:
    if max_images is None:
        return
    keep_ids = set(coco.getImgIds()[: int(max_images)])
    coco.dataset["images"] = [img for img in coco.dataset["images"] if int(img["id"]) in keep_ids]
    coco.dataset["annotations"] = [
        ann for ann in coco.dataset["annotations"] if int(ann["image_id"]) in keep_ids
    ]
    coco.createIndex()


def attach_file_names(coco: COCO) -> None:
    file_by_image = {int(img["id"]): img["file_name"] for img in coco.dataset.get("images", [])}
    for ann in coco.dataset.get("annotations", []):
        ann["file_name"] = file_by_image.get(int(ann["image_id"]))


def apply_nms(coco: COCO, iou_threshold: float = 0.5):
    new_annotations_nms = []
    image_ids = coco.getImgIds()
    for image_id in tqdm(image_ids):
        annotation_ids = coco.getAnnIds(imgIds=image_id)
        annotations = coco.loadAnns(annotation_ids)
        boxes = []
        scores = []
        for ann in annotations:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            scores.append(ann["score"])
        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            scores = torch.tensor(scores)
            keep_indices = nms(boxes, scores, iou_threshold)
            filtered_annotations = [annotations[i] for i in keep_indices]
            new_annotations_nms += filtered_annotations
    return new_annotations_nms


def build_model():
    transform = T.Compose(
        [
            T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    dinov2_repo = resolve_dinov2_repo()
    if dinov2_repo is not None:
        print(f"[RMD-X] Loading DINOv2 from local cache: {dinov2_repo}", flush=True)
        dinov2_vitb14_reg = torch.hub.load(
            str(dinov2_repo), "dinov2_vitb14_reg", source="local"
        )
    else:
        print("[RMD-X] Loading DINOv2 from remote torch.hub: facebookresearch/dinov2", flush=True)
        dinov2_vitb14_reg = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vitb14_reg", trust_repo=True
        )
    dinov2_vitb14_reg = dinov2_vitb14_reg.cuda()
    return transform, dinov2_vitb14_reg


def resolve_dinov2_repo() -> Path | None:
    env_repo = os.environ.get("RMDX_DINOV2_REPO")
    if env_repo:
        env_path = Path(env_repo).expanduser()
        if env_path.is_dir():
            return env_path
        print(f"[RMD-X] RMDX_DINOV2_REPO is set but invalid: {env_path}", flush=True)

    hub_dir = Path(torch.hub.get_dir())
    candidates = []
    if hub_dir.is_dir():
        candidates.extend(sorted(hub_dir.glob("facebookresearch_dinov2*")))
        candidates.extend(sorted(hub_dir.glob("*dinov2*")))

    for candidate in candidates:
        if not candidate.is_dir():
            continue
        if (candidate / "hubconf.py").is_file() or (candidate / "dinov2").is_dir():
            return candidate
    return None


def assemble_prototypes(prototypes_path, vocab):
    if not os.path.exists(prototypes_path):
        raise Exception(f"{prototypes_path} not exists")
    support_set = torch.load(prototypes_path)
    prototypes = []
    mapper = {}
    cat_in_each_index = []
    cat_in_each_index_original = []
    for cat, values in support_set.items():
        if values["supercategory"] in vocab or cat == "negative":
            if "neg_" in cat or cat == "negative":
                prototypes.append(values["prototype"])
                cat_in_each_index.append("negative")
                cat_in_each_index_original.append(cat)
            else:
                prototypes.append(values["prototype"])
                mapper[cat] = values["supercategory"]
                cat_in_each_index.append(values["supercategory"])
                cat_in_each_index_original.append(cat)
    support_set_tensor = torch.stack(prototypes)
    return support_set_tensor, cat_in_each_index, cat_in_each_index_original


def _canonical_index_for_category(
    category_name: str,
    original_mapper: dict[str, int],
    cat_name_in_each_index: list[str],
) -> int:
    if category_name in original_mapper:
        return original_mapper[category_name]
    for idx, name in enumerate(cat_name_in_each_index):
        if name == category_name:
            return idx
    raise KeyError(category_name)


def main_known(args):
    coco = COCO(args.json_res)
    limit_coco_images(coco, args.max_images)
    mapper_name_to_real = {c["name"]: c["id"] for c in coco.dataset["categories"]}
    vocab = list(mapper_name_to_real.keys())

    print("Assembling prototypes...")
    prototypes, cat_name_in_each_index, cat_in_each_index_original = assemble_prototypes(
        args.prototypes, vocab
    )
    cat_in_each_index_original_mapper = {x: i for i, x in enumerate(cat_in_each_index_original)}
    prototypes = prototypes.cuda()

    print("Preparing material lookup...")
    material_lookup = None
    if args.rmdx_mode != "visual_only":
        material_lookup = build_category_material_matrix(
            vocab,
            cat_name_in_each_index,
            load_material_bank(args.material_bank),
        )

    print("Applying NMS...")
    annotations_nms = apply_nms(coco, float(args.nms))
    if args.debug_limit_proposals is not None:
        annotations_nms = annotations_nms[: int(args.debug_limit_proposals)]
    coco.dataset["annotations"] = annotations_nms
    attach_file_names(coco)
    coco.createIndex()

    print("Building DINO model...")
    transforms, model = build_model()
    data = CocoPrototypes_with_masks(coco, args.image_path, transforms)
    dataloader = DataLoader(
        data,
        batch_size=256,
        shuffle=False,
        num_workers=8,
        collate_fn=custom_collate_fn,
    )

    print("RMD-X pre-argmax reclassification...")
    new_anots = []
    for batch in tqdm(dataloader):
        crops, annotations, masks = batch
        crops = crops.cuda()
        masks = masks.cuda()
        with torch.no_grad():
            outputs = model.get_intermediate_layers(crops)[0]
        masks_expanded = masks.unsqueeze(-1)
        masked_t = outputs * masks_expanded
        sum_masked = masked_t.sum(dim=1, keepdim=True)
        count_masked = masks_expanded.sum(dim=1, keepdim=True).clamp(min=1)
        outs = sum_masked / count_masked

        visual_similarity = torch.nn.functional.cosine_similarity(
            outs, prototypes.unsqueeze(0), dim=-1
        )
        detector_scores = torch.tensor(
            [float(anot["score"]) for anot in annotations],
            dtype=torch.float32,
            device=visual_similarity.device,
        )
        reliability_gate = torch.ones(
            visual_similarity.shape[0], dtype=torch.float32, device=visual_similarity.device
        )
        material_similarity = torch.zeros_like(visual_similarity)

        if args.rmdx_mode == "visual_only":
            fused_similarity = fuse_similarity_visual_only(visual_similarity)
        else:
            material_similarity = compute_material_similarity_for_proposals(
                annotations,
                args.image_path,
                material_lookup,
                device=visual_similarity.device,
            )
            if args.rmdx_mode == "material_preargmax":
                fused_similarity = fuse_similarity_material(
                    visual_similarity,
                    material_similarity,
                    args.lambda_material,
                )
            else:
                if args.reliability_gate == "none":
                    reliability_gate = torch.ones_like(detector_scores)
                    fused_similarity = fuse_similarity_material(
                        visual_similarity,
                        material_similarity,
                        args.lambda_material,
                    )
                else:
                    fused_similarity, reliability_gate = fuse_similarity_material_reliability(
                        visual_similarity,
                        material_similarity,
                        detector_scores,
                        args.lambda_material,
                        args.lambda_reliability,
                    )

        visual_similarity_cpu = visual_similarity.cpu()
        fused_similarity_cpu = fused_similarity.cpu()
        material_similarity_cpu = material_similarity.cpu()
        reliability_gate_cpu = reliability_gate.cpu()

        for s_visual, s_fused, s_material, gate, anot in zip(
            visual_similarity_cpu,
            fused_similarity_cpu,
            material_similarity_cpu,
            reliability_gate_cpu,
            annotations,
        ):
            _, visual_max_indice = s_visual.max(dim=0)
            _, max_indice = s_fused.max(dim=0)
            if cat_name_in_each_index[max_indice] == "negative":
                continue
            predicted_name = cat_name_in_each_index[max_indice]
            predicted_class = mapper_name_to_real[predicted_name]
            visual_name = cat_name_in_each_index[visual_max_indice]
            visual_class = mapper_name_to_real.get(visual_name)

            canonical_idx = _canonical_index_for_category(
                predicted_name,
                cat_in_each_index_original_mapper,
                cat_name_in_each_index,
            )
            s_predicted = s_fused[canonical_idx]
            indices = [
                _canonical_index_for_category(c_name, cat_in_each_index_original_mapper, cat_name_in_each_index)
                for c_name in vocab
                if c_name != predicted_name
            ]
            s_mean = np.mean([s_fused[i] for i in indices])
            dist_cosine_inter = s_predicted - s_mean
            dist_cosine_intra = s_fused[max_indice] - s_predicted
            score = float(anot["score"])
            if args.replace_score_with_fused:
                score = max(0.0, score * (1.0 + float(s_fused[max_indice] - s_visual[visual_max_indice])))

            anot_new = {
                "category_id": predicted_class,
                "image_id": anot["image_id"],
                "bbox": anot["bbox"],
                "score": score,
                "dist_cosine_inter": float(dist_cosine_inter.item()),
                "dist_cosine_intra": float(dist_cosine_intra.item()),
                "score_rmdx": score,
                "rmdx_visual_similarity": float(s_visual[visual_max_indice].item()),
                "rmdx_material_score": float(s_material[max_indice].item()),
                "rmdx_reliability_gate": float(gate.item()),
                "rmdx_fused_similarity": float(s_fused[max_indice].item()),
                "rmdx_mode": args.rmdx_mode,
                "rmdx_lambda_material": float(args.lambda_material),
                "rmdx_lambda_reliability": float(args.lambda_reliability),
                "rmdx_original_category_id": visual_class,
                "rmdx_original_score": float(anot["score"]),
            }
            new_anots.append(anot_new)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(new_anots, f)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.branch == "known":
        main_known(args)
    elif args.branch == "unknown":
        pass
    else:
        raise Exception("Method not implemented")
