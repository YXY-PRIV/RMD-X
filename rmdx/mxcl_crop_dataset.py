"""Crop dataset helpers for Batch53 MXCL-RMD-X.

MXCL uses only non-test annotations/crops for adapter training and validation.
Full-test annotations are not used here except by downstream final AP
evaluation scripts.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from rmdx.xcl_adapter_dataset import (
    clip_box,
    crop_image,
    image_size,
    iou_xyxy,
    jitter_box,
    parse_bbox,
    xywh_to_xyxy,
    xyxy_to_xywh,
)


def safe_name(name: str) -> str:
    return " ".join(str(name).replace("_", " ").replace("-", " ").split())


def global_label(dataset: str, category_name: str) -> str:
    return f"{dataset}::{category_name}"


def read_json(path: str | Path) -> Any:
    with Path(path).open() as f:
        return json.load(f)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def coco_objects(
    *,
    dataset: str,
    annotation_path: str | Path,
    image_root: str | Path,
    source_note: str,
) -> list[dict[str, Any]]:
    data = read_json(annotation_path)
    cats = {int(c["id"]): str(c["name"]) for c in data.get("categories", [])}
    images = {int(im["id"]): im for im in data.get("images", [])}
    out: list[dict[str, Any]] = []
    image_root = Path(image_root)
    for ann in data.get("annotations", []):
        image = images.get(int(ann["image_id"]))
        if not image:
            continue
        image_path = image_root / str(image["file_name"])
        bbox = [float(v) for v in ann.get("bbox", [])]
        if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
            continue
        out.append(
            {
                "dataset": dataset,
                "image_path": str(image_path),
                "image_id": str(image["id"]),
                "source_annotation_id": str(ann.get("id", "")),
                "category_id": int(ann["category_id"]),
                "category_name": cats.get(int(ann["category_id"]), str(ann["category_id"])),
                "bbox_xywh": bbox,
                "source_note": source_note,
            }
        )
    return out


def hixray_objects(manifest_path: str | Path) -> list[dict[str, Any]]:
    import pandas as pd

    df = pd.read_csv(manifest_path)
    df = df[df["valid_object"].astype(str).str.lower().isin(["true", "1"])]
    out: list[dict[str, Any]] = []
    for i, row in df.iterrows():
        bbox = parse_bbox(row["bbox_xywh"])
        if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
            continue
        out.append(
            {
                "dataset": "HiXray",
                "image_path": str(row["image_path"]),
                "image_id": str(row["image_id_or_name"]),
                "source_annotation_id": f"{row.get('image_id_or_name', '')}:{row.get('object_index', i)}",
                "category_id": int(row["mapped_fulltest_category_id"]),
                "category_name": str(row["mapped_fulltest_category_name"]),
                "bbox_xywh": bbox,
                "source_note": "hixray_train_object_manifest_user_confirmed_mapping",
            }
        )
    return out


def stratified_split(objects: list[dict[str, Any]], *, val_ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for obj in objects:
        by_label[str(obj["category_name"])].append(obj)
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    for _label, rows in by_label.items():
        rows = list(rows)
        rng.shuffle(rows)
        n_val = max(1, int(round(len(rows) * val_ratio))) if len(rows) >= 10 else max(0, min(1, len(rows) - 1))
        val.extend(rows[:n_val])
        train.extend(rows[n_val:])
    return train, val


def cap_objects(objects: list[dict[str, Any]], *, per_label_cap: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for obj in objects:
        by_label[str(obj["category_name"])].append(obj)
    selected: list[dict[str, Any]] = []
    for _label, rows in by_label.items():
        rows = list(rows)
        rng.shuffle(rows)
        selected.extend(rows[:per_label_cap])
    return selected


def make_crop_record(
    obj: dict[str, Any],
    *,
    split_role: str,
    crop_type: str,
    crop_box_xyxy: list[float],
    gt_xyxy: list[float],
    iou: float,
    low_sample: bool,
    crop_index: int,
) -> dict[str, Any]:
    dataset = str(obj["dataset"])
    category_name = str(obj["category_name"])
    crop_id = f"{dataset}_{split_role}_{obj['source_annotation_id']}_{crop_type}_{crop_index}".replace("/", "_")
    return {
        "crop_id": crop_id,
        "dataset": dataset,
        "split_role": split_role,
        "image_path": obj["image_path"],
        "image_id": obj["image_id"],
        "source_annotation_id": obj["source_annotation_id"],
        "category_id": int(obj["category_id"]),
        "category_name": category_name,
        "global_label": global_label(dataset, category_name),
        "crop_box_xyxy": json.dumps([round(v, 3) for v in crop_box_xyxy]),
        "crop_box_xywh": json.dumps([round(v, 3) for v in xyxy_to_xywh(crop_box_xyxy)]),
        "crop_type": crop_type,
        "iou_with_gt": round(float(iou), 4),
        "source_is_full_test": False,
        "source_is_test": False,
        "leakage_check": "no_full_test_gt_used",
        "low_sample_flag": bool(low_sample),
        "notes": obj.get("source_note", "non_test_training_source"),
    }


def expand_objects_to_crops(
    objects: list[dict[str, Any]],
    *,
    split_role: str,
    seed: int,
    low_sample_threshold: int = 3,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    label_counts: dict[str, int] = defaultdict(int)
    for obj in objects:
        label_counts[global_label(str(obj["dataset"]), str(obj["category_name"]))] += 1
    records: list[dict[str, Any]] = []
    for obj in objects:
        image_path = Path(obj["image_path"])
        if not image_path.exists():
            continue
        try:
            width, height = image_size(str(image_path))
        except Exception:
            continue
        gt = clip_box(xywh_to_xyxy([float(v) for v in obj["bbox_xywh"]]), width, height)
        if iou_xyxy(gt, gt) <= 0:
            continue
        low = label_counts[global_label(str(obj["dataset"]), str(obj["category_name"]))] < low_sample_threshold
        records.append(make_crop_record(obj, split_role=split_role, crop_type="gt_positive", crop_box_xyxy=gt, gt_xyxy=gt, iou=1.0, low_sample=low, crop_index=0))
        if split_role == "train":
            for idx in range(2):
                box, score = jitter_box(gt, width, height, rng, "positive")
                records.append(make_crop_record(obj, split_role=split_role, crop_type="jitter_positive", crop_box_xyxy=box, gt_xyxy=gt, iou=score, low_sample=low, crop_index=idx))
            box, score = jitter_box(gt, width, height, rng, "hard_jitter")
            records.append(make_crop_record(obj, split_role=split_role, crop_type="hard_jitter", crop_box_xyxy=box, gt_xyxy=gt, iou=score, low_sample=low, crop_index=0))
    return records


__all__ = [
    "cap_objects",
    "coco_objects",
    "crop_image",
    "expand_objects_to_crops",
    "global_label",
    "hixray_objects",
    "safe_name",
    "stratified_split",
    "write_csv",
]
