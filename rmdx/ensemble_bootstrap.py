"""COCO bootstrap utilities for fixed detector ensemble analysis."""

from __future__ import annotations

import copy
import json
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def coco_ap(gt: dict[str, Any], dt: list[dict[str, Any]]) -> float:
    with tempfile.TemporaryDirectory() as tmp:
        gt_path = Path(tmp) / "gt.json"
        dt_path = Path(tmp) / "dt.json"
        gt_path.write_text(json.dumps(gt), encoding="utf-8")
        dt_path.write_text(json.dumps(dt), encoding="utf-8")
        coco_gt = COCO(str(gt_path))
        coco_dt = coco_gt.loadRes(str(dt_path))
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        precision = evaluator.eval["precision"][:, :, :, 0, -1]
        valid = precision[precision > -1]
        return float(np.mean(valid)) if valid.size else float("nan")


def build_bootstrap_subset(gt_full: dict[str, Any], fixed_dt: list[dict[str, Any]], ensemble_dt: list[dict[str, Any]], sampled_image_ids: list[int]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    images_by_id = {int(im["id"]): im for im in gt_full["images"]}
    anns_by_img: dict[int, list[dict[str, Any]]] = {}
    for ann in gt_full["annotations"]:
        anns_by_img.setdefault(int(ann["image_id"]), []).append(ann)
    fixed_by_img: dict[int, list[dict[str, Any]]] = {}
    ens_by_img: dict[int, list[dict[str, Any]]] = {}
    for det in fixed_dt:
        fixed_by_img.setdefault(int(det["image_id"]), []).append(det)
    for det in ensemble_dt:
        ens_by_img.setdefault(int(det["image_id"]), []).append(det)

    new_images = []
    new_anns = []
    new_fixed = []
    new_ens = []
    ann_id = 1
    for new_img_id, old_img_id in enumerate(sampled_image_ids, start=1):
        image = copy.deepcopy(images_by_id[int(old_img_id)])
        image["id"] = new_img_id
        new_images.append(image)
        for ann in anns_by_img.get(int(old_img_id), []):
            row = copy.deepcopy(ann)
            row["id"] = ann_id
            row["image_id"] = new_img_id
            ann_id += 1
            new_anns.append(row)
        for det in fixed_by_img.get(int(old_img_id), []):
            row = {k: copy.deepcopy(v) for k, v in det.items() if k in {"image_id", "category_id", "bbox", "score"}}
            row["image_id"] = new_img_id
            new_fixed.append(row)
        for det in ens_by_img.get(int(old_img_id), []):
            row = {k: copy.deepcopy(v) for k, v in det.items() if k in {"image_id", "category_id", "bbox", "score"}}
            row["image_id"] = new_img_id
            new_ens.append(row)
    gt = {k: copy.deepcopy(v) for k, v in gt_full.items() if k not in {"images", "annotations"}}
    gt["images"] = new_images
    gt["annotations"] = new_anns
    return gt, new_fixed, new_ens


def bootstrap_delta(gt_full: dict[str, Any], fixed_dt: list[dict[str, Any]], ensemble_dt: list[dict[str, Any]], *, iterations: int = 100, seed: int = 55) -> list[dict[str, float]]:
    rng = random.Random(seed)
    image_ids = [int(im["id"]) for im in gt_full["images"]]
    rows = []
    for i in range(iterations):
        sampled = [rng.choice(image_ids) for _ in image_ids]
        gt, fixed_sub, ens_sub = build_bootstrap_subset(gt_full, fixed_dt, ensemble_dt, sampled)
        fixed_ap = coco_ap(gt, fixed_sub)
        ensemble_ap = coco_ap(gt, ens_sub)
        rows.append({"iteration": i, "fixed_AP": fixed_ap, "ensemble_AP": ensemble_ap, "delta_AP": ensemble_ap - fixed_ap})
    return rows
