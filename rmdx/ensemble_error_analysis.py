"""Per-class and count analysis for Batch55 fixed ensemble replay."""

from __future__ import annotations

import json
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def load_json(path: str | Path) -> Any:
    with Path(path).open() as f:
        return json.load(f)


def per_class_ap(gt_path: str | Path, dt_path: str | Path) -> dict[int, float]:
    coco_gt = COCO(str(gt_path))
    coco_dt = coco_gt.loadRes(str(dt_path))
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    precision = evaluator.eval["precision"]  # T,R,K,A,M
    cat_ids = evaluator.params.catIds
    out = {}
    for k, cat_id in enumerate(cat_ids):
        vals = precision[:, :, k, 0, -1]
        vals = vals[vals > -1]
        out[int(cat_id)] = float(np.mean(vals)) if vals.size else float("nan")
    return out


def detection_counts(dets: list[dict[str, Any]]) -> tuple[Counter, Counter]:
    by_image: Counter = Counter()
    by_class: Counter = Counter()
    for det in dets:
        by_image[int(det["image_id"])] += 1
        by_class[int(det["category_id"])] += 1
    return by_image, by_class


def source_contribution(dets: list[dict[str, Any]]) -> Counter:
    counts: Counter = Counter()
    for det in dets:
        src = det.get("ensemble_input_detectors") or det.get("detector") or []
        if isinstance(src, list):
            for s in src:
                counts[str(s)] += 1
        else:
            counts[str(src)] += 1
    return counts
