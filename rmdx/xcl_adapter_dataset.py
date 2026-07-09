"""Dataset helpers for Batch52 XCL-Adapter.

All records come from non-test DET-COMPASS support/prototype or calibration
splits.  Full-test annotations are not used for training or selection.
"""

from __future__ import annotations

import ast
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class CropRecord:
    crop_id: str
    dataset: str
    split_role: str
    image_path: str
    image_id: str
    source_annotation_id: str
    category_id: int
    category_name: str
    crop_box_xyxy: list[float]
    crop_box_xywh: list[float]
    crop_type: str
    iou_with_gt: float
    source_is_full_test: bool
    source_is_test: bool
    leakage_check: str
    low_sample_flag: bool
    notes: str


def parse_bbox(value: str | list[float]) -> list[float]:
    vals = ast.literal_eval(value) if isinstance(value, str) else value
    return [float(v) for v in vals]


def xywh_to_xyxy(bbox: list[float]) -> list[float]:
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def xyxy_to_xywh(box: list[float]) -> list[float]:
    x1, y1, x2, y2 = box
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def clip_box(box: list[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = box
    x1 = min(max(0.0, x1), float(width - 1))
    y1 = min(max(0.0, y1), float(height - 1))
    x2 = min(max(x1 + 1.0, x2), float(width))
    y2 = min(max(y1 + 1.0, y2), float(height))
    return [x1, y1, x2, y2]


def iou_xyxy(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(area_a + area_b - inter, 1e-9)


def jitter_box(gt: list[float], width: int, height: int, rng: random.Random, target: str) -> tuple[list[float], float]:
    x1, y1, x2, y2 = gt
    w, h = x2 - x1, y2 - y1
    if w <= 1 or h <= 1:
        return gt, 1.0
    if target == "positive":
        shift = 0.10
        scale = 0.15
        min_iou, max_iou = 0.5, 1.0
    else:
        shift = 0.35
        scale = 0.35
        min_iou, max_iou = 0.3, 0.5
    best = gt
    best_iou = 1.0
    for _ in range(50):
        cx = (x1 + x2) / 2 + rng.uniform(-shift, shift) * w
        cy = (y1 + y2) / 2 + rng.uniform(-shift, shift) * h
        nw = w * (1.0 + rng.uniform(-scale, scale))
        nh = h * (1.0 + rng.uniform(-scale, scale))
        cand = clip_box([cx - nw / 2, cy - nh / 2, cx + nw / 2, cy + nh / 2], width, height)
        score = iou_xyxy(gt, cand)
        if min_iou <= score <= max_iou:
            return cand, score
        if target == "positive" and score > best_iou:
            best, best_iou = cand, score
        elif target != "positive" and abs(score - 0.4) < abs(best_iou - 0.4):
            best, best_iou = cand, score
    return best, best_iou


def image_size(path: str) -> tuple[int, int]:
    with Image.open(path) as img:
        return img.width, img.height


def cap_rows_by_category(rows: list[dict[str, str]], cap: int, seed: int = 52) -> list[dict[str, str]]:
    rng = random.Random(seed)
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["category_id"])].append(row)
    selected: list[dict[str, str]] = []
    for _cid, items in grouped.items():
        items = list(items)
        rng.shuffle(items)
        selected.extend(items[:cap])
    return selected


def crop_image(record: dict[str, Any]):
    img = Image.open(record["image_path"]).convert("RGB")
    x1, y1, x2, y2 = [int(round(v)) for v in record["crop_box_xyxy"]]
    x1 = max(0, min(x1, img.width - 1))
    y1 = max(0, min(y1, img.height - 1))
    x2 = max(x1 + 1, min(x2, img.width))
    y2 = max(y1 + 1, min(y2, img.height))
    return img.crop((x1, y1, x2, y2))

