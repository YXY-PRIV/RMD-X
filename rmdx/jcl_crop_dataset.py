"""JCL-RMD-X crop dataset helpers.

All helpers here operate on non-test manifests or inference-time detector
proposals.  They do not read full_test GT boxes for training or model
selection.
"""

from __future__ import annotations

import ast
import hashlib
import math
from pathlib import Path
from typing import Any

from PIL import Image


def parse_bbox(value: str | list[float]) -> list[float]:
    if isinstance(value, list):
        return [float(v) for v in value[:4]]
    parsed = ast.literal_eval(value)
    return [float(v) for v in parsed[:4]]


def xywh_to_xyxy(box: list[float]) -> list[float]:
    x, y, w, h = [float(v) for v in box[:4]]
    return [x, y, x + max(w, 1.0), y + max(h, 1.0)]


def xyxy_to_xywh(box: list[float]) -> list[float]:
    x0, y0, x1, y1 = [float(v) for v in box[:4]]
    return [x0, y0, max(1.0, x1 - x0), max(1.0, y1 - y0)]


def clip_xyxy(box: list[float], width: int, height: int) -> list[float]:
    x0, y0, x1, y1 = [float(v) for v in box[:4]]
    x0 = max(0.0, min(float(width - 1), x0))
    y0 = max(0.0, min(float(height - 1), y0))
    x1 = max(x0 + 1.0, min(float(width), x1))
    y1 = max(y0 + 1.0, min(float(height), y1))
    return [x0, y0, x1, y1]


def box_iou_xyxy(a: list[float], b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    denom = area_a + area_b - inter
    return 0.0 if denom <= 0 else inter / denom


def stable_float(seed_text: str, low: float, high: float) -> float:
    digest = hashlib.sha1(seed_text.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    return low + (high - low) * value


def jitter_xyxy(gt_xyxy: list[float], width: int, height: int, key: str, strength: float) -> list[float]:
    x0, y0, x1, y1 = gt_xyxy
    w, h = max(1.0, x1 - x0), max(1.0, y1 - y0)
    cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
    dx = stable_float(key + ":dx", -strength, strength) * w
    dy = stable_float(key + ":dy", -strength, strength) * h
    sw = math.exp(stable_float(key + ":sw", -strength, strength))
    sh = math.exp(stable_float(key + ":sh", -strength, strength))
    nw, nh = w * sw, h * sh
    out = [cx + dx - nw * 0.5, cy + dy - nh * 0.5, cx + dx + nw * 0.5, cy + dy + nh * 0.5]
    return clip_xyxy(out, width, height)


def background_xyxy(gt_xyxy: list[float], width: int, height: int, key: str) -> list[float]:
    gx0, gy0, gx1, gy1 = gt_xyxy
    gw, gh = max(16.0, gx1 - gx0), max(16.0, gy1 - gy0)
    for attempt in range(24):
        scale = stable_float(f"{key}:scale:{attempt}", 0.6, 1.4)
        bw, bh = min(float(width), gw * scale), min(float(height), gh * scale)
        x0 = stable_float(f"{key}:x:{attempt}", 0.0, max(1.0, width - bw))
        y0 = stable_float(f"{key}:y:{attempt}", 0.0, max(1.0, height - bh))
        box = clip_xyxy([x0, y0, x0 + bw, y0 + bh], width, height)
        if box_iou_xyxy(box, gt_xyxy) < 0.2:
            return box
    return clip_xyxy([0.0, 0.0, min(gw, width), min(gh, height)], width, height)


def image_size(path: str | Path) -> tuple[int, int]:
    with Image.open(path) as img:
        return img.size


def make_crop_records(source: dict[str, Any], *, crop_id_prefix: str) -> list[dict[str, Any]]:
    image_path = Path(str(source["image_path"]))
    width, height = image_size(image_path)
    gt_xywh = parse_bbox(source["bbox"])
    gt_xyxy = clip_xyxy(xywh_to_xyxy(gt_xywh), width, height)
    base = {
        "dataset": source.get("dataset", "DET-COMPASS"),
        "split_role": "train" if source["split_role"] == "bank_train" else "val",
        "image_path": str(image_path),
        "image_id": source.get("image_id", ""),
        "source_annotation_id": source.get("annotation_id", ""),
        "category_id": int(source["category_id"]),
        "category_name": source.get("category_name", ""),
        "source_is_full_test": "False",
        "source_is_test": "False",
        "leakage_check": "no_full_test_gt_used",
    }
    records: list[dict[str, Any]] = []

    def add(crop_type: str, box_xyxy: list[float], idx: int) -> None:
        iou = box_iou_xyxy(box_xyxy, gt_xyxy)
        records.append(
            {
                **base,
                "crop_id": f"{crop_id_prefix}_{crop_type}_{idx}",
                "crop_box_xyxy": [round(float(v), 3) for v in box_xyxy],
                "crop_box_xywh": [round(float(v), 3) for v in xyxy_to_xywh(box_xyxy)],
                "crop_type": crop_type,
                "iou_with_gt": round(float(iou), 6),
                "notes": "jittered from non-test support/prototype split",
            }
        )

    add("gt_positive", gt_xyxy, 0)
    for idx in range(3):
        add("jitter_positive", jitter_xyxy(gt_xyxy, width, height, f"{crop_id_prefix}:jp:{idx}", 0.18), idx)
    for idx in range(2):
        add("hard_jitter", jitter_xyxy(gt_xyxy, width, height, f"{crop_id_prefix}:hj:{idx}", 0.55), idx)
    for idx in range(1):
        add("background", background_xyxy(gt_xyxy, width, height, f"{crop_id_prefix}:bg:{idx}"), idx)
    return records
