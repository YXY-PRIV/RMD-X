"""Training-table helpers for Calib-VTC.

The helpers here intentionally do not read full_test GT.  They are used only
when a non-test calibration proposal cache is available.
"""

from __future__ import annotations

from typing import Sequence


def xywh_to_xyxy(bbox: Sequence[float]) -> tuple[float, float, float, float]:
    x, y, w, h = [float(v) for v in bbox]
    return x, y, x + max(0.0, w), y + max(0.0, h)


def bbox_iou_xywh(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = xywh_to_xyxy(a)
    bx1, by1, bx2, by2 = xywh_to_xyxy(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return 0.0 if denom <= 0 else inter / denom


def zscore(values: list[float], eps: float = 1e-6) -> list[float]:
    if not values:
        return []
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    std = (var ** 0.5) + eps
    return [(v - mean) / std for v in values]

