"""Class-aware NMS and Soft-NMS helpers for detector ensemble."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def xywh_to_xyxy(bbox: list[float]) -> list[float]:
    x, y, w, h = [float(v) for v in bbox]
    return [x, y, x + w, y + h]


def xyxy_to_xywh(box: list[float]) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def iou_xywh(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = xywh_to_xyxy(a)
    bx1, by1, bx2, by2 = xywh_to_xyxy(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(area_a + area_b - inter, 1e-9)


def group_by_image_category(dets: list[dict[str, Any]]) -> dict[tuple[int, int], list[dict[str, Any]]]:
    groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for det in dets:
        groups[(int(det["image_id"]), int(det["category_id"]))].append(det)
    return groups


def limit_per_image(dets: list[dict[str, Any]], max_dets_per_image: int) -> list[dict[str, Any]]:
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for det in dets:
        by_image[int(det["image_id"])].append(det)
    out: list[dict[str, Any]] = []
    for rows in by_image.values():
        out.extend(sorted(rows, key=lambda r: float(r.get("score", 0.0)), reverse=True)[:max_dets_per_image])
    return out


def class_aware_nms(dets: list[dict[str, Any]], *, iou_thr: float, score_thr: float, max_dets_per_image: int) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for rows in group_by_image_category([d for d in dets if float(d.get("score", 0.0)) >= score_thr]).values():
        pending = sorted(rows, key=lambda r: float(r.get("score", 0.0)), reverse=True)
        while pending:
            best = pending.pop(0)
            kept.append(best)
            pending = [r for r in pending if iou_xywh(best["bbox"], r["bbox"]) < iou_thr]
    return limit_per_image(kept, max_dets_per_image)


def soft_nms(dets: list[dict[str, Any]], *, iou_thr: float, sigma: float, score_thr: float, max_dets_per_image: int, method: str = "linear") -> list[dict[str, Any]]:
    import math

    kept: list[dict[str, Any]] = []
    for rows in group_by_image_category([dict(d) for d in dets]).values():
        pending = rows
        while pending:
            pending.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)
            best = pending.pop(0)
            if float(best.get("score", 0.0)) >= score_thr:
                kept.append(best)
            new_pending = []
            for row in pending:
                ov = iou_xywh(best["bbox"], row["bbox"])
                if method == "gaussian":
                    row["score"] = float(row.get("score", 0.0)) * math.exp(-(ov * ov) / max(sigma, 1e-9))
                elif ov >= iou_thr:
                    row["score"] = float(row.get("score", 0.0)) * (1.0 - ov)
                if float(row.get("score", 0.0)) >= score_thr:
                    new_pending.append(row)
            pending = new_pending
    return limit_per_image(kept, max_dets_per_image)
