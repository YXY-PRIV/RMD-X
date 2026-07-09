"""Analysis utilities for RMD-X result comparison."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def load_json(path: str | Path) -> Any:
    with Path(path).open() as f:
        return json.load(f)


def load_result_list(path: str | Path) -> list[dict[str, Any]]:
    data = load_json(path)
    if not isinstance(data, list):
        raise TypeError(f"{path} is not a COCO result list")
    return data


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def read_best_from_grid(csv_path: str | Path) -> dict[str, str]:
    rows = list(csv.DictReader(Path(csv_path).open()))
    if not rows:
        raise ValueError(f"{csv_path} has no rows")
    return max(rows, key=lambda row: float(row["AP"]))


def result_basic_stats(dets: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [float(det["score"]) for det in dets if "score" in det]
    by_image = Counter(int(det["image_id"]) for det in dets)
    by_category = Counter(int(det["category_id"]) for det in dets)
    return {
        "detection_count": len(dets),
        "image_count": len(by_image),
        "category_count": len(by_category),
        "score_range": {
            "min": min(scores) if scores else None,
            "max": max(scores) if scores else None,
            "mean": mean(scores) if scores else None,
            "std": pstdev(scores) if len(scores) > 1 else 0.0,
        },
        "detections_per_image": {
            "min": min(by_image.values()) if by_image else None,
            "max": max(by_image.values()) if by_image else None,
            "mean": mean(by_image.values()) if by_image else None,
            "top10": by_image.most_common(10),
        },
        "detections_per_category_top20": by_category.most_common(20),
    }


def validate_result_alignment(
    annotation: dict[str, Any], dets: list[dict[str, Any]]
) -> dict[str, Any]:
    image_ids = {int(img["id"]) for img in annotation.get("images", [])}
    category_ids = {int(cat["id"]) for cat in annotation.get("categories", [])}
    det_image_ids = {int(det["image_id"]) for det in dets}
    det_category_ids = {int(det["category_id"]) for det in dets}
    return {
        "missing_image_ids": sorted(det_image_ids - image_ids),
        "missing_category_ids": sorted(det_category_ids - category_ids),
        "ok": not (det_image_ids - image_ids) and not (det_category_ids - category_ids),
    }


def evaluate_per_category(annotation_path: str | Path, result_path: str | Path) -> dict[int, dict[str, float | None]]:
    coco_gt = COCO(str(annotation_path))
    coco_dt = coco_gt.loadRes(str(result_path))
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    precision = evaluator.eval["precision"]
    cat_ids = evaluator.params.catIds
    out: dict[int, dict[str, float | None]] = {}
    for k, category_id in enumerate(cat_ids):
        p_all = precision[:, :, k, 0, -1]
        valid = p_all[p_all > -1]
        p50 = precision[0, :, k, 0, -1]
        p50 = p50[p50 > -1]
        p75 = precision[5, :, k, 0, -1]
        p75 = p75[p75 > -1]
        out[int(category_id)] = {
            "AP": float(np.mean(valid)) if valid.size else None,
            "AP50": float(np.mean(p50)) if p50.size else None,
            "AP75": float(np.mean(p75)) if p75.size else None,
        }
    return out


def bbox_xywh_to_xyxy(bbox: list[float]) -> tuple[float, float, float, float]:
    x, y, w, h = [float(v) for v in bbox[:4]]
    return x, y, x + w, y + h


def bbox_iou_xywh(a: list[float], b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = bbox_xywh_to_xyxy(a)
    bx0, by0, bx1, by1 = bbox_xywh_to_xyxy(b)
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def group_by_image(items: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        grouped[int(item["image_id"])].append(item)
    return grouped


def greedy_match_image(
    gt_anns: list[dict[str, Any]],
    dets: list[dict[str, Any]],
    *,
    iou_thr: float = 0.5,
    max_dets: int = 100,
) -> dict[str, Any]:
    sorted_dets = sorted(dets, key=lambda det: float(det.get("score", 0.0)), reverse=True)[:max_dets]
    used_gt: set[int] = set()
    tp, fp = 0, 0
    tp_dets, fp_dets = [], []
    for det in sorted_dets:
        best_idx = None
        best_iou = 0.0
        for idx, gt in enumerate(gt_anns):
            if idx in used_gt:
                continue
            if int(gt["category_id"]) != int(det["category_id"]):
                continue
            iou = bbox_iou_xywh(gt["bbox"], det["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx is not None and best_iou >= iou_thr:
            used_gt.add(best_idx)
            tp += 1
            copied = dict(det)
            copied["_match_iou"] = best_iou
            tp_dets.append(copied)
        else:
            fp += 1
            copied = dict(det)
            copied["_best_iou"] = best_iou
            fp_dets.append(copied)
    fn = max(0, len(gt_anns) - tp)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "gt_count": len(gt_anns),
        "det_count_top": len(sorted_dets),
        "tp_dets": tp_dets,
        "fp_dets": fp_dets,
        "score_simple": tp - fn - 0.01 * fp,
    }


def image_level_comparison(
    annotation: dict[str, Any],
    baseline: list[dict[str, Any]],
    material: list[dict[str, Any]],
    fusion: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    gt_by_image = group_by_image(annotation.get("annotations", []))
    baseline_by_image = group_by_image(baseline)
    material_by_image = group_by_image(material)
    fusion_by_image = group_by_image(fusion)
    rows = []
    for image in annotation.get("images", []):
        image_id = int(image["id"])
        gt = gt_by_image.get(image_id, [])
        b = greedy_match_image(gt, baseline_by_image.get(image_id, []))
        m = greedy_match_image(gt, material_by_image.get(image_id, []))
        f = greedy_match_image(gt, fusion_by_image.get(image_id, []))
        rows.append(
            {
                "image_id": image_id,
                "file_name": image["file_name"],
                "gt_count": len(gt),
                "baseline_tp": b["tp"],
                "baseline_fp": b["fp"],
                "baseline_fn": b["fn"],
                "material_tp": m["tp"],
                "material_fp": m["fp"],
                "material_fn": m["fn"],
                "fusion_tp": f["tp"],
                "fusion_fp": f["fp"],
                "fusion_fn": f["fn"],
                "baseline_score_simple": b["score_simple"],
                "material_score_simple": m["score_simple"],
                "fusion_score_simple": f["score_simple"],
                "fusion_delta_vs_baseline": f["score_simple"] - b["score_simple"],
                "fusion_delta_vs_material": f["score_simple"] - m["score_simple"],
                "fp_reduction_vs_baseline": b["fp"] - f["fp"],
                "tp_delta_vs_baseline": f["tp"] - b["tp"],
            }
        )
    return rows


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    mx, my = mean(x), mean(y)
    vx = sum((v - mx) ** 2 for v in x)
    vy = sum((v - my) ** 2 for v in y)
    if vx <= 0 or vy <= 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(vx * vy)


def numeric_range(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "max": None, "mean": None, "std": None}
    return {
        "min": min(values),
        "max": max(values),
        "mean": mean(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
    }


def score_correlation_summary(fusion: list[dict[str, Any]]) -> dict[str, Any]:
    score = [float(det.get("rmdx_original_score", det.get("score", 0.0))) for det in fusion]
    score_rmdx = [float(det.get("score_rmdx", det.get("score", 0.0))) for det in fusion]
    material = [float(det.get("rmdx_material_score", 0.0)) for det in fusion]
    reliability = [float(det.get("rmdx_reliability", 0.0)) for det in fusion]
    delta = [b - a for a, b in zip(score, score_rmdx)]
    return {
        "count": len(fusion),
        "score_range": numeric_range(score),
        "score_rmdx_range": numeric_range(score_rmdx),
        "score_delta_range": numeric_range(delta),
        "material_score_range": numeric_range(material),
        "reliability_score_range": numeric_range(reliability),
        "pearson": {
            "score_vs_score_rmdx": pearson(score, score_rmdx),
            "material_vs_score_delta": pearson(material, delta),
            "reliability_vs_score_delta": pearson(reliability, delta),
            "material_vs_reliability": pearson(material, reliability),
        },
    }


def write_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
