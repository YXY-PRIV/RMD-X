"""Colour-only material feature and scoring helpers for RMD-X.

This module is an external post-processing MVP. It uses only colour X-ray
images plus COCO annotations/results and does not modify RAXO internals.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


FEATURE_NAMES = [
    "rgb_mean_r",
    "rgb_mean_g",
    "rgb_mean_b",
    "rgb_std_r",
    "rgb_std_g",
    "rgb_std_b",
    "hsv_mean_h",
    "hsv_mean_s",
    "hsv_mean_v",
    "hsv_std_h",
    "hsv_std_s",
    "hsv_std_v",
    "gray_mean",
    "gray_std",
    "gray_q10",
    "gray_q25",
    "gray_q50",
    "gray_q75",
    "gray_q90",
    "edge_density",
    "area_ratio",
    "aspect_ratio",
]


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def xywh_to_int_bounds(bbox: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    x, y, w, h = [safe_float(v) for v in bbox[:4]]
    x0 = max(0, min(width, int(math.floor(x))))
    y0 = max(0, min(height, int(math.floor(y))))
    x1 = max(x0 + 1, min(width, int(math.ceil(x + max(w, 1.0)))))
    y1 = max(y0 + 1, min(height, int(math.ceil(y + max(h, 1.0)))))
    return x0, y0, x1, y1


def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    rgb = np.clip(rgb, 0.0, 1.0)
    maxc = rgb.max(axis=-1)
    minc = rgb.min(axis=-1)
    delta = maxc - minc

    h = np.zeros_like(maxc)
    nonzero = delta > 1e-12
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mask = nonzero & (maxc == r)
    h[mask] = ((g[mask] - b[mask]) / delta[mask]) % 6.0
    mask = nonzero & (maxc == g)
    h[mask] = ((b[mask] - r[mask]) / delta[mask]) + 2.0
    mask = nonzero & (maxc == b)
    h[mask] = ((r[mask] - g[mask]) / delta[mask]) + 4.0
    h /= 6.0

    s = np.zeros_like(maxc)
    max_nonzero = maxc > 1e-12
    s[max_nonzero] = delta[max_nonzero] / maxc[max_nonzero]
    v = maxc
    return np.stack([h, s, v], axis=-1)


def _edge_density(gray: np.ndarray) -> float:
    if gray.size <= 1:
        return 0.0
    gx = np.abs(np.diff(gray, axis=1)) if gray.shape[1] > 1 else np.empty((0,))
    gy = np.abs(np.diff(gray, axis=0)) if gray.shape[0] > 1 else np.empty((0,))
    values = []
    if gx.size:
        values.append(gx.reshape(-1))
    if gy.size:
        values.append(gy.reshape(-1))
    if not values:
        return 0.0
    grad = np.concatenate(values)
    return float(np.mean(grad > 0.08))


def decode_mask(mask_obj: Any, width: int, height: int) -> np.ndarray | None:
    if mask_obj is None:
        return None
    try:
        from pycocotools import mask as mask_utils

        decoded = mask_utils.decode(mask_obj)
        if decoded.ndim == 3:
            decoded = decoded[:, :, 0]
        if decoded.shape[:2] != (height, width):
            return None
        return decoded.astype(bool)
    except Exception:
        return None


def _sample_pixels(pixels: np.ndarray, max_pixels: int = 8192) -> np.ndarray:
    if len(pixels) <= max_pixels:
        return pixels
    stride = max(1, int(math.ceil(len(pixels) / max_pixels)))
    return pixels[::stride][:max_pixels]


def _sample_gray_crop(gray: np.ndarray, max_side: int = 256) -> np.ndarray:
    if gray.shape[0] <= max_side and gray.shape[1] <= max_side:
        return gray
    stride = max(1, int(math.ceil(max(gray.shape[:2]) / max_side)))
    return gray[::stride, ::stride]


def extract_colour_feature_from_array(
    rgb_array_uint8: np.ndarray,
    bbox: list[float],
    *,
    mode: str = "bbox_only",
    mask: np.ndarray | None = None,
) -> list[float]:
    height, width = rgb_array_uint8.shape[:2]
    x0, y0, x1, y1 = xywh_to_int_bounds(bbox, width, height)
    crop = rgb_array_uint8[y0:y1, x0:x1].astype(np.float32) / 255.0

    pixels = crop.reshape(-1, 3)
    if mode == "mask_weighted" and mask is not None:
        crop_mask = mask[y0:y1, x0:x1]
        if crop_mask.shape[:2] == crop.shape[:2] and np.any(crop_mask):
            pixels = crop[crop_mask.astype(bool)]

    if pixels.size == 0:
        pixels = crop.reshape(-1, 3)
    pixels = _sample_pixels(pixels)

    rgb_mean = pixels.mean(axis=0)
    rgb_std = pixels.std(axis=0)
    hsv = _rgb_to_hsv(pixels.reshape(-1, 1, 3)).reshape(-1, 3)
    hsv_mean = hsv.mean(axis=0)
    hsv_std = hsv.std(axis=0)
    gray_pixels = (0.299 * pixels[:, 0]) + (0.587 * pixels[:, 1]) + (0.114 * pixels[:, 2])
    gray_mean = float(gray_pixels.mean())
    gray_std = float(gray_pixels.std())
    quantiles = np.quantile(gray_pixels, [0.10, 0.25, 0.50, 0.75, 0.90])
    gray_crop = (0.299 * crop[:, :, 0]) + (0.587 * crop[:, :, 1]) + (0.114 * crop[:, :, 2])
    gray_crop = _sample_gray_crop(gray_crop)
    area_ratio = float(((x1 - x0) * (y1 - y0)) / max(width * height, 1))
    aspect_ratio = float((x1 - x0) / max(y1 - y0, 1))

    feature = np.concatenate(
        [
            rgb_mean,
            rgb_std,
            hsv_mean,
            hsv_std,
            np.array([gray_mean, gray_std]),
            quantiles,
            np.array([_edge_density(gray_crop), area_ratio, aspect_ratio]),
        ]
    )
    return [float(v) for v in feature]


def extract_colour_feature_from_image(
    image: Image.Image,
    bbox: list[float],
    *,
    mode: str = "bbox_only",
    mask: np.ndarray | None = None,
) -> list[float]:
    rgb_array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return extract_colour_feature_from_array(rgb_array, bbox, mode=mode, mask=mask)


def compute_z_stats(features: list[list[float]]) -> dict[str, list[float]]:
    arr = np.asarray(features, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"Expected feature matrix with {len(FEATURE_NAMES)} columns")
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    std[std == 0] = 1.0
    return {"mean": mean.tolist(), "std": std.tolist()}


def zscore_feature(feature: list[float], stats: dict[str, list[float]]) -> np.ndarray:
    arr = np.asarray(feature, dtype=np.float64)
    mean = np.asarray(stats["mean"], dtype=np.float64)
    std = np.asarray(stats["std"], dtype=np.float64)
    std = np.where(std == 0, 1.0, std)
    return (arr - mean) / std


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def negative_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(-np.linalg.norm(a - b))


def build_material_bank(
    annotation: dict[str, Any],
    image_dir: str | Path,
    *,
    mode: str = "bbox_only",
    target_annotation: dict[str, Any] | None = None,
    bank_name: str = "material_colour_bank",
) -> tuple[dict[str, Any], dict[str, Any]]:
    image_dir = Path(image_dir)
    images = {int(img["id"]): img for img in annotation.get("images", [])}
    categories = {int(cat["id"]): cat for cat in annotation.get("categories", [])}
    if target_annotation is None:
        source_to_eval = {category_id: category_id for category_id in categories}
        target_categories = categories
        unmapped_source_categories = []
    else:
        target_by_name = {
            str(cat.get("name")): int(cat["id"])
            for cat in target_annotation.get("categories", [])
        }
        target_categories = {
            int(cat["id"]): cat for cat in target_annotation.get("categories", [])
        }
        source_to_eval = {}
        unmapped_source_categories = []
        for category_id, cat in categories.items():
            eval_id = target_by_name.get(str(cat.get("name")))
            if eval_id is None:
                unmapped_source_categories.append(
                    {"source_category_id": category_id, "category_name": cat.get("name")}
                )
            else:
                source_to_eval[category_id] = eval_id

    grouped: dict[int, list[list[float]]] = defaultdict(list)
    source_ids_by_eval: dict[int, set[int]] = defaultdict(set)
    missing_images: Counter[str] = Counter()
    skipped = 0
    skipped_unmapped = 0
    image_cache: dict[int, np.ndarray] = {}

    for ann in annotation.get("annotations", []):
        image_id = int(ann["image_id"])
        source_category_id = int(ann["category_id"])
        category_id = source_to_eval.get(source_category_id)
        if category_id is None:
            skipped_unmapped += 1
            skipped += 1
            continue
        image_info = images.get(image_id)
        if not image_info:
            skipped += 1
            continue
        image_path = image_dir / image_info["file_name"]
        if not image_path.exists():
            missing_images[str(image_path)] += 1
            skipped += 1
            continue
        if image_id not in image_cache:
            image_cache[image_id] = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        grouped[category_id].append(
            extract_colour_feature_from_array(
                image_cache[image_id],
                ann["bbox"],
                mode=mode,
            )
        )
        source_ids_by_eval[category_id].add(source_category_id)

    prototypes = {}
    all_means = []
    sample_counts = []
    for category_id, features in sorted(grouped.items()):
        arr = np.asarray(features, dtype=np.float64)
        mean_feature = arr.mean(axis=0).tolist()
        std_feature = arr.std(axis=0).tolist()
        cat = target_categories.get(category_id, categories.get(category_id, {}))
        source_ids = sorted(source_ids_by_eval.get(category_id, {category_id}))
        prototypes[str(category_id)] = {
            "category_id": category_id,
            "category_id_for_eval": category_id,
            "category_name": cat.get("name"),
            "source_category_id": source_ids[0] if len(source_ids) == 1 else None,
            "source_category_ids": source_ids,
            "sample_count": len(features),
            "mean_feature": [float(v) for v in mean_feature],
            "std_feature": [float(v) for v in std_feature],
            "feature_names": FEATURE_NAMES,
        }
        all_means.append(mean_feature)
        sample_counts.append(len(features))

    bank_stats = compute_z_stats(all_means) if all_means else {"mean": [], "std": []}
    bank = {
        "version": "material_colour_only_mvp_v1",
        "bank_name": bank_name,
        "mode": mode,
        "feature_names": FEATURE_NAMES,
        "prototype_count": len(prototypes),
        "prototypes": prototypes,
        "bank_feature_stats": bank_stats,
        "category_mapping": {
            str(source_id): eval_id for source_id, eval_id in sorted(source_to_eval.items())
        },
        "unmapped_source_categories": unmapped_source_categories,
        "source_note": (
            "Built from GT boxes in the requested annotation. If this is support/prototype "
            "data, it is intended to reduce test-set leakage compared with full_test."
        ),
    }
    summary = {
        "bank_name": bank_name,
        "mode": mode,
        "category_count": len(prototypes),
        "source_category_count": len(categories),
        "target_category_count": len(target_categories),
        "mapped_source_category_count": len(source_to_eval),
        "unmapped_source_category_count": len(unmapped_source_categories),
        "gt_annotation_count": len(annotation.get("annotations", [])),
        "used_annotation_count": int(sum(sample_counts)),
        "skipped_annotation_count": skipped,
        "skipped_unmapped_annotation_count": skipped_unmapped,
        "missing_images": dict(missing_images),
        "feature_names": FEATURE_NAMES,
        "sample_count_distribution": {
            "min": min(sample_counts) if sample_counts else None,
            "max": max(sample_counts) if sample_counts else None,
            "mean": (sum(sample_counts) / len(sample_counts)) if sample_counts else None,
            "histogram": dict(Counter(sample_counts)),
        },
        "unmapped_source_categories_top20": unmapped_source_categories[:20],
        "source_note": bank["source_note"],
    }
    return bank, summary


def load_material_bank(path: str | Path) -> dict[str, Any]:
    import json

    with Path(path).open() as f:
        bank = json.load(f)
    if "prototypes" not in bank:
        raise KeyError(f"{path} missing prototypes")
    return bank


def score_detections_with_material(
    dets: list[dict[str, Any]],
    annotation: dict[str, Any],
    image_dir: str | Path,
    material_bank: dict[str, Any],
    *,
    mode: str = "bbox_only",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    image_dir = Path(image_dir)
    images = {int(img["id"]): img for img in annotation.get("images", [])}
    prototypes = material_bank["prototypes"]
    image_cache: dict[int, np.ndarray] = {}
    scored_base = []
    detection_features: list[list[float]] = []
    prototype_features = [proto["mean_feature"] for proto in prototypes.values()]

    for idx, det in enumerate(dets):
        image_id = int(det["image_id"])
        image_info = images.get(image_id)
        if not image_info:
            raise KeyError(f"Detection {idx} references missing image_id {image_id}")
        image_path = image_dir / image_info["file_name"]
        if image_id not in image_cache:
            image_cache[image_id] = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        image = image_cache[image_id]
        mask = None
        if mode == "mask_weighted":
            height, width = image.shape[:2]
            mask = decode_mask(det.get("mask") or det.get("segmentation"), width, height)
        feature = extract_colour_feature_from_array(
            image,
            det["bbox"],
            mode=mode,
            mask=mask,
        )
        copied = dict(det)
        copied["_rmdx_material_feature"] = feature
        scored_base.append(copied)
        detection_features.append(feature)

    combined_stats = compute_z_stats(prototype_features + detection_features)
    raw_values = []
    neg_l2_values = []
    for det in scored_base:
        proto = prototypes.get(str(int(det["category_id"])))
        if not proto:
            raw_values.append(0.0)
            neg_l2_values.append(float("-inf"))
            continue
        z_det = zscore_feature(det["_rmdx_material_feature"], combined_stats)
        z_proto = zscore_feature(proto["mean_feature"], combined_stats)
        raw_values.append(cosine_similarity(z_det, z_proto))
        neg_l2_values.append(negative_l2(z_det, z_proto))

    output = []
    for det, raw, neg in zip(scored_base, raw_values, neg_l2_values):
        out = dict(det)
        out.pop("_rmdx_material_feature", None)
        original_score = safe_float(out.get("score"))
        material_score = max(0.0, min(1.0, (raw + 1.0) / 2.0))
        out["rmdx_material_score"] = material_score
        out["rmdx_material_similarity_raw"] = raw
        out["rmdx_material_negative_l2"] = neg
        out["rmdx_material_mode"] = mode
        out["rmdx_original_score"] = original_score
        out["score_rmdx"] = original_score
        output.append(out)

    scores = [safe_float(det["rmdx_material_score"]) for det in output]
    summary = {
        "input_count": len(dets),
        "output_count": len(output),
        "mode": mode,
        "prototype_count": len(prototypes),
        "image_count": len({int(det["image_id"]) for det in dets}),
        "material_score_range": {
            "min": min(scores) if scores else None,
            "max": max(scores) if scores else None,
            "mean": (sum(scores) / len(scores)) if scores else None,
        },
    }
    return output, summary


def apply_material_append_only(dets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for det in dets:
        out = dict(det)
        score = safe_float(out["score"])
        out["score_rmdx"] = score
        out["rmdx_material_strategy"] = "material_score_append_only"
        out["rmdx_original_score"] = score
        output.append(out)
    return output


def apply_material_rescore(
    dets: list[dict[str, Any]],
    lambda_material: float,
    *,
    replace_score: bool = True,
) -> list[dict[str, Any]]:
    material_scores = [safe_float(det["rmdx_material_score"]) for det in dets]
    mean_score = sum(material_scores) / len(material_scores) if material_scores else 0.0
    output = []
    for det in dets:
        out = dict(det)
        original_score = safe_float(out["score"])
        centered = safe_float(out["rmdx_material_score"]) - mean_score
        score_rmdx = original_score * (1.0 + float(lambda_material) * centered)
        score_rmdx = max(0.0, score_rmdx)
        out["score_rmdx"] = score_rmdx
        out["rmdx_material_strategy"] = "material_rescore"
        out["rmdx_lambda_material"] = float(lambda_material)
        out["rmdx_material_centered_score"] = centered
        out["rmdx_original_score"] = original_score
        if replace_score:
            out["score"] = score_rmdx
        output.append(out)
    return output


def apply_material_filter_and_rescore(
    dets: list[dict[str, Any]],
    threshold: float,
    lambda_material: float,
) -> list[dict[str, Any]]:
    filtered = [
        det for det in dets if safe_float(det.get("rmdx_material_score")) >= float(threshold)
    ]
    rescored = apply_material_rescore(filtered, lambda_material, replace_score=True)
    for det in rescored:
        det["rmdx_material_strategy"] = "material_filter_and_rescore"
        det["rmdx_material_threshold"] = float(threshold)
    return rescored
