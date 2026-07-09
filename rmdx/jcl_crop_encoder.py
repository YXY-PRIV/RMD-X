"""Lightweight crop feature encoder for JCL-RMD-X."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from rmdx.material import extract_colour_feature_from_array


def xywh_to_bounds(bbox: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    x, y, w, h = [float(v) for v in bbox[:4]]
    x0 = max(0, min(width - 1, int(math.floor(x))))
    y0 = max(0, min(height - 1, int(math.floor(y))))
    x1 = max(x0 + 1, min(width, int(math.ceil(x + max(w, 1.0)))))
    y1 = max(y0 + 1, min(height, int(math.ceil(y + max(h, 1.0)))))
    return x0, y0, x1, y1


def resize_gray_feature(crop_rgb: np.ndarray, size: int = 8) -> list[float]:
    img = Image.fromarray(crop_rgb.astype(np.uint8)).convert("L").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return [float(v) for v in arr.reshape(-1)]


def geometry_feature(bbox_xywh: list[float], width: int, height: int) -> list[float]:
    x, y, w, h = [float(v) for v in bbox_xywh[:4]]
    area = max(w, 1.0) * max(h, 1.0)
    img_area = max(float(width * height), 1.0)
    return [
        x / max(width, 1),
        y / max(height, 1),
        w / max(width, 1),
        h / max(height, 1),
        area / img_area,
        w / max(h, 1.0),
    ]


def encode_crop(image_path: str | Path, bbox_xywh: list[float]) -> dict[str, Any]:
    image = Image.open(image_path).convert("RGB")
    arr = np.asarray(image, dtype=np.uint8)
    height, width = arr.shape[:2]
    x0, y0, x1, y1 = xywh_to_bounds(bbox_xywh, width, height)
    crop = arr[y0:y1, x0:x1]
    if crop.size == 0:
        crop = arr[max(0, y0):max(1, y0 + 1), max(0, x0):max(1, x0 + 1)]
    visual_embedding = resize_gray_feature(crop, 8)
    material_feature = extract_colour_feature_from_array(arr, bbox_xywh)
    geom = geometry_feature(bbox_xywh, width, height)
    return {
        "visual_embedding": visual_embedding,
        "material_feature_22d": material_feature,
        "geometry_feature": geom,
        "feature_vector": visual_embedding + material_feature + geom,
        "feature_dim": len(visual_embedding) + len(material_feature) + len(geom),
        "extractor_type": "lightweight_crop_statistics",
        "status": "OK",
    }


def normalize_matrix(train: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std < 1e-6] = 1.0
    return (values - mean) / std, mean, std
