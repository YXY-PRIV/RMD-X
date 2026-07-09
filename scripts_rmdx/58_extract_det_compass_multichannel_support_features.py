#!/usr/bin/env python3
"""Batch27: extract DET-COMPASS multi-channel support object features.

CPU-only. Uses support/prototype split metadata exported in Batch26. It does
not read full_test boxes and does not modify any RAXO files.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path("/workspace/RAXO")
if not ROOT.exists():
    ROOT = Path("/mnt/data/yxy/program/RAXO")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rmdx.material import FEATURE_NAMES as LEGACY_COLOUR_FEATURE_NAMES
from rmdx.material import extract_colour_feature_from_array


MC_DIR = ROOT / "outputs_rmdx" / "multichannel_det_compass"
OUT_ROOT = ROOT / "outputs_rmdx" / "material_multichannel_det_compass"
FEATURE_DIR = OUT_ROOT / "features"
DOC_DIR = ROOT / "docs_rmdx"
SUPPORT_MANIFEST = MC_DIR / "support_multichannel_manifest.csv"
SOURCE_DISCOVERY = MC_DIR / "source_discovery.json"
CHANNELS = ["colour", "grey", "high", "low", "density"]

GENERIC_FEATURE_NAMES = [
    "mean",
    "std",
    "min",
    "max",
    "median",
    "q10",
    "q25",
    "q75",
    "q90",
    "entropy",
    "nonzero_ratio",
    "bbox_area_ratio",
    "bbox_aspect_ratio",
    "crop_width",
    "crop_height",
    "edge_density",
    "gradient_mean",
    "gradient_std",
]

CROSS_FEATURE_NAMES = [
    "high_minus_low_mean",
    "high_low_ratio_mean",
    "density_minus_grey_mean",
    "colour_gray_minus_density_mean",
    "channel_mean_colour",
    "channel_mean_grey",
    "channel_mean_high",
    "channel_mean_low",
    "channel_mean_density",
    "channel_std_colour",
    "channel_std_grey",
    "channel_std_high",
    "channel_std_low",
    "channel_std_density",
    "channel_contrast_colour",
    "channel_contrast_grey",
    "channel_contrast_high",
    "channel_contrast_low",
    "channel_contrast_density",
    "high_low_contrast",
]


def dataset_root() -> Path:
    for root in [
        Path("/datasets/xray-datasets"),
        Path("/mnt/data/raxo/datasets/xray-datasets"),
        Path("/data/raxo/datasets/xray-datasets"),
    ]:
        if (root / "DET-COMPASS" / "annotations" / "full_test.json").exists():
            return root
    return Path("/mnt/data/raxo/datasets/xray-datasets")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def full_test_category_map() -> dict[str, int]:
    path = dataset_root() / "DET-COMPASS" / "annotations" / "full_test.json"
    with path.open("r", encoding="utf-8") as f:
        ann = json.load(f)
    out: dict[str, int] = {}
    for cat in ann.get("categories", []):
        name = str(cat["name"])
        out[name] = int(cat["id"])
        out[normalize_category_name(name)] = int(cat["id"])
    return out


def full_test_category_name_map() -> dict[str, str]:
    path = dataset_root() / "DET-COMPASS" / "annotations" / "full_test.json"
    with path.open("r", encoding="utf-8") as f:
        ann = json.load(f)
    out: dict[str, str] = {}
    for cat in ann.get("categories", []):
        name = str(cat["name"])
        out[name] = name
        out[normalize_category_name(name)] = name
    return out


def normalize_category_name(name: str) -> str:
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


def support_parquet_files_from_discovery() -> list[str]:
    data = read_json(SOURCE_DISCOVERY, {})
    files: list[str] = []
    for group in data.get("parquet_groups", []):
        if group.get("group_name") == "raxo_prototypes":
            files = [str(p) for p in group.get("files", [])]
            break
    if files:
        return files
    by_manifest = sorted({row["source_path"] for row in read_csv(SUPPORT_MANIFEST) if row.get("source_path")})
    return by_manifest


def manifest_by_row() -> dict[int, dict[str, Any]]:
    grouped: dict[int, dict[str, Any]] = defaultdict(lambda: {"channel_paths": {}, "file_name": "", "categories": "", "xray_bbox": ""})
    for row in read_csv(SUPPORT_MANIFEST):
        if row.get("status") not in {"exported", "reused_existing"}:
            continue
        idx = int(row["source_row_index"])
        channel = row["channel"]
        grouped[idx]["channel_paths"][channel] = row["target_path"]
        grouped[idx]["file_name"] = row["file_name"]
        grouped[idx]["categories"] = row.get("categories", "")
        grouped[idx]["xray_bbox"] = row.get("xray_bbox", "[]")
        grouped[idx]["source_path"] = row.get("source_path", "")
        grouped[idx]["width"] = row.get("width", "")
        grouped[idx]["height"] = row.get("height", "")
    return dict(grouped)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def normalize_bbox_xywh(bbox: list[Any], width: int, height: int) -> tuple[list[float] | None, str, str]:
    if len(bbox) < 4:
        return None, "unknown", "bbox length < 4"
    x, y, w, h = [safe_float(v) for v in bbox[:4]]
    # Batch26/27 stats show support/prototype xray_bbox is consistent with xywh.
    if w > 0 and h > 0 and x + w <= width + 5 and y + h <= height + 5:
        fmt = "xywh"
    elif w > x and h > y and w <= width + 5 and h <= height + 5:
        w, h = w - x, h - y
        fmt = "xyxy_converted_to_xywh"
    else:
        return None, "bbox_format_unknown", f"bbox={bbox} outside image {width}x{height}"
    x0 = max(0.0, min(float(width - 1), x))
    y0 = max(0.0, min(float(height - 1), y))
    x1 = max(x0 + 1.0, min(float(width), x0 + max(1.0, w)))
    y1 = max(y0 + 1.0, min(float(height), y0 + max(1.0, h)))
    out = [x0, y0, x1 - x0, y1 - y0]
    if out[2] <= 0 or out[3] <= 0:
        return None, fmt, "bbox area <= 0 after clipping"
    return out, fmt, ""


def int_bounds(bbox: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    x0 = max(0, min(width - 1, int(math.floor(x))))
    y0 = max(0, min(height - 1, int(math.floor(y))))
    x1 = max(x0 + 1, min(width, int(math.ceil(x + max(w, 1.0)))))
    y1 = max(y0 + 1, min(height, int(math.ceil(y + max(h, 1.0)))))
    return x0, y0, x1, y1


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
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
    return np.stack([h, s, maxc], axis=-1)


def entropy01(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    hist, _ = np.histogram(values, bins=32, range=(0.0, 1.0), density=False)
    prob = hist.astype(np.float64)
    prob = prob[prob > 0]
    prob /= prob.sum() if prob.sum() else 1.0
    return float(-(prob * np.log2(prob)).sum())


def gradient_stats(gray: np.ndarray) -> tuple[float, float, float]:
    if gray.size <= 1:
        return 0.0, 0.0, 0.0
    gx = np.diff(gray, axis=1) if gray.shape[1] > 1 else np.zeros_like(gray)
    gy = np.diff(gray, axis=0) if gray.shape[0] > 1 else np.zeros_like(gray)
    vals = []
    if gx.size:
        vals.append(np.abs(gx).reshape(-1))
    if gy.size:
        vals.append(np.abs(gy).reshape(-1))
    if not vals:
        return 0.0, 0.0, 0.0
    grad = np.concatenate(vals)
    return float(np.mean(grad > 0.08)), float(np.mean(grad)), float(np.std(grad))


def channel_generic_features(image: Image.Image, bbox: list[float]) -> tuple[dict[str, float], np.ndarray]:
    arr_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = arr_rgb.shape[:2]
    x0, y0, x1, y1 = int_bounds(bbox, width, height)
    crop = arr_rgb[y0:y1, x0:x1].astype(np.float32) / 255.0
    gray = (0.299 * crop[:, :, 0]) + (0.587 * crop[:, :, 1]) + (0.114 * crop[:, :, 2])
    vals = gray.reshape(-1)
    if vals.size == 0:
        vals = np.array([0.0], dtype=np.float32)
    edge_density, grad_mean, grad_std = gradient_stats(gray)
    area_ratio = float(((x1 - x0) * (y1 - y0)) / max(width * height, 1))
    aspect_ratio = float((x1 - x0) / max(y1 - y0, 1))
    qs = np.quantile(vals, [0.10, 0.25, 0.50, 0.75, 0.90])
    feat = {
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "median": float(qs[2]),
        "q10": float(qs[0]),
        "q25": float(qs[1]),
        "q75": float(qs[3]),
        "q90": float(qs[4]),
        "entropy": entropy01(vals),
        "nonzero_ratio": float(np.mean(vals > 1e-6)),
        "bbox_area_ratio": area_ratio,
        "bbox_aspect_ratio": aspect_ratio,
        "crop_width": float(x1 - x0),
        "crop_height": float(y1 - y0),
        "edge_density": edge_density,
        "gradient_mean": grad_mean,
        "gradient_std": grad_std,
    }
    return feat, gray


def colour_extra_features(image: Image.Image, bbox: list[float]) -> dict[str, float]:
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    legacy = extract_colour_feature_from_array(arr, bbox)
    out = {f"legacy_{name}": float(value) for name, value in zip(LEGACY_COLOUR_FEATURE_NAMES, legacy)}
    height, width = arr.shape[:2]
    x0, y0, x1, y1 = int_bounds(bbox, width, height)
    pixels = arr[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32) / 255.0
    hsv = rgb_to_hsv(pixels.reshape(-1, 1, 3)).reshape(-1, 3) if pixels.size else np.zeros((1, 3), dtype=np.float32)
    for idx, name in enumerate(["r", "g", "b"]):
        out[f"rgb_mean_{name}"] = float(pixels[:, idx].mean()) if pixels.size else 0.0
        out[f"rgb_std_{name}"] = float(pixels[:, idx].std()) if pixels.size else 0.0
    for idx, name in enumerate(["h", "s", "v"]):
        out[f"hsv_mean_{name}"] = float(hsv[:, idx].mean())
        out[f"hsv_std_{name}"] = float(hsv[:, idx].std())
    return out


def extract_object_features(row_meta: dict[str, Any], object_index: int, category_name: str, source_category_name: str, bbox: list[float], category_id: int | None) -> dict[str, Any]:
    channel_paths = row_meta["channel_paths"]
    images = {channel: Image.open(channel_paths[channel]) for channel in CHANNELS}
    try:
        width, height = images["colour"].size
        norm_bbox, bbox_format, bbox_note = normalize_bbox_xywh(bbox, width, height)
        if norm_bbox is None:
            raise ValueError(bbox_note)
        record: dict[str, Any] = {
            "support_image_id": int(row_meta["source_row_index"]),
            "source_row_index": int(row_meta["source_row_index"]),
            "file_name": row_meta["file_name"],
            "object_index": object_index,
            "category_name": category_name,
            "source_category_name": source_category_name,
            "category_id": category_id if category_id is not None else "",
            "bbox": json.dumps([float(v) for v in norm_bbox]),
            "bbox_original": json.dumps([float(v) for v in bbox]),
            "bbox_format": bbox_format,
            "bbox_format_original": "xywh_or_xyxy_checked",
            "channel_paths": json.dumps(channel_paths, ensure_ascii=False),
            "source_split": "support/prototype",
            "source_is_full_test": False,
            "source_parquet_path": row_meta.get("source_path", ""),
            "feature_missing": False,
            "skip_reason": "",
        }
        gray_by_channel: dict[str, np.ndarray] = {}
        for channel in CHANNELS:
            generic, gray = channel_generic_features(images[channel], norm_bbox)
            gray_by_channel[channel] = gray
            for name, value in generic.items():
                record[f"{channel}__{name}"] = value
        for name, value in colour_extra_features(images["colour"], norm_bbox).items():
            record[f"colour__{name}"] = value
        means = {channel: float(gray_by_channel[channel].mean()) for channel in CHANNELS}
        stds = {channel: float(gray_by_channel[channel].std()) for channel in CHANNELS}
        contrasts = {channel: float(gray_by_channel[channel].max() - gray_by_channel[channel].min()) for channel in CHANNELS}
        record["cross__high_minus_low_mean"] = means["high"] - means["low"]
        record["cross__high_low_ratio_mean"] = means["high"] / max(means["low"], 1e-6)
        record["cross__density_minus_grey_mean"] = means["density"] - means["grey"]
        record["cross__colour_gray_minus_density_mean"] = means["colour"] - means["density"]
        for channel in CHANNELS:
            record[f"cross__channel_mean_{channel}"] = means[channel]
            record[f"cross__channel_std_{channel}"] = stds[channel]
            record[f"cross__channel_contrast_{channel}"] = contrasts[channel]
        record["cross__high_low_contrast"] = contrasts["high"] - contrasts["low"]
        return record
    finally:
        for image in images.values():
            image.close()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# DET-COMPASS 多通道 support object feature 提取摘要",
        "",
        f"- support/prototype row 数：{summary['support_row_count']}",
        f"- object 总数：{summary['object_count_total']}",
        f"- 有效 object 数：{summary['object_count_valid']}",
        f"- skipped_no_bbox：{summary['skipped_no_bbox']}",
        f"- skipped_bad_bbox：{summary['skipped_bad_bbox']}",
        f"- skipped_missing_category：{summary['skipped_missing_category']}",
        f"- leakage_check：{summary['leakage_check']}",
        "",
        "## 每通道有效 object 数",
        "| channel | valid_count |",
        "|---|---:|",
    ]
    for channel, count in summary["per_channel_valid_count"].items():
        lines.append(f"| {channel} | {count} |")
    lines.extend(["", "## bank feature 维度", "| bank_type | feature_dim |", "|---|---:|"])
    for bank, dim in summary["feature_dim_by_bank_type"].items():
        lines.append(f"| {bank} | {dim} |")
    lines.extend(["", "## 每类别 object 数 top/bottom", "| category_name | count |", "|---|---:|"])
    counts = summary["per_category_object_count"]
    for name, count in sorted(counts.items(), key=lambda kv: (kv[1], kv[0]))[:20]:
        lines.append(f"| {name} | {count} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    category_map = full_test_category_map()
    category_name_map = full_test_category_name_map()
    rows_by_source = manifest_by_row()
    parquet_files = support_parquet_files_from_discovery()
    if not parquet_files:
        raise FileNotFoundError("Could not locate raxo/prototypes parquet files from Batch26 discovery or manifest")

    records: list[dict[str, Any]] = []
    skipped_no_bbox = 0
    skipped_bad_bbox = 0
    skipped_missing_category = 0
    object_total = 0

    for source_row_index, meta in sorted(rows_by_source.items()):
        meta["source_row_index"] = source_row_index
        channel_paths = meta.get("channel_paths", {})
        if set(channel_paths) != set(CHANNELS):
            skipped_bad_bbox += 1
            continue
        if not all(Path(channel_paths[ch]).exists() for ch in CHANNELS):
            skipped_bad_bbox += 1
            continue
        categories = [c for c in str(meta.get("categories", "")).split(";") if c]
        try:
            bboxes = json.loads(meta.get("xray_bbox") or "[]")
        except Exception:
            bboxes = []
        object_total += max(len(categories), len(bboxes))
        if not categories:
            skipped_missing_category += 1
            continue
        if not bboxes:
            skipped_no_bbox += len(categories)
            continue
        for obj_idx, category_name in enumerate(categories):
            if obj_idx >= len(bboxes):
                skipped_no_bbox += 1
                continue
            if not category_name:
                skipped_missing_category += 1
                continue
            lookup_key = category_name if category_name in category_map else normalize_category_name(category_name)
            category_id = category_map.get(lookup_key)
            if category_id is None:
                skipped_missing_category += 1
                continue
            target_category_name = category_name_map.get(lookup_key, category_name)
            try:
                record = extract_object_features(meta, obj_idx, target_category_name, category_name, bboxes[obj_idx], category_id)
                records.append(record)
            except Exception:
                skipped_bad_bbox += 1

    df = pd.DataFrame(records)
    if not df.empty:
        df["category_id"] = df["category_id"].astype("int64")
        df["support_image_id"] = df["support_image_id"].astype("int64")
        df["source_row_index"] = df["source_row_index"].astype("int64")
        df["object_index"] = df["object_index"].astype("int64")
    parquet_path = FEATURE_DIR / "support_multichannel_object_features.parquet"
    csv_path = FEATURE_DIR / "support_multichannel_object_features.csv"
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)

    per_channel_valid = {channel: int(len(df)) for channel in CHANNELS}
    per_category = Counter(df["category_name"].tolist()) if not df.empty else Counter()
    feature_dim_by_bank_type = {
        "colour": len(LEGACY_COLOUR_FEATURE_NAMES),
        "grey": len(GENERIC_FEATURE_NAMES),
        "high": len(GENERIC_FEATURE_NAMES),
        "low": len(GENERIC_FEATURE_NAMES),
        "density": len(GENERIC_FEATURE_NAMES),
        "multi_channel_all": len(CHANNELS) * len(GENERIC_FEATURE_NAMES) + len(CROSS_FEATURE_NAMES),
    }
    summary = {
        "support_row_count": len(rows_by_source),
        "source_parquet_files": parquet_files,
        "object_count_total": object_total,
        "object_count_valid": int(len(df)),
        "skipped_no_bbox": skipped_no_bbox,
        "skipped_bad_bbox": skipped_bad_bbox,
        "skipped_missing_category": skipped_missing_category,
        "per_channel_valid_count": per_channel_valid,
        "per_category_object_count": dict(sorted(per_category.items())),
        "feature_dim_by_bank_type": feature_dim_by_bank_type,
        "leakage_check": "no_full_test_bbox_used",
        "source_split": "support/prototype",
        "source_is_full_test": False,
    }
    write_json(FEATURE_DIR / "support_multichannel_feature_summary.json", summary)
    write_summary_md(FEATURE_DIR / "support_multichannel_feature_summary.md", summary)
    print(json.dumps({"object_features": len(df), "parquet": str(parquet_path), "csv": str(csv_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
