"""Multi-channel material feature adapter for RMD-X cache fusion.

This module is intentionally external to RAXO. It reads detector proposals
from the locked RMD-X cache, extracts inference-time material features from
DET-COMPASS multi-channel test images, and compares them against support /
prototype material banks built without full_test GT boxes.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from rmdx.material import FEATURE_NAMES as LEGACY_COLOUR_FEATURE_NAMES
from rmdx.material import _edge_density as legacy_edge_density
from rmdx.material import _rgb_to_hsv as legacy_rgb_to_hsv
from rmdx.material import extract_colour_feature_from_array, xywh_to_int_bounds


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
    "cross__high_minus_low_mean",
    "cross__high_low_ratio_mean",
    "cross__density_minus_grey_mean",
    "cross__colour_gray_minus_density_mean",
    "cross__channel_mean_colour",
    "cross__channel_mean_grey",
    "cross__channel_mean_high",
    "cross__channel_mean_low",
    "cross__channel_mean_density",
    "cross__channel_std_colour",
    "cross__channel_std_grey",
    "cross__channel_std_high",
    "cross__channel_std_low",
    "cross__channel_std_density",
    "cross__channel_contrast_colour",
    "cross__channel_contrast_grey",
    "cross__channel_contrast_high",
    "cross__channel_contrast_low",
    "cross__channel_contrast_density",
    "cross__high_low_contrast",
]


def load_json(path: str | Path) -> Any:
    with Path(path).open() as f:
        return json.load(f)


def save_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def resolve_first_existing(candidates: list[str | Path]) -> Path:
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path
    return Path(candidates[0])


def default_multichannel_test_root() -> Path:
    return resolve_first_existing(
        [
            "/datasets/xray-datasets/DET-COMPASS/multichannel/test",
            "/mnt/data/raxo/datasets/xray-datasets/DET-COMPASS/multichannel/test",
            "/data/raxo/datasets/xray-datasets/DET-COMPASS/multichannel/test",
        ]
    )


def infer_bank_channel_set(feature_names: list[str], channel_hint: str | None = None) -> list[str]:
    if feature_names == LEGACY_COLOUR_FEATURE_NAMES:
        return ["colour"]
    if all(name in GENERIC_FEATURE_NAMES for name in feature_names):
        if channel_hint in CHANNELS:
            return [channel_hint]
        return []
    prefixes = {name.split("__", 1)[0] for name in feature_names if "__" in name}
    return [channel for channel in CHANNELS if channel in prefixes]


def load_material_bank(path: str | Path) -> dict[str, Any]:
    bank = load_json(path)
    if "feature_names" not in bank or "prototypes" not in bank:
        raise ValueError(f"Invalid material bank schema: {path}")
    prototypes = bank["prototypes"]
    if not isinstance(prototypes, dict):
        raise ValueError(f"Expected prototypes dict in material bank: {path}")
    return bank


def bank_matrix_for_categories(
    bank: dict[str, Any],
    category_meta: list[dict[str, Any]],
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Build a category-aligned bank feature matrix.

    Missing categories are filled with zeros and marked in the returned meta.
    Batch27 coverage is 307/307, so missing entries should only indicate a
    schema mismatch.
    """

    feature_names = list(bank["feature_names"])
    feature_dim = len(feature_names)
    by_id: dict[int, dict[str, Any]] = {}
    by_name: dict[str, dict[str, Any]] = {}
    for raw_key, proto in bank["prototypes"].items():
        category_id = proto.get("category_id_for_eval", proto.get("category_id", raw_key))
        try:
            by_id[int(category_id)] = proto
        except (TypeError, ValueError):
            pass
        name = proto.get("category_name")
        if name is not None:
            by_name[str(name)] = proto

    rows = []
    meta = []
    for class_index, category in enumerate(category_meta):
        proto = by_id.get(int(category["category_id"])) or by_name.get(str(category["category_name"]))
        if proto is None:
            rows.append([0.0] * feature_dim)
            meta.append(
                {
                    "class_index": class_index,
                    "category_id": int(category["category_id"]),
                    "category_name": str(category["category_name"]),
                    "missing_bank_entry": True,
                    "sample_count": 0,
                }
            )
            continue
        proto_feature_names = list(proto.get("feature_names", feature_names))
        if proto_feature_names != feature_names:
            raise ValueError(f"Bank feature_names mismatch for category {category}")
        rows.append([float(v) for v in proto["mean_feature"]])
        meta.append(
            {
                "class_index": class_index,
                "category_id": int(category["category_id"]),
                "category_name": str(category["category_name"]),
                "missing_bank_entry": False,
                "sample_count": int(proto.get("sample_count", 0)),
                "low_sample_count": bool(proto.get("low_sample_count", False)),
            }
        )
    return torch.tensor(rows, dtype=torch.float32), meta


def _entropy(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    hist, _ = np.histogram(values, bins=32, range=(0.0, 1.0), density=False)
    total = hist.sum()
    if total <= 0:
        return 0.0
    probs = hist.astype(np.float64) / float(total)
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum() / math.log2(32))


def _edge_and_gradient(gray: np.ndarray) -> tuple[float, float, float]:
    if gray.size <= 1:
        return 0.0, 0.0, 0.0
    if gray.shape[0] > 256 or gray.shape[1] > 256:
        stride = max(1, int(math.ceil(max(gray.shape[:2]) / 256)))
        gray = gray[::stride, ::stride]
    parts = []
    if gray.shape[1] > 1:
        gx = np.abs(np.diff(gray, axis=1))
        parts.append(gx.reshape(-1))
    if gray.shape[0] > 1:
        gy = np.abs(np.diff(gray, axis=0))
        parts.append(gy.reshape(-1))
    if not parts:
        return 0.0, 0.0, 0.0
    grad = np.concatenate(parts)
    return float(np.mean(grad > 0.08)), float(grad.mean()), float(grad.std())


def _as_gray_float(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L"), dtype=np.float32) / 255.0


def extract_generic_channel_feature_from_gray_array(
    gray: np.ndarray,
    bbox: list[float],
    expected_names: list[str] | None = None,
) -> list[float]:
    height, width = gray.shape[:2]
    x0, y0, x1, y1 = xywh_to_int_bounds(bbox, width, height)
    crop = gray[y0:y1, x0:x1]
    if crop.size == 0:
        crop = gray[max(0, y0) : max(0, y0 + 1), max(0, x0) : max(0, x0 + 1)]
    values = crop.reshape(-1)
    if values.size > 8192:
        stride = max(1, int(math.ceil(values.size / 8192)))
        values = values[::stride][:8192]
    q10, q25, q75, q90 = np.quantile(values, [0.10, 0.25, 0.75, 0.90])
    edge_density, gradient_mean, gradient_std = _edge_and_gradient(crop)
    area_ratio = float(((x1 - x0) * (y1 - y0)) / max(width * height, 1))
    aspect_ratio = float((x1 - x0) / max(y1 - y0, 1))
    feature = [
        float(values.mean()),
        float(values.std()),
        float(values.min()),
        float(values.max()),
        float(np.median(values)),
        float(q10),
        float(q25),
        float(q75),
        float(q90),
        _entropy(values),
        float(np.mean(values > 1e-6)),
        area_ratio,
        aspect_ratio,
        float(x1 - x0),
        float(y1 - y0),
        edge_density,
        gradient_mean,
        gradient_std,
    ]
    if expected_names is not None and expected_names != GENERIC_FEATURE_NAMES:
        raise ValueError(f"Unexpected generic feature order: {expected_names}")
    return feature


def extract_legacy_colour_feature_fast(rgb_array_uint8: np.ndarray, bbox: list[float]) -> list[float]:
    """Compute the 22D legacy colour feature with bounded crop sampling.

    The original helper samples pixels but still materializes full crops for
    large proposals. Batch28 uses this bounded variant only for newly rebuilt
    colour proposal features; the exact legacy reference still reuses the
    locked Batch17/18 material similarity.
    """

    height, width = rgb_array_uint8.shape[:2]
    x0, y0, x1, y1 = xywh_to_int_bounds(bbox, width, height)
    crop = rgb_array_uint8[y0:y1, x0:x1]
    if crop.size == 0:
        crop = rgb_array_uint8[max(0, y0) : max(0, y0 + 1), max(0, x0) : max(0, x0 + 1)]
    if crop.shape[0] > 192 or crop.shape[1] > 192:
        stride = max(1, int(math.ceil(max(crop.shape[:2]) / 192)))
        crop_for_stats = crop[::stride, ::stride]
    else:
        crop_for_stats = crop
    pixels = crop_for_stats.reshape(-1, 3).astype(np.float32) / 255.0
    if pixels.shape[0] > 8192:
        stride = max(1, int(math.ceil(pixels.shape[0] / 8192)))
        pixels = pixels[::stride][:8192]
    rgb_mean = pixels.mean(axis=0)
    rgb_std = pixels.std(axis=0)
    hsv = legacy_rgb_to_hsv(pixels.reshape(-1, 1, 3)).reshape(-1, 3)
    hsv_mean = hsv.mean(axis=0)
    hsv_std = hsv.std(axis=0)
    gray_pixels = (0.299 * pixels[:, 0]) + (0.587 * pixels[:, 1]) + (0.114 * pixels[:, 2])
    quantiles = np.quantile(gray_pixels, [0.10, 0.25, 0.50, 0.75, 0.90])
    gray_crop = (
        (0.299 * crop_for_stats[:, :, 0])
        + (0.587 * crop_for_stats[:, :, 1])
        + (0.114 * crop_for_stats[:, :, 2])
    ) / 255.0
    area_ratio = float(((x1 - x0) * (y1 - y0)) / max(width * height, 1))
    aspect_ratio = float((x1 - x0) / max(y1 - y0, 1))
    feature = np.concatenate(
        [
            rgb_mean,
            rgb_std,
            hsv_mean,
            hsv_std,
            np.array([float(gray_pixels.mean()), float(gray_pixels.std())]),
            quantiles,
            np.array([legacy_edge_density(gray_crop), area_ratio, aspect_ratio]),
        ]
    )
    return [float(v) for v in feature]


def extract_generic_channel_feature(
    image: Image.Image,
    bbox: list[float],
    expected_names: list[str] | None = None,
) -> list[float]:
    gray = _as_gray_float(image)
    return extract_generic_channel_feature_from_gray_array(gray, bbox, expected_names=expected_names)


def extract_features_for_arrays(
    channel_arrays: dict[str, np.ndarray],
    bbox: list[float],
    feature_names: list[str],
    channel_hint: str | None = None,
) -> list[float]:
    if feature_names == LEGACY_COLOUR_FEATURE_NAMES:
        return extract_legacy_colour_feature_fast(channel_arrays["colour"], bbox)

    if all(name in GENERIC_FEATURE_NAMES for name in feature_names):
        if channel_hint not in CHANNELS:
            raise ValueError("Unprefixed generic feature bank requires a channel_hint")
        generic_values = extract_generic_channel_feature_from_gray_array(channel_arrays[channel_hint], bbox)
        generic = dict(zip(GENERIC_FEATURE_NAMES, generic_values))
        return [float(generic[name]) for name in feature_names]

    channel_features: dict[str, dict[str, float]] = {}
    for channel in infer_bank_channel_set(feature_names, channel_hint=channel_hint):
        generic_values = extract_generic_channel_feature_from_gray_array(channel_arrays[channel], bbox)
        channel_features[channel] = dict(zip(GENERIC_FEATURE_NAMES, generic_values))

    if any(name.startswith("cross__") for name in feature_names):
        for channel in CHANNELS:
            if channel not in channel_features:
                generic_values = extract_generic_channel_feature_from_gray_array(channel_arrays[channel], bbox)
                channel_features[channel] = dict(zip(GENERIC_FEATURE_NAMES, generic_values))
        means = {channel: channel_features[channel]["mean"] for channel in CHANNELS}
        stds = {channel: channel_features[channel]["std"] for channel in CHANNELS}
        contrasts = {channel: channel_features[channel]["max"] - channel_features[channel]["min"] for channel in CHANNELS}
        cross_features = {
            "cross__high_minus_low_mean": means["high"] - means["low"],
            "cross__high_low_ratio_mean": means["high"] / max(means["low"], 1e-6),
            "cross__density_minus_grey_mean": means["density"] - means["grey"],
            "cross__colour_gray_minus_density_mean": means["colour"] - means["density"],
            "cross__channel_mean_colour": means["colour"],
            "cross__channel_mean_grey": means["grey"],
            "cross__channel_mean_high": means["high"],
            "cross__channel_mean_low": means["low"],
            "cross__channel_mean_density": means["density"],
            "cross__channel_std_colour": stds["colour"],
            "cross__channel_std_grey": stds["grey"],
            "cross__channel_std_high": stds["high"],
            "cross__channel_std_low": stds["low"],
            "cross__channel_std_density": stds["density"],
            "cross__channel_contrast_colour": contrasts["colour"],
            "cross__channel_contrast_grey": contrasts["grey"],
            "cross__channel_contrast_high": contrasts["high"],
            "cross__channel_contrast_low": contrasts["low"],
            "cross__channel_contrast_density": contrasts["density"],
            "cross__high_low_contrast": contrasts["high"] - contrasts["low"],
        }
    else:
        cross_features = {}

    values = []
    for name in feature_names:
        if name.startswith("cross__"):
            values.append(float(cross_features[name]))
            continue
        if "__" not in name:
            raise ValueError(f"Feature name lacks channel prefix: {name}")
        channel, raw_name = name.split("__", 1)
        if channel not in channel_features or raw_name not in channel_features[channel]:
            raise ValueError(f"Cannot extract feature {name}")
        values.append(float(channel_features[channel][raw_name]))
    return values


def extract_features_for_proposal(
    channel_images: dict[str, Image.Image],
    bbox: list[float],
    feature_names: list[str],
    channel_hint: str | None = None,
) -> list[float]:
    """Extract one feature vector whose order exactly matches feature_names."""

    if feature_names == LEGACY_COLOUR_FEATURE_NAMES:
        colour = channel_images["colour"]
        rgb = np.asarray(colour.convert("RGB"), dtype=np.uint8)
        return extract_colour_feature_from_array(rgb, bbox)

    if all(name in GENERIC_FEATURE_NAMES for name in feature_names):
        if channel_hint not in CHANNELS:
            raise ValueError("Unprefixed generic feature bank requires a channel_hint")
        generic_values = extract_generic_channel_feature(channel_images[channel_hint], bbox)
        generic = dict(zip(GENERIC_FEATURE_NAMES, generic_values))
        return [float(generic[name]) for name in feature_names]

    channel_features: dict[str, dict[str, float]] = {}
    for channel in infer_bank_channel_set(feature_names, channel_hint=channel_hint):
        generic_values = extract_generic_channel_feature(channel_images[channel], bbox)
        channel_features[channel] = dict(zip(GENERIC_FEATURE_NAMES, generic_values))

    if any(name.startswith("cross__") for name in feature_names):
        for channel in CHANNELS:
            if channel not in channel_features:
                generic_values = extract_generic_channel_feature(channel_images[channel], bbox)
                channel_features[channel] = dict(zip(GENERIC_FEATURE_NAMES, generic_values))
        means = {channel: channel_features[channel]["mean"] for channel in CHANNELS}
        stds = {channel: channel_features[channel]["std"] for channel in CHANNELS}
        contrasts = {channel: channel_features[channel]["max"] - channel_features[channel]["min"] for channel in CHANNELS}
        cross_features = {
            "cross__high_minus_low_mean": means["high"] - means["low"],
            "cross__high_low_ratio_mean": means["high"] / max(means["low"], 1e-6),
            "cross__density_minus_grey_mean": means["density"] - means["grey"],
            "cross__colour_gray_minus_density_mean": means["colour"] - means["density"],
            "cross__channel_mean_colour": means["colour"],
            "cross__channel_mean_grey": means["grey"],
            "cross__channel_mean_high": means["high"],
            "cross__channel_mean_low": means["low"],
            "cross__channel_mean_density": means["density"],
            "cross__channel_std_colour": stds["colour"],
            "cross__channel_std_grey": stds["grey"],
            "cross__channel_std_high": stds["high"],
            "cross__channel_std_low": stds["low"],
            "cross__channel_std_density": stds["density"],
            "cross__channel_contrast_colour": contrasts["colour"],
            "cross__channel_contrast_grey": contrasts["grey"],
            "cross__channel_contrast_high": contrasts["high"],
            "cross__channel_contrast_low": contrasts["low"],
            "cross__channel_contrast_density": contrasts["density"],
            "cross__high_low_contrast": contrasts["high"] - contrasts["low"],
        }
    else:
        cross_features = {}

    values = []
    for name in feature_names:
        if name.startswith("cross__"):
            values.append(float(cross_features[name]))
            continue
        if "__" not in name:
            raise ValueError(f"Feature name lacks channel prefix: {name}")
        channel, raw_name = name.split("__", 1)
        if channel not in channel_features or raw_name not in channel_features[channel]:
            raise ValueError(f"Cannot extract feature {name}")
        values.append(float(channel_features[channel][raw_name]))
    return values


def _open_channel_images(
    multichannel_test_root: Path,
    file_name: str,
    required_channels: list[str],
) -> dict[str, Image.Image]:
    images = {}
    for channel in required_channels:
        path = multichannel_test_root / channel / file_name
        if not path.exists():
            raise FileNotFoundError(f"Missing {channel} image for {file_name}: {path}")
        images[channel] = Image.open(path).copy()
    return images


def extract_proposal_feature_matrix(
    proposal_meta: list[dict[str, Any]],
    feature_names: list[str],
    multichannel_test_root: str | Path | None = None,
    channel_hint: str | None = None,
) -> torch.Tensor:
    """Extract material features for all cached proposals.

    The function groups proposals by image to avoid repeatedly reopening the
    same multi-channel files.
    """

    root = Path(multichannel_test_root) if multichannel_test_root else default_multichannel_test_root()
    if not root.exists():
        raise FileNotFoundError(f"Multi-channel test root not found: {root}")

    if feature_names == LEGACY_COLOUR_FEATURE_NAMES:
        required_channels = ["colour"]
    else:
        required_channels = infer_bank_channel_set(feature_names, channel_hint=channel_hint)
        if any(name.startswith("cross__") for name in feature_names):
            required_channels = CHANNELS[:]
    if not required_channels:
        raise ValueError("Could not infer required channels from feature_names")

    by_file: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for idx, proposal in enumerate(proposal_meta):
        by_file[str(proposal["file_name"])].append((idx, proposal))

    features: list[list[float] | None] = [None] * len(proposal_meta)
    for file_index, (file_name, rows) in enumerate(sorted(by_file.items())):
        channel_images = _open_channel_images(root, file_name, required_channels)
        channel_arrays: dict[str, np.ndarray] = {}
        for channel, image in channel_images.items():
            if feature_names == LEGACY_COLOUR_FEATURE_NAMES and channel == "colour":
                channel_arrays[channel] = np.asarray(image.convert("RGB"), dtype=np.uint8)
            else:
                channel_arrays[channel] = _as_gray_float(image)
        for idx, proposal in rows:
            features[idx] = extract_features_for_arrays(
                channel_arrays,
                proposal["bbox"],
                feature_names,
                channel_hint=channel_hint,
            )
        for image in channel_images.values():
            image.close()
        if (file_index + 1) % 25 == 0:
            print(
                f"[material_multichannel] extracted {file_index + 1}/{len(by_file)} images "
                f"for {len(feature_names)}D bank",
                flush=True,
            )

    if any(row is None for row in features):
        missing = sum(row is None for row in features)
        raise RuntimeError(f"Missing extracted feature rows: {missing}")
    return torch.tensor(features, dtype=torch.float32)


def zscore_with_bank_stats(
    proposal_features: torch.Tensor,
    class_features: torch.Tensor,
    bank: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    stats = bank.get("bank_feature_stats") or {}
    mean = torch.tensor(stats.get("mean", []), dtype=torch.float32)
    std = torch.tensor(stats.get("std", []), dtype=torch.float32)
    if mean.numel() != proposal_features.shape[1] or std.numel() != proposal_features.shape[1]:
        mean = class_features.mean(dim=0)
        std = class_features.std(dim=0)
    std = torch.where(std.abs() < 1e-6, torch.ones_like(std), std)
    return (proposal_features - mean) / std, (class_features - mean) / std


def cosine_class_similarity(
    proposal_features: torch.Tensor,
    class_features: torch.Tensor,
    bank: dict[str, Any],
    *,
    chunk_size: int = 4096,
) -> torch.Tensor:
    proposal_z, class_z = zscore_with_bank_stats(proposal_features, class_features, bank)
    proposal_z = torch.nan_to_num(proposal_z.float(), nan=0.0, posinf=0.0, neginf=0.0)
    class_z = torch.nan_to_num(class_z.float(), nan=0.0, posinf=0.0, neginf=0.0)
    proposal_z = torch.nn.functional.normalize(proposal_z, p=2, dim=1, eps=1e-6)
    class_z = torch.nn.functional.normalize(class_z, p=2, dim=1, eps=1e-6)
    chunks = []
    for start in range(0, proposal_z.shape[0], chunk_size):
        chunks.append(proposal_z[start : start + chunk_size] @ class_z.T)
    return torch.cat(chunks, dim=0).float()


def build_material_similarity_for_bank(
    *,
    cache_dir: str | Path,
    bank_path: str | Path,
    multichannel_test_root: str | Path | None = None,
    channel_hint: str | None = None,
    proposal_feature_output: str | Path | None = None,
    similarity_output: str | Path | None = None,
    meta_output: str | Path | None = None,
) -> dict[str, Any]:
    cache_dir = Path(cache_dir)
    bank_path = Path(bank_path)
    bank = load_material_bank(bank_path)
    feature_names = list(bank["feature_names"])
    proposal_meta = load_jsonl(cache_dir / "proposal_meta.jsonl")
    category_meta = load_json(cache_dir / "category_meta.json")

    proposal_features = extract_proposal_feature_matrix(
        proposal_meta,
        feature_names,
        multichannel_test_root=multichannel_test_root,
        channel_hint=channel_hint,
    )
    class_features, class_meta = bank_matrix_for_categories(bank, category_meta)
    material_similarity = cosine_class_similarity(proposal_features, class_features, bank)

    if proposal_feature_output:
        Path(proposal_feature_output).parent.mkdir(parents=True, exist_ok=True)
        torch.save(proposal_features, proposal_feature_output)
    if similarity_output:
        Path(similarity_output).parent.mkdir(parents=True, exist_ok=True)
        torch.save(material_similarity, similarity_output)
    meta = {
        "bank_path": str(bank_path),
        "bank_name": bank.get("bank_name", bank_path.stem),
        "feature_dim": len(feature_names),
        "feature_names": feature_names,
        "channel_set": infer_bank_channel_set(feature_names, channel_hint=channel_hint),
        "proposal_count": len(proposal_meta),
        "category_count": len(category_meta),
        "proposal_feature_shape": list(proposal_features.shape),
        "material_class_similarity_shape": list(material_similarity.shape),
        "missing_bank_entry_count": sum(1 for row in class_meta if row["missing_bank_entry"]),
        "low_sample_category_count": sum(1 for row in class_meta if row.get("low_sample_count")),
        "leakage_check": bank.get("leakage_check", "no_full_test_bbox_used"),
    }
    if meta_output:
        save_json(meta_output, meta)
    return meta
