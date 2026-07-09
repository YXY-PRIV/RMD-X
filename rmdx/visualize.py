"""Visualization helpers for RMD-X case studies."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


COLORS = {
    "gt": (40, 220, 80),
    "baseline": (255, 80, 80),
    "material": (80, 150, 255),
    "fusion": (255, 210, 40),
}


def _font(size: int = 12) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def xywh_to_xyxy(bbox: list[float]) -> tuple[float, float, float, float]:
    x, y, w, h = [float(v) for v in bbox[:4]]
    return x, y, x + w, y + h


def draw_box(
    draw: ImageDraw.ImageDraw,
    bbox: list[float],
    color: tuple[int, int, int],
    label: str,
    *,
    width: int = 2,
) -> None:
    x0, y0, x1, y1 = xywh_to_xyxy(bbox)
    for offset in range(width):
        draw.rectangle([x0 - offset, y0 - offset, x1 + offset, y1 + offset], outline=color)
    font = _font(11)
    text_bbox = draw.textbbox((x0, y0), label, font=font)
    bg = [text_bbox[0], text_bbox[1], text_bbox[2] + 2, text_bbox[3] + 2]
    draw.rectangle(bg, fill=(0, 0, 0))
    draw.text((x0 + 1, y0 + 1), label, fill=color, font=font)


def detection_label(det: dict[str, Any], cat_names: dict[int, str], prefix: str) -> str:
    cat = cat_names.get(int(det["category_id"]), str(det["category_id"]))
    score = float(det.get("score", 0.0))
    parts = [f"{prefix}:{cat}", f"s={score:.2f}"]
    if "score_rmdx" in det:
        parts.append(f"rx={float(det['score_rmdx']):.2f}")
    if "rmdx_material_score" in det:
        parts.append(f"m={float(det['rmdx_material_score']):.2f}")
    if "rmdx_reliability" in det:
        parts.append(f"r={float(det['rmdx_reliability']):.2f}")
    return " ".join(parts)


def top_dets_for_image(
    dets_by_image: dict[int, list[dict[str, Any]]],
    image_id: int,
    limit: int = 8,
) -> list[dict[str, Any]]:
    return sorted(
        dets_by_image.get(image_id, []),
        key=lambda det: float(det.get("score", 0.0)),
        reverse=True,
    )[:limit]


def render_case_study(
    *,
    image_path: str | Path,
    output_path: str | Path,
    image_id: int,
    title: str,
    gt_by_image: dict[int, list[dict[str, Any]]],
    baseline_by_image: dict[int, list[dict[str, Any]]],
    material_by_image: dict[int, list[dict[str, Any]]],
    fusion_by_image: dict[int, list[dict[str, Any]]],
    cat_names: dict[int, str],
    max_dets_per_method: int = 8,
) -> None:
    image = Image.open(image_path).convert("RGB")
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    font = _font(14)
    draw.rectangle([0, 0, canvas.width, 28], fill=(0, 0, 0))
    draw.text((6, 6), title, fill=(255, 255, 255), font=font)

    for gt in gt_by_image.get(image_id, []):
        cat = cat_names.get(int(gt["category_id"]), str(gt["category_id"]))
        draw_box(draw, gt["bbox"], COLORS["gt"], f"GT:{cat}", width=3)

    for det in top_dets_for_image(baseline_by_image, image_id, max_dets_per_method):
        draw_box(draw, det["bbox"], COLORS["baseline"], detection_label(det, cat_names, "B"), width=2)
    for det in top_dets_for_image(material_by_image, image_id, max_dets_per_method):
        draw_box(draw, det["bbox"], COLORS["material"], detection_label(det, cat_names, "M"), width=2)
    for det in top_dets_for_image(fusion_by_image, image_id, max_dets_per_method):
        draw_box(draw, det["bbox"], COLORS["fusion"], detection_label(det, cat_names, "F"), width=2)

    legend_h = 22
    out = Image.new("RGB", (canvas.width, canvas.height + legend_h), (255, 255, 255))
    out.paste(canvas, (0, 0))
    ldraw = ImageDraw.Draw(out)
    x = 6
    for key, name in [
        ("gt", "GT"),
        ("baseline", "Baseline"),
        ("material", "Support material"),
        ("fusion", "Fusion"),
    ]:
        ldraw.rectangle([x, canvas.height + 5, x + 12, canvas.height + 17], fill=COLORS[key])
        ldraw.text((x + 16, canvas.height + 3), name, fill=(0, 0, 0), font=_font(12))
        x += 125
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(output_path)
