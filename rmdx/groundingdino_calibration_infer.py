#!/usr/bin/env python3
"""GroundingDINO inference wrapper for non-test calibration images.

This wrapper reads a calibration compatibility COCO annotation and emits COCO
result-list proposals. It never reads GT boxes as proposals.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image


DEFAULT_SOURCE = Path("/mnt/data/raxo/external_artifacts/groundingdino/source/GroundingDINO")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GroundingDINO calibration proposal inference")
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--coco-json", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--max-dets-per-image", type=int, default=300)
    parser.add_argument("--prompt-mode", choices=["all_categories", "per_category_prompt"], default="all_categories")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit-images", type=int, default=None)
    parser.add_argument("--log-path", default="")
    parser.add_argument("--source-path", default=str(DEFAULT_SOURCE))
    return parser.parse_args()


def normalize(text: str) -> str:
    return " ".join(text.lower().replace("_", " ").replace("-", " ").split())


def load_coco(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("info", {}).get("is_real_full_test") is True:
        raise RuntimeError("Refusing to run on a real full_test annotation")
    return data


def import_groundingdino(source_path: Path):
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))
    from groundingdino.util.inference import load_model, load_image, predict

    return load_model, load_image, predict


def xyxy_to_xywh(box: list[float], width: int, height: int) -> list[float] | None:
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(x1), float(width)))
    x2 = max(0.0, min(float(x2), float(width)))
    y1 = max(0.0, min(float(y1), float(height)))
    y2 = max(0.0, min(float(y2), float(height)))
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    if w <= 0 or h <= 0:
        return None
    return [x1, y1, w, h]


def predict_all_categories(model, load_image, predict, image_path: Path, categories: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[str]]:
    prompt = ". ".join(str(cat["name"]).replace("_", " ") for cat in categories)
    name_to_cat = {normalize(str(cat["name"])): cat for cat in categories}
    with Image.open(image_path) as im:
        width, height = im.size
    _image_source, image = load_image(str(image_path))
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        device=args.device,
    )
    rows = []
    unmapped = []
    for box, score, phrase in zip(boxes.tolist(), logits.tolist(), phrases):
        cat = name_to_cat.get(normalize(str(phrase)))
        if cat is None:
            unmapped.append(str(phrase))
            continue
        cx, cy, bw, bh = [float(v) for v in box]
        xywh = xyxy_to_xywh([(cx - bw / 2) * width, (cy - bh / 2) * height, (cx + bw / 2) * width, (cy + bh / 2) * height], width, height)
        if xywh is None:
            continue
        rows.append({"category_id": int(cat["id"]), "category_name": str(cat["name"]), "bbox": xywh, "score": float(score)})
    return rows, unmapped


def predict_per_category(model, load_image, predict, image_path: Path, categories: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[str]]:
    with Image.open(image_path) as im:
        width, height = im.size
    _image_source, image = load_image(str(image_path))
    rows = []
    unmapped: list[str] = []
    for cat in categories:
        prompt = str(cat["name"]).replace("_", " ")
        boxes, logits, _phrases = predict(
            model=model,
            image=image,
            caption=prompt,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            device=args.device,
        )
        for box, score in zip(boxes.tolist(), logits.tolist()):
            cx, cy, bw, bh = [float(v) for v in box]
            xywh = xyxy_to_xywh([(cx - bw / 2) * width, (cy - bh / 2) * height, (cx + bw / 2) * width, (cy + bh / 2) * height], width, height)
            if xywh is None:
                continue
            rows.append({"category_id": int(cat["id"]), "category_name": str(cat["name"]), "bbox": xywh, "score": float(score)})
    return rows, unmapped


def main() -> int:
    args = parse_args()
    source_path = Path(args.source_path)
    coco_json = Path(args.coco_json)
    image_root = Path(args.image_root)
    output_json = Path(args.output_json)
    status_rows = []
    data = load_coco(coco_json)
    images = data.get("images", [])[: args.limit_images]
    categories = data.get("categories", [])

    if args.dry_run:
        import_groundingdino(source_path)
        missing_images = [img["file_name"] for img in images if not (image_root / img["file_name"]).exists()]
        if missing_images:
            raise RuntimeError(f"Missing images: {missing_images[:5]}")
        print(json.dumps({"dry_run": True, "image_count": len(images), "category_count": len(categories)}, ensure_ascii=False))
        return 0

    load_model, load_image, predict = import_groundingdino(source_path)
    model = load_model(args.config_path, args.checkpoint_path, device=args.device)
    detections: list[dict[str, Any]] = []
    for img in images:
        start = time.time()
        image_path = image_root / img["file_name"]
        if args.prompt_mode == "per_category_prompt":
            preds, unmapped = predict_per_category(model, load_image, predict, image_path, categories, args)
        else:
            preds, unmapped = predict_all_categories(model, load_image, predict, image_path, categories, args)
        preds = sorted(preds, key=lambda row: row["score"], reverse=True)[: args.max_dets_per_image]
        for pred in preds:
            detections.append(
                {
                    "image_id": int(img["id"]),
                    "category_id": int(pred["category_id"]),
                    "bbox": pred["bbox"],
                    "score": float(pred["score"]),
                }
            )
        status_rows.append(
            {
                "image_id": int(img["id"]),
                "file_name": img["file_name"],
                "proposal_count": len(preds),
                "unmapped_count": len(unmapped),
                "runtime_sec": f"{time.time() - start:.3f}",
                "unmapped_phrase_sample": ";".join(unmapped[:20]),
            }
        )
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(detections, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.log_path:
        log_path = Path(args.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(status_rows[0].keys()) if status_rows else ["image_id"])
            writer.writeheader()
            writer.writerows(status_rows)
    print(json.dumps({"proposal_count": len(detections), "image_count": len(images), "output_json": str(output_json)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
