"""Dataset utilities for XDA-GDINO non-test calibration data."""

from __future__ import annotations

import ast
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SPLIT_MANIFEST = ROOT / "outputs_rmdx/calib_vtc/splits/calibration_split_manifest.csv"
FULL_TEST_ANN = Path("/mnt/data/raxo/datasets/xray-datasets/DET-COMPASS/annotations/full_test.json")


def parse_bbox(text: str) -> list[float] | None:
    try:
        vals = ast.literal_eval(text)
        if len(vals) != 4:
            return None
        x, y, w, h = [float(v) for v in vals]
        if w <= 0 or h <= 0:
            return None
        return [x, y, w, h]
    except Exception:
        return None


def build_coco_from_manifest(manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    full = json.loads(FULL_TEST_ANN.read_text(encoding="utf-8"))
    categories = full.get("categories", [])
    cat_ids = {int(c["id"]) for c in categories}

    by_role: dict[str, dict[str, Any]] = {
        "train": {"images": {}, "annotations": []},
        "val": {"images": {}, "annotations": []},
    }
    skipped = Counter()
    ann_id = 1
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("dataset") != "DET-COMPASS":
                continue
            if row.get("source_is_full_test") == "True" or row.get("source_is_test") == "True":
                skipped["leakage_guard"] += 1
                continue
            split_role = row.get("split_role")
            role = "train" if split_role == "bank_train" else "val" if split_role == "calibration" else ""
            if not role:
                skipped["unknown_role"] += 1
                continue
            image_path = Path(row.get("image_path", ""))
            if not image_path.exists():
                skipped["missing_image"] += 1
                continue
            bbox = parse_bbox(row.get("bbox", ""))
            if bbox is None:
                skipped["bad_bbox"] += 1
                continue
            category_id = int(row["category_id"])
            if category_id not in cat_ids:
                skipped["bad_category"] += 1
                continue
            try:
                with Image.open(image_path) as im:
                    width, height = im.size
            except Exception:
                skipped["unreadable_image"] += 1
                continue
            image_id = int(row["image_id"])
            by_role[role]["images"].setdefault(
                image_id,
                {
                    "id": image_id,
                    "file_name": image_path.name,
                    "width": width,
                    "height": height,
                    "source_path": str(image_path),
                    "source_split": split_role,
                    "is_real_full_test": False,
                },
            )
            by_role[role]["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    outputs = {}
    for role, data in by_role.items():
        coco = {
            "info": {"description": f"XDA-GDINO DET-COMPASS {role}; non-test only", "is_real_full_test": False},
            "images": list(data["images"].values()),
            "annotations": data["annotations"],
            "categories": categories,
        }
        path = output_dir / f"{role}_coco.json"
        path.write_text(json.dumps(coco, ensure_ascii=False) + "\n", encoding="utf-8")
        outputs[role] = {"path": str(path), "image_count": len(coco["images"]), "object_count": len(coco["annotations"])}

    prompts = {
        str(cat["id"]): [str(cat["name"]), f"x-ray image of {cat['name']}", f"security x-ray object {cat['name']}"]
        for cat in categories
    }
    (output_dir / "category_prompts.json").write_text(json.dumps(prompts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    cat_counts = Counter()
    for role in by_role.values():
        for ann in role["annotations"]:
            cat_counts[int(ann["category_id"])] += 1
    low_sample = sum(1 for c in categories if cat_counts[int(c["id"])] < 3)
    return {
        "train_coco": outputs["train"]["path"],
        "val_coco": outputs["val"]["path"],
        "category_prompts": str(output_dir / "category_prompts.json"),
        "train_image_count": outputs["train"]["image_count"],
        "train_object_count": outputs["train"]["object_count"],
        "val_image_count": outputs["val"]["image_count"],
        "val_object_count": outputs["val"]["object_count"],
        "category_count": len(categories),
        "low_sample_category_count": low_sample,
        "skipped_json": json.dumps(skipped, ensure_ascii=False),
        "leakage_check": "no_full_test_or_test_rows_used",
        "status": "READY_XDA_GDINO_DATA" if outputs["train"]["object_count"] and outputs["val"]["object_count"] else "BLOCKED_DATA_FORMAT",
    }


def markdown_summary(row: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Batch56 XDA-GDINO train / val 数据构建",
            "",
            f"- status：`{row['status']}`",
            f"- train images / objects：`{row['train_image_count']}` / `{row['train_object_count']}`",
            f"- val images / objects：`{row['val_image_count']}` / `{row['val_object_count']}`",
            f"- category_count：`{row['category_count']}`",
            f"- low_sample_category_count：`{row['low_sample_category_count']}`",
            f"- leakage_check：`{row['leakage_check']}`",
            f"- skipped：`{row['skipped_json']}`",
            "",
            "说明：该数据来自 DET-COMPASS support/prototype non-test split；未使用 full_test 图像或 bbox。",
        ]
    )

