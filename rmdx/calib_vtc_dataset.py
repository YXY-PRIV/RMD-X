"""Dataset helpers for non-test Calib-VTC splits."""

from __future__ import annotations

import ast
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_literal(value: str) -> Any:
    if value in {"", None}:  # type: ignore[comparison-overlap]
        return None
    try:
        return ast.literal_eval(value)
    except Exception:
        return value


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def full_test_basenames(dataset_root: Path, dataset: str) -> set[str]:
    ann = dataset_root / dataset / "annotations" / "full_test.json"
    if not ann.exists():
        return set()
    data = load_json(ann)
    return {Path(img.get("file_name", "")).name for img in data.get("images", [])}


def stratified_roles(
    rows: list[dict[str, Any]],
    *,
    category_key: str = "category_name",
    calibration_ratio: float = 0.10,
    max_calibration_per_class: int = 500,
) -> list[dict[str, Any]]:
    """Assign deterministic bank_train/calibration roles by category."""

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[str(row.get(category_key, ""))].append(row)
    out: list[dict[str, Any]] = []
    for category, category_rows in sorted(by_category.items()):
        category_rows = sorted(
            category_rows,
            key=lambda r: (str(r.get("image_id", "")), str(r.get("annotation_id", ""))),
        )
        n = len(category_rows)
        if n == 0:
            continue
        calib_n = max(1, int(round(n * calibration_ratio)))
        calib_n = min(calib_n, max_calibration_per_class, n)
        for idx, row in enumerate(category_rows):
            row = dict(row)
            row["split_role"] = "calibration" if idx < calib_n else "bank_train"
            row["notes"] = (row.get("notes", "") + f"; class_count={n}; calib_count={calib_n}").strip("; ")
            out.append(row)
    return out


def count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row.get(key, ""))] += 1
    return dict(sorted(counts.items()))

