"""Helpers for building non-test calibration compatibility datasets."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

from PIL import Image


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else ["status"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def safe_symlink(source: Path, target: Path) -> tuple[bool, str]:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if target.is_symlink() and Path(os.readlink(target)) == source:
            return True, "existing_symlink_ok"
        if target.exists() and target.resolve() == source.resolve():
            return True, "existing_path_ok"
        return False, "target_exists_different"
    target.symlink_to(source)
    return True, "created"


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size
