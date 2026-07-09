#!/usr/bin/env python3
"""Batch24: discover non-test support/train sources for material banks.

Read-only discovery. Ambiguous candidates are not used as main material bank
sources in this batch.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any


def find_root() -> Path:
    for candidate in [Path("/workspace/RAXO"), Path("/mnt/data/yxy/program/RAXO"), Path("/home/xinxueshi/yxy/program/RAXO")]:
        if candidate.exists() and (candidate / "outputs_rmdx").exists():
            return candidate
    return Path("/mnt/data/yxy/program/RAXO")


ROOT = find_root()
OUT_DIR = ROOT / "outputs_rmdx" / "material_multidataset"
DATASETS = ["DET-COMPASS", "PIXray", "pidray", "CLCXray", "HiXray", "DvXray"]
SEARCH_ROOTS = [
    Path("/datasets/xray-datasets"),
    Path("/mnt/data/raxo/datasets/xray-datasets"),
    Path("/home/xinxueshi/yxy/data"),
    Path("/home/xinxueshi/yxy/raxo_downloads"),
    ROOT / "outputs_rmdx" / "material",
]

FIELDS = [
    "dataset",
    "candidate_path",
    "candidate_type",
    "split_guess",
    "annotation_format_guess",
    "image_dir_guess",
    "is_full_test",
    "safe_for_main_material_bank",
    "expected_category_overlap",
    "recommendation",
    "notes",
]


def dataset_dir(dataset: str) -> Path | None:
    for root in SEARCH_ROOTS[:2]:
        path = root / dataset
        if path.exists():
            return path
    return None


def full_test_paths(dataset: str) -> set[str]:
    paths = set()
    base = dataset_dir(dataset)
    if base:
        paths.add(str((base / "annotations" / "full_test.json").resolve()))
    return paths


def split_guess(path: Path) -> str:
    text = str(path).lower()
    if "full_test" in text or "/test" in text or "_test" in text:
        return "test"
    if "support" in text:
        return "support"
    if "train" in text or "training" in text:
        return "train"
    if "val" in text or "validation" in text:
        return "val"
    return "unknown"


def annotation_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".json":
        try:
            if path.stat().st_size > 20 * 1024 * 1024:
                with path.open("r", encoding="utf-8", errors="ignore") as f:
                    head = f.read(8192)
                if all(token in head for token in ['"images"', '"annotations"', '"categories"']):
                    return "coco_json_large_light"
                return "json_large_skipped"
        except OSError:
            return "json_unreadable:OSError"
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and {"images", "annotations", "categories"}.issubset(data.keys()):
                return "coco_json"
            return "json_other"
        except Exception as exc:  # noqa: BLE001
            return f"json_unreadable:{type(exc).__name__}"
    if suffix == ".xml":
        return "xml"
    return "unknown"


def image_dir_for_candidate(path: Path, dataset: str) -> str:
    candidates = []
    if path.is_file():
        candidates.extend([path.parent.parent / "images", path.parent.parent / split_guess(path), path.parent.parent / "JPEGImages"])
    else:
        candidates.extend([path / "images", path / "JPEGImages", path / split_guess(path)])
    base = dataset_dir(dataset)
    if base:
        candidates.extend([base / "images", base / "train", base / "support", base / "JPEGImages"])
    for cand in candidates:
        if cand.exists() and cand.is_dir():
            return str(cand)
    return ""


def safe_status(path: Path, dataset: str) -> tuple[str, bool, str]:
    resolved = str(path.resolve()) if path.exists() else str(path)
    is_full = resolved in full_test_paths(dataset) or path.name == "full_test.json"
    split = split_guess(path)
    if is_full or split == "test":
        return "False", is_full, "test/full_test 不能作为主 material bank 来源"
    img_dir = image_dir_for_candidate(path, dataset)
    if split in {"train", "support"} and img_dir:
        return "True", is_full, "可作为候选，但本批次仍优先 transfer bank，需后续人工确认 split"
    if split in {"train", "support", "val"}:
        return "Unknown", is_full, "候选 split 非 test，但缺少明确 image dir 或格式需复核"
    return "Unknown", is_full, "无法判断 split，不能作为本批次主结果来源"


def expected_overlap(path: Path, dataset: str) -> str:
    if not path.is_file() or path.suffix.lower() != ".json":
        return "unknown"
    try:
        if path.stat().st_size > 20 * 1024 * 1024:
            return "skipped_large_json"
    except OSError:
        return "unknown"
    fmt = annotation_format(path)
    if fmt != "coco_json":
        return "unknown"
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return str(len(data.get("categories", [])))
    except Exception:
        return "unknown"


def collect_candidates(dataset: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    needles = ["train", "training", "support", "val", "validation", "annotations", "instances", "json", "xml", "images", "jpegimages"]
    roots = []
    base = dataset_dir(dataset)
    if base:
        roots.append(base)
    roots.extend(SEARCH_ROOTS[2:])
    seen = set()
    for root in roots:
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            current = Path(dirpath)
            depth = len(current.relative_to(root).parts) if current != root else 0
            if depth > 5:
                dirnames[:] = []
                continue
            all_names = [*dirnames, *filenames]
            for name in all_names:
                path = current / name
                lower = str(path).lower()
                if dataset.lower().replace("-", "") not in lower.replace("-", "") and root not in SEARCH_ROOTS[:2]:
                    continue
                if not any(needle in lower for needle in needles):
                    continue
                if path in seen:
                    continue
                seen.add(path)
                ctype = "dir" if path.is_dir() else "file"
                fmt = annotation_format(path) if path.is_file() else ("image_dir" if path.is_dir() else "unknown")
                safe, is_full, recommendation = safe_status(path, dataset)
                rows.append(
                    {
                        "dataset": dataset,
                        "candidate_path": str(path),
                        "candidate_type": ctype,
                        "split_guess": split_guess(path),
                        "annotation_format_guess": fmt,
                        "image_dir_guess": image_dir_for_candidate(path, dataset),
                        "is_full_test": is_full,
                        "safe_for_main_material_bank": safe,
                        "expected_category_overlap": expected_overlap(path, dataset),
                        "recommendation": recommendation,
                        "notes": "只读发现；本批次不使用未知来源构建主结果",
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# 多 Dataset support/train 来源发现",
        "",
        "本报告只做只读发现。`safe_for_main_material_bank=True` 也只是候选，后续仍需确认 split 不含 test 泄漏。",
        "",
        "| dataset | candidate_path | split | format | image_dir | safe | recommendation |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | `{row['candidate_path']}` | {row['split_guess']} | "
            f"{row['annotation_format_guess']} | `{row['image_dir_guess']}` | "
            f"{row['safe_for_main_material_bank']} | {row['recommendation']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        rows.extend(collect_candidates(dataset))
    write_csv(OUT_DIR / "support_source_discovery.csv", rows)
    write_md(OUT_DIR / "support_source_discovery.md", rows)
    (OUT_DIR / "support_source_discovery.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"candidate_count": len(rows), "safe_count": sum(1 for r in rows if r["safe_for_main_material_bank"] == "True")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
