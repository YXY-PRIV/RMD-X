#!/usr/bin/env python3
"""Batch24: inspect category mappings from each dataset to DET-COMPASS support bank.

Only reads annotations and an existing support material bank. It does not use
target full_test GT boxes for material extraction.
"""

from __future__ import annotations

import csv
import json
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def find_root() -> Path:
    for candidate in [Path("/workspace/RAXO"), Path("/mnt/data/yxy/program/RAXO"), Path("/home/xinxueshi/yxy/program/RAXO")]:
        if (candidate / "outputs_rmdx/material/material_colour_bank_support_full_bbox_only.json").exists():
            return candidate
    return Path("/mnt/data/yxy/program/RAXO")


ROOT = find_root()
DATASET_ROOTS = [Path("/datasets/xray-datasets"), Path("/mnt/data/raxo/datasets/xray-datasets"), Path("/data/raxo/datasets/xray-datasets")]
OUT_DIR = ROOT / "outputs_rmdx" / "material_multidataset" / "category_mapping"
SUPPORT_BANK = ROOT / "outputs_rmdx" / "material" / "material_colour_bank_support_full_bbox_only.json"
DATASETS = ["DET-COMPASS", "PIXray", "pidray", "CLCXray", "HiXray", "DvXray"]

ALIASES = {
    "pressure_vessel": "gas_canister",
    "bat": "baseball_bat",
    "scissors": "scissors",
    "razor_blade": "razor_blades",
    "pliers": "pliers",
    "dart": None,
    "knife": "knife",
    "fireworks": None,
    "battery": "charger",
    "saw_blade": "saw",
    "hammer": "hammer",
    "screwdriver": "screwdriver",
    "wrench": "wrench",
    "lighter": "lighter",
    "baton": None,
    "powerbank": "power_supply",
    "bullet": None,
    "sprayer": None,
    "handcuffs": None,
    "blade": "razor_blades",
    "swissarmyknife": "penknife",
    "plasticbottle": "water_bottle",
    "cans": "canned_food",
    "vacuumcup": "thermos",
    "glassbottle": "beer_bottle",
    "cartondrinks": "milk_carton",
    "tin": "tin_of_sweets",
    "spraycans": "paint_can",
    "mobile_phone": "smartphone",
    "laptop": "laptop",
    "portable_charger": "charger",
    "tablet": "ipad",
    "cosmetic": "lipstick",
    "water": "water_bottle",
    "nonmetallic_lighter": "lighter",
}


def dataset_dir(dataset: str) -> Path:
    for root in DATASET_ROOTS:
        path = root / dataset
        if path.exists():
            return path
    return DATASET_ROOTS[0] / dataset


def annotation_path(dataset: str) -> Path:
    return dataset_dir(dataset) / "annotations" / "full_test.json"


def norm_basic(name: str) -> str:
    return str(name).strip()


def norm_lower(name: str) -> str:
    return norm_basic(name).lower()


def norm_punct(name: str) -> str:
    value = norm_lower(name).replace("-", "_").replace(" ", "_")
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def norm_compact(name: str) -> str:
    return "".join(ch for ch in norm_lower(name) if ch not in string.punctuation and not ch.isspace())


def singular(value: str) -> str:
    if value.endswith("ies"):
        return value[:-3] + "y"
    if value.endswith("s") and not value.endswith("ss"):
        return value[:-1]
    return value


def load_annotation(dataset: str) -> dict[str, Any]:
    with annotation_path(dataset).open("r", encoding="utf-8") as f:
        return json.load(f)


def category_stats(annotation: dict[str, Any]) -> dict[int, dict[str, Any]]:
    ann_count: Counter[int] = Counter()
    image_ids: dict[int, set[int]] = defaultdict(set)
    for ann in annotation.get("annotations", []):
        cid = int(ann["category_id"])
        ann_count[cid] += 1
        image_ids[cid].add(int(ann["image_id"]))
    out = {}
    for cat in annotation.get("categories", []):
        cid = int(cat["id"])
        out[cid] = {
            "category_id": cid,
            "category_name": str(cat.get("name", "")),
            "supercategory": cat.get("supercategory", ""),
            "annotation_count": ann_count[cid],
            "image_count": len(image_ids[cid]),
        }
    return out


def source_bank_names(bank: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out = {}
    for key, proto in bank.get("prototypes", {}).items():
        name = str(proto.get("category_name", ""))
        out[name] = {"source_key": key, "category_id": proto.get("category_id"), "prototype": proto}
    return out


def build_indices(names: dict[str, dict[str, Any]]) -> dict[str, dict[str, str]]:
    return {
        "exact": {name: name for name in names},
        "lower": {norm_lower(name): name for name in names},
        "punct": {norm_punct(name): name for name in names},
        "compact": {norm_compact(name): name for name in names},
        "singular": {singular(norm_punct(name)): name for name in names},
    }


def map_name(target_name: str, source_names: dict[str, dict[str, Any]], indices: dict[str, dict[str, str]]) -> tuple[str | None, str, str, str]:
    if target_name in indices["exact"]:
        return indices["exact"][target_name], "exact_match", "high", ""
    lower = norm_lower(target_name)
    if lower in indices["lower"]:
        return indices["lower"][lower], "lower_case_match", "high", ""
    punct = norm_punct(target_name)
    if punct in indices["punct"]:
        return indices["punct"][punct], "punctuation_insensitive_match", "high", ""
    compact = norm_compact(target_name)
    if compact in indices["compact"]:
        return indices["compact"][compact], "whitespace_insensitive_match", "high", ""
    sing = singular(punct)
    if sing in indices["singular"]:
        return indices["singular"][sing], "singular_plural_simple_match", "medium", ""
    alias = ALIASES.get(punct, ALIASES.get(compact))
    if alias and alias in source_names:
        return alias, "manual_alias_match", "medium", f"alias {target_name}->{alias}"
    if alias is None and (punct in ALIASES or compact in ALIASES):
        return None, "manual_alias_missing", "missing", f"alias explicitly unavailable for {target_name}"
    return None, "missing", "missing", "no reliable source category match"


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, title: str, rows: list[dict[str, Any]], fields: list[str]) -> None:
    lines = [f"# {title}", "", "| " + " | ".join(fields) + " |", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bank = json.load(SUPPORT_BANK.open("r", encoding="utf-8"))
    source_names = source_bank_names(bank)
    indices = build_indices(source_names)
    fields = [
        "dataset",
        "target_category_id",
        "target_category_name",
        "target_annotation_count",
        "target_image_count",
        "source_bank_category_name",
        "source_bank_category_id_if_any",
        "match_type",
        "match_confidence",
        "is_mapped",
        "notes",
    ]
    all_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        annotation = load_annotation(dataset)
        stats = category_stats(annotation)
        rows = []
        for cid, rec in sorted(stats.items()):
            source_name, match_type, confidence, notes = map_name(rec["category_name"], source_names, indices)
            source_id = source_names.get(source_name or "", {}).get("category_id")
            row = {
                "dataset": dataset,
                "target_category_id": cid,
                "target_category_name": rec["category_name"],
                "target_annotation_count": rec["annotation_count"],
                "target_image_count": rec["image_count"],
                "source_bank_category_name": source_name or "",
                "source_bank_category_id_if_any": source_id if source_id is not None else "",
                "match_type": match_type,
                "match_confidence": confidence,
                "is_mapped": bool(source_name),
                "notes": notes,
            }
            rows.append(row)
            all_rows.append(row)
        write_csv(OUT_DIR / f"{dataset}_category_mapping.csv", rows, fields)
        write_md(OUT_DIR / f"{dataset}_category_mapping.md", f"{dataset} 类别映射", rows, fields)
        (OUT_DIR / f"{dataset}_category_mapping.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        mapped = sum(1 for row in rows if row["is_mapped"])
        summary_rows.append(
            {
                "dataset": dataset,
                "category_count": len(rows),
                "mapped_count": mapped,
                "missing_count": len(rows) - mapped,
                "mapping_coverage": round(mapped / len(rows), 4) if rows else 0,
                "high_count": sum(1 for row in rows if row["match_confidence"] == "high"),
                "medium_count": sum(1 for row in rows if row["match_confidence"] == "medium"),
                "missing_confidence_count": sum(1 for row in rows if row["match_confidence"] == "missing"),
            }
        )
    write_csv(OUT_DIR / "all_dataset_category_mapping_summary.csv", summary_rows, list(summary_rows[0]))
    write_md(OUT_DIR / "all_dataset_category_mapping_summary.md", "全部数据集类别映射摘要", summary_rows, list(summary_rows[0]))
    alias_rows = [{"target_normalized": k, "source_category_name": v if v is not None else "", "notes": "manual alias rule"} for k, v in sorted(ALIASES.items())]
    write_csv(OUT_DIR / "manual_alias_rules.csv", alias_rows, ["target_normalized", "source_category_name", "notes"])
    write_md(OUT_DIR / "manual_alias_rules.md", "手工 Alias 规则", alias_rows, ["target_normalized", "source_category_name", "notes"])
    print(json.dumps({"summary": summary_rows, "out_dir": str(OUT_DIR)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
