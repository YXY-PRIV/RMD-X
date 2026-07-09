#!/usr/bin/env python3
"""Batch24: build multi-dataset transfer material banks.

The generated banks transfer colour-only material prototypes from the
DET-COMPASS support material bank to target dataset category ids. This script
does not read target dataset ground-truth boxes for material extraction.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def find_root() -> Path:
    for candidate in [Path("/workspace/RAXO"), Path("/mnt/data/yxy/program/RAXO"), Path("/home/xinxueshi/yxy/program/RAXO")]:
        if candidate.exists() and (candidate / "outputs_rmdx").exists():
            return candidate
    return Path("/mnt/data/yxy/program/RAXO")


ROOT = find_root()
DATASETS = ["DET-COMPASS", "PIXray", "pidray", "CLCXray", "HiXray", "DvXray"]
SOURCE_BANK = ROOT / "outputs_rmdx" / "material" / "material_colour_bank_support_full_bbox_only.json"
MAPPING_DIR = ROOT / "outputs_rmdx" / "material_multidataset" / "category_mapping"
BANK_DIR = ROOT / "outputs_rmdx" / "material_multidataset" / "banks"
COVERAGE_DIR = ROOT / "outputs_rmdx" / "material_multidataset" / "validation"
RESULT_DIR = ROOT / "outputs_rmdx" / "rmdx_results"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, title: str, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", "", "| " + " | ".join(fields) + " |", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_mapping(dataset: str) -> list[dict[str, Any]]:
    path = MAPPING_DIR / f"{dataset}_category_mapping.csv"
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["target_category_id"] = int(row["target_category_id"])
        row["target_annotation_count"] = int(row.get("target_annotation_count") or 0)
        row["target_image_count"] = int(row.get("target_image_count") or 0)
        row["is_mapped"] = str(row.get("is_mapped", "")).lower() in {"true", "1", "yes"}
    return rows


def source_prototypes_by_name(bank: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(proto.get("category_name", "")): proto for proto in bank.get("prototypes", {}).values()}


def weighted_global_prior(bank: dict[str, Any]) -> dict[str, Any]:
    protos = list(bank.get("prototypes", {}).values())
    feature_names = bank.get("feature_names", [])
    dim = len(feature_names)
    weights = [max(float(proto.get("sample_count") or 0), 1.0) for proto in protos]
    total = sum(weights) or 1.0
    mean = [0.0] * dim
    std = [0.0] * dim
    for proto, weight in zip(protos, weights):
        proto_mean = proto.get("mean_feature", [0.0] * dim)
        proto_std = proto.get("std_feature", [0.0] * dim)
        for i in range(dim):
            mean[i] += float(proto_mean[i]) * weight / total
            std[i] += float(proto_std[i]) * weight / total
    return {
        "category_id": None,
        "category_id_for_eval": None,
        "category_name": "global_material_prior",
        "feature_names": feature_names,
        "mean_feature": mean,
        "std_feature": std,
        "sample_count": 0,
        "source_category_id": None,
        "source_category_ids": [],
    }


def build_target_proto(
    dataset: str,
    row: dict[str, Any],
    source_proto: dict[str, Any],
    feature_names: list[str],
    is_fallback: bool,
) -> dict[str, Any]:
    target_id = int(row["target_category_id"])
    target_name = str(row["target_category_name"])
    source_category_id = source_proto.get("category_id")
    return {
        "target_dataset": dataset,
        "target_category_id": target_id,
        "target_category_name": target_name,
        "category_id": target_id,
        "category_id_for_eval": target_id,
        "category_name": target_name,
        "source_bank_type": "det_compass_support_material_bank" if not is_fallback else "global_material_prior_from_det_compass_support_bank",
        "source_category_name": source_proto.get("category_name"),
        "source_category_id_if_any": source_category_id,
        "source_category_id": source_category_id,
        "source_category_ids": source_proto.get("source_category_ids") or ([source_category_id] if source_category_id is not None else []),
        "feature_names": feature_names,
        "mean_feature": source_proto.get("mean_feature", []),
        "std_feature": source_proto.get("std_feature", []),
        "sample_count": int(source_proto.get("sample_count") or 0),
        "mapping_confidence": "fallback" if is_fallback else row.get("match_confidence", ""),
        "mapping_type": "global_material_prior" if is_fallback else row.get("match_type", ""),
        "is_fallback": is_fallback,
        "notes": row.get("notes", "") if not is_fallback else "类别未可靠映射；使用 DET-COMPASS support bank 全局均值先验，不作为强 material 证据。",
    }


def fallback_policy(fallback_count: int, total_count: int) -> tuple[str, bool]:
    ratio = fallback_count / total_count if total_count else 1.0
    if ratio == 0:
        return "NO_FALLBACK", True
    if ratio <= 0.30:
        return "PARTIAL_FALLBACK_OK_WITH_WARNING", True
    return "HIGH_FALLBACK_EXPLORATORY_ONLY", False


def build_bank_for_dataset(dataset: str, source_bank: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    rows = load_mapping(dataset)
    feature_names = list(source_bank.get("feature_names", []))
    source_by_name = source_prototypes_by_name(source_bank)
    global_prior = weighted_global_prior(source_bank)
    prototypes: dict[str, dict[str, Any]] = {}
    coverage_rows: list[dict[str, Any]] = []
    mapped_count = 0
    fallback_count = 0
    for row in rows:
        source_name = row.get("source_bank_category_name") or ""
        source_proto = source_by_name.get(source_name)
        is_fallback = source_proto is None
        if is_fallback:
            source_proto = global_prior
            fallback_count += 1
        else:
            mapped_count += 1
        target_id = int(row["target_category_id"])
        prototypes[str(target_id)] = build_target_proto(dataset, row, source_proto, feature_names, is_fallback)
        coverage_rows.append(
            {
                "dataset": dataset,
                "target_category_id": target_id,
                "target_category_name": row["target_category_name"],
                "source_category_name": source_name if not is_fallback else "global_material_prior",
                "source_category_id_if_any": row.get("source_bank_category_id_if_any") if not is_fallback else "",
                "match_type": row.get("match_type") if not is_fallback else "global_material_prior",
                "match_confidence": row.get("match_confidence") if not is_fallback else "fallback",
                "is_mapped": not is_fallback,
                "uses_fallback": is_fallback,
                "target_annotation_count": row.get("target_annotation_count", 0),
                "notes": prototypes[str(target_id)]["notes"],
            }
        )
    fallback_status, usable_for_main = fallback_policy(fallback_count, len(rows))
    bank = {
        "version": "material_colour_transfer_v1",
        "bank_name": f"{dataset}_transfer_from_det_compass_support",
        "mode": source_bank.get("mode", "bbox_only"),
        "feature_names": feature_names,
        "prototype_count": len(prototypes),
        "prototypes": prototypes,
        "bank_feature_stats": source_bank.get("bank_feature_stats", {}),
        "category_mapping": {str(k): int(k) for k in prototypes},
        "source_note": "Transfer bank built from DET-COMPASS support material bank; no target full_test bbox was used.",
        "transfer_metadata": {
            "target_dataset": dataset,
            "source_bank_path": str(SOURCE_BANK),
            "source_bank_name": source_bank.get("bank_name"),
            "target_category_count": len(rows),
            "mapped_count": mapped_count,
            "fallback_count": fallback_count,
            "fallback_ratio": round(fallback_count / len(rows), 4) if rows else 1.0,
            "fallback_policy": fallback_status,
            "usable_for_main_table": usable_for_main,
            "leakage_note": "只使用目标 full_test 的类别名/id 做映射；不读取目标 full_test bbox 构建 material feature。",
        },
    }
    summary = {
        "dataset": dataset,
        "bank_strategy": "TRANSFER_FROM_DET_COMPASS_SUPPORT_BANK" if fallback_count == 0 else "TRANSFER_WITH_PARTIAL_COVERAGE",
        "bank_path": str(BANK_DIR / f"{dataset}_material_bank_transfer_from_det_compass_support.json"),
        "category_count": len(rows),
        "mapped_count": mapped_count,
        "fallback_count": fallback_count,
        "fallback_ratio": round(fallback_count / len(rows), 4) if rows else 1.0,
        "fallback_policy": fallback_status,
        "usable_for_main_table": usable_for_main,
        "source_bank": str(SOURCE_BANK),
        "notes": "使用 DET-COMPASS support bank 迁移；不使用目标 full_test GT bbox。",
    }
    return bank, summary, coverage_rows


def main() -> None:
    BANK_DIR.mkdir(parents=True, exist_ok=True)
    COVERAGE_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    source_bank = load_json(SOURCE_BANK)

    strategy_rows: list[dict[str, Any]] = []
    all_coverage_rows: list[dict[str, Any]] = []
    det_compass_summary = {
        "dataset": "DET-COMPASS",
        "bank_strategy": "EXISTING_SUPPORT_BANK",
        "bank_path": str(SOURCE_BANK),
        "category_count": len(source_bank.get("prototypes", {})),
        "mapped_count": len(source_bank.get("prototypes", {})),
        "fallback_count": 0,
        "fallback_ratio": 0.0,
        "fallback_policy": "NO_FALLBACK",
        "usable_for_main_table": True,
        "source_bank": str(SOURCE_BANK),
        "notes": "DET-COMPASS 使用已去泄漏化的 support_full material bank。",
    }
    strategy_rows.append(det_compass_summary)

    coverage_fields = [
        "dataset",
        "target_category_id",
        "target_category_name",
        "source_category_name",
        "source_category_id_if_any",
        "match_type",
        "match_confidence",
        "is_mapped",
        "uses_fallback",
        "target_annotation_count",
        "notes",
    ]
    for dataset in DATASETS:
        if dataset == "DET-COMPASS":
            continue
        bank, summary, coverage_rows = build_bank_for_dataset(dataset, source_bank)
        write_json(BANK_DIR / f"{dataset}_material_bank_transfer_from_det_compass_support.json", bank)
        write_csv(COVERAGE_DIR / f"{dataset}_transfer_bank_coverage.csv", coverage_rows, coverage_fields)
        write_md(COVERAGE_DIR / f"{dataset}_transfer_bank_coverage.md", f"{dataset} transfer material bank 覆盖率", coverage_rows, coverage_fields)
        strategy_rows.append(summary)
        all_coverage_rows.extend(coverage_rows)

    strategy_fields = [
        "dataset",
        "bank_strategy",
        "bank_path",
        "category_count",
        "mapped_count",
        "fallback_count",
        "fallback_ratio",
        "fallback_policy",
        "usable_for_main_table",
        "source_bank",
        "notes",
    ]
    write_csv(RESULT_DIR / "multidataset_material_bank_strategy.csv", strategy_rows, strategy_fields)
    write_md(RESULT_DIR / "multidataset_material_bank_strategy.md", "多 Dataset material bank 策略", strategy_rows, strategy_fields)
    write_json(RESULT_DIR / "multidataset_material_bank_strategy.json", strategy_rows)
    write_csv(COVERAGE_DIR / "all_transfer_bank_coverage.csv", all_coverage_rows, coverage_fields)
    write_md(COVERAGE_DIR / "all_transfer_bank_coverage.md", "全部 transfer material bank 覆盖率", all_coverage_rows, coverage_fields)
    print(json.dumps({"strategy": strategy_rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
