#!/usr/bin/env python3
"""Build DET-COMPASS multi-channel material similarity tensors for Batch28."""

from __future__ import annotations

import csv
import json
import shutil
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rmdx.material import FEATURE_NAMES as LEGACY_COLOUR_FEATURE_NAMES
from rmdx.material_multichannel import (
    bank_matrix_for_categories,
    build_material_similarity_for_bank,
    cosine_class_similarity,
    default_multichannel_test_root,
    load_json,
    load_material_bank,
)


CACHE_DIR = REPO_ROOT / "outputs_rmdx/cache/det_compass_groundingDINO_100_0_full"
OUT_ROOT = REPO_ROOT / "outputs_rmdx/material_multichannel_det_compass"
FEATURE_DIR = OUT_ROOT / "proposal_features"
SIM_DIR = OUT_ROOT / "material_similarity"
RESULT_DIR = REPO_ROOT / "outputs_rmdx/rmdx_results"
DOC_DIR = REPO_ROOT / "docs_rmdx"

BANKS = [
    {
        "bank_name": "legacy_colour_bank",
        "channel_set": "colour",
        "bank_path": REPO_ROOT / "outputs_rmdx/material/material_colour_bank_support_full_bbox_only.json",
        "reuse_cache_material": True,
    },
    {
        "bank_name": "new_colour_bank",
        "channel_set": "colour",
        "bank_path": OUT_ROOT / "banks/material_bank_colour_support.json",
    },
    {"bank_name": "grey_bank", "channel_set": "grey", "bank_path": OUT_ROOT / "banks/material_bank_grey_support.json"},
    {"bank_name": "high_bank", "channel_set": "high", "bank_path": OUT_ROOT / "banks/material_bank_high_support.json"},
    {"bank_name": "low_bank", "channel_set": "low", "bank_path": OUT_ROOT / "banks/material_bank_low_support.json"},
    {
        "bank_name": "density_bank",
        "channel_set": "density",
        "bank_path": OUT_ROOT / "banks/material_bank_density_support.json",
    },
    {
        "bank_name": "all_channel_bank",
        "channel_set": "colour+grey+high+low+density+cross",
        "bank_path": OUT_ROOT / "banks/material_bank_multichannel_all_support.json",
    },
]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# Batch28 material similarity 构建状态",
        "",
        "本表只构建 proposal 与 support material bank 的相似度，不运行 DINOv2、不运行 RAXO method，也不使用 full_test GT bbox。",
        "",
    ]
    if rows:
        headers = list(rows[0].keys())
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in rows:
            lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    path.write_text("\n".join(lines) + "\n")


def reuse_legacy_material(bank: dict[str, object]) -> dict[str, object]:
    bank_name = str(bank["bank_name"])
    bank_data = load_material_bank(bank["bank_path"])
    src_sim = CACHE_DIR / "material_class_similarity.pt"
    src_feature = CACHE_DIR / "proposal_features.pt"
    out_sim = SIM_DIR / f"{bank_name}_material_class_similarity.pt"
    out_feature = FEATURE_DIR / f"{bank_name}_proposal_material_features.pt"
    out_meta = SIM_DIR / f"{bank_name}_material_similarity_meta.json"
    if not out_sim.exists():
        shutil.copy2(src_sim, out_sim)
    if src_feature.exists() and not out_feature.exists():
        shutil.copy2(src_feature, out_feature)
    sim = torch.load(out_sim, map_location="cpu")
    meta = {
        "bank_name": bank_name,
        "bank_path": str(bank["bank_path"]),
        "feature_dim": len(bank_data["feature_names"]),
        "feature_names": bank_data["feature_names"],
        "channel_set": ["colour"],
        "proposal_count": int(sim.shape[0]),
        "category_count": int(sim.shape[1]),
        "proposal_feature_shape": "reused_from_locked_cache",
        "material_class_similarity_shape": list(sim.shape),
        "reuse_source": str(src_sim),
        "leakage_check": "legacy_support_bank_no_full_test_bbox_claim_from_prior_batches",
    }
    out_meta.write_text(json.dumps(meta, indent=2))
    return {
        "bank_name": bank_name,
        "channel_set": bank["channel_set"],
        "feature_dim": len(bank_data["feature_names"]),
        "proposal_count": int(sim.shape[0]),
        "category_count": int(sim.shape[1]),
        "output_path": str(out_sim),
        "build_time_sec": 0.0,
        "status": "REUSED_LOCKED_CACHE",
        "failure_reason": "",
        "sample_count_low_warning": "legacy_reference",
        "notes": "复用 Batch17/18 locked cache 中已有 legacy material_class_similarity，保证 legacy best 可复现。",
    }


def build_colour_similarity_from_locked_features(bank: dict[str, object]) -> dict[str, object]:
    bank_name = str(bank["bank_name"])
    bank_data = load_material_bank(bank["bank_path"])
    if list(bank_data["feature_names"]) != LEGACY_COLOUR_FEATURE_NAMES:
        raise ValueError(f"{bank_name} is not legacy 22D colour-compatible")
    src_feature = CACHE_DIR / "proposal_features.pt"
    out_feature = FEATURE_DIR / f"{bank_name}_proposal_material_features.pt"
    out_sim = SIM_DIR / f"{bank_name}_material_class_similarity.pt"
    out_meta = SIM_DIR / f"{bank_name}_material_similarity_meta.json"
    if not src_feature.exists():
        raise FileNotFoundError(src_feature)
    proposal_features = torch.load(src_feature, map_location="cpu").float()
    if proposal_features.ndim != 2 or proposal_features.shape[1] != len(LEGACY_COLOUR_FEATURE_NAMES):
        raise ValueError(
            f"Locked proposal_features.pt is {tuple(proposal_features.shape)}, not 22D colour material features"
        )
    if not out_feature.exists():
        shutil.copy2(src_feature, out_feature)
    category_meta = load_json(CACHE_DIR / "category_meta.json")
    class_features, class_meta = bank_matrix_for_categories(bank_data, category_meta)
    material_similarity = cosine_class_similarity(proposal_features, class_features, bank_data)
    torch.save(material_similarity, out_sim)
    meta = {
        "bank_name": bank_name,
        "bank_path": str(bank["bank_path"]),
        "feature_dim": len(bank_data["feature_names"]),
        "feature_names": bank_data["feature_names"],
        "channel_set": ["colour"],
        "proposal_count": int(proposal_features.shape[0]),
        "category_count": int(class_features.shape[0]),
        "proposal_feature_shape": list(proposal_features.shape),
        "material_class_similarity_shape": list(material_similarity.shape),
        "missing_bank_entry_count": sum(1 for row in class_meta if row["missing_bank_entry"]),
        "low_sample_category_count": sum(1 for row in class_meta if row.get("low_sample_count")),
        "reuse_source": str(src_feature),
        "leakage_check": bank_data.get("leakage_check", "no_full_test_bbox_used"),
    }
    out_meta.write_text(json.dumps(meta, indent=2))
    return {
        "bank_name": bank_name,
        "channel_set": bank["channel_set"],
        "feature_dim": len(bank_data["feature_names"]),
        "proposal_count": int(proposal_features.shape[0]),
        "category_count": int(class_features.shape[0]),
        "output_path": str(out_sim),
        "build_time_sec": "",
        "status": "SUCCESS_REUSED_LOCKED_PROPOSAL_FEATURES",
        "failure_reason": "",
        "sample_count_low_warning": meta["low_sample_category_count"],
        "notes": "复用 Batch17/18 locked proposal_features.pt，重新按该 bank 计算 class similarity。",
    }


def main() -> int:
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    SIM_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    test_root = default_multichannel_test_root()
    print(f"[Batch28] multichannel test root: {test_root}", flush=True)

    for bank in BANKS:
        bank_name = str(bank["bank_name"])
        print(f"[Batch28] building material similarity for {bank_name}", flush=True)
        start = time.time()
        try:
            if bank.get("reuse_cache_material"):
                row = reuse_legacy_material(bank)
                rows.append(row)
                continue
            bank_data = load_material_bank(bank["bank_path"])
            out_sim = SIM_DIR / f"{bank_name}_material_class_similarity.pt"
            out_feature = FEATURE_DIR / f"{bank_name}_proposal_material_features.pt"
            out_meta = SIM_DIR / f"{bank_name}_material_similarity_meta.json"
            if out_sim.exists() and out_meta.exists():
                meta = json.loads(out_meta.read_text())
                rows.append(
                    {
                        "bank_name": bank_name,
                        "channel_set": bank["channel_set"],
                        "feature_dim": meta.get("feature_dim", len(bank_data["feature_names"])),
                        "proposal_count": meta.get("proposal_count", ""),
                        "category_count": meta.get("category_count", ""),
                        "output_path": str(out_sim),
                        "build_time_sec": 0.0,
                        "status": "SKIPPED_REUSE",
                        "failure_reason": "",
                        "sample_count_low_warning": meta.get("low_sample_category_count", ""),
                        "notes": "已有 material similarity，未重算。",
                    }
                )
                continue
            if list(bank_data["feature_names"]) == LEGACY_COLOUR_FEATURE_NAMES:
                try:
                    row = build_colour_similarity_from_locked_features(bank)
                    row["build_time_sec"] = round(time.time() - start, 2)
                    rows.append(row)
                    continue
                except ValueError as exc:
                    print(f"[Batch28] {bank_name} cannot reuse proposal_features.pt: {exc}", flush=True)
            meta = build_material_similarity_for_bank(
                cache_dir=CACHE_DIR,
                bank_path=bank["bank_path"],
                multichannel_test_root=test_root,
                channel_hint=str(bank["channel_set"]) if str(bank["channel_set"]) in {"grey", "high", "low", "density"} else None,
                proposal_feature_output=out_feature,
                similarity_output=out_sim,
                meta_output=out_meta,
            )
            rows.append(
                {
                    "bank_name": bank_name,
                    "channel_set": bank["channel_set"],
                    "feature_dim": meta["feature_dim"],
                    "proposal_count": meta["proposal_count"],
                    "category_count": meta["category_count"],
                    "output_path": str(out_sim),
                    "build_time_sec": round(time.time() - start, 2),
                    "status": "SUCCESS",
                    "failure_reason": "",
                    "sample_count_low_warning": meta.get("low_sample_category_count", ""),
                    "notes": "使用 support/prototype bank 与 test proposal bbox 推理期特征构建。",
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "bank_name": bank_name,
                    "channel_set": bank["channel_set"],
                    "feature_dim": "",
                    "proposal_count": "",
                    "category_count": "",
                    "output_path": str(SIM_DIR / f"{bank_name}_material_class_similarity.pt"),
                    "build_time_sec": round(time.time() - start, 2),
                    "status": "FAILED",
                    "failure_reason": repr(exc),
                    "sample_count_low_warning": "",
                    "notes": "失败后未伪造 similarity。",
                }
            )
            print(f"[Batch28] FAILED {bank_name}: {exc!r}", flush=True)

    write_csv(SIM_DIR / "material_similarity_build_status.csv", rows)
    write_markdown(SIM_DIR / "material_similarity_build_status.md", rows)
    print(json.dumps(rows, indent=2), flush=True)
    return 0 if all(row["status"] in {"SUCCESS", "SKIPPED_REUSE", "REUSED_LOCKED_CACHE"} for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
