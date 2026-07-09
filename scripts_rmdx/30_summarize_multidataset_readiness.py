#!/usr/bin/env python3
"""Summarize Batch22 multidataset readiness."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path("/workspace/RAXO")
OUT_DIR = ROOT / "outputs_rmdx" / "resource_checks"
DOC_DIR = ROOT / "docs_rmdx"
INV = OUT_DIR / "multidataset_resource_inventory.json"
MATRIX_CSV = OUT_DIR / "multidataset_readiness_matrix.csv"
MATRIX_MD = OUT_DIR / "multidataset_readiness_matrix.md"
REPORT = DOC_DIR / "batch22_multidataset_resource_check_report.md"

DATASETS = ["DET-COMPASS", "PIXray", "pidray", "CLCXray", "HiXray", "DvXray"]

FIELDS = [
    "dataset",
    "annotation_status",
    "test_image_status",
    "image_count",
    "category_count",
    "groundingDINO_initial",
    "groundingDINO_with_masks",
    "groundingDINO_known_final_uncertanty",
    "raxo_groundingDINO_readiness",
    "material_bank_readiness",
    "recommended_next_action",
    "blocking_reason",
    "estimated_effort_level",
    "notes",
]


def yesno(value: Any) -> str:
    return "yes" if value else "no"


def det_map(resources: list[dict[str, Any]], dataset: str, detector: str) -> dict[str, Any]:
    return next((r for r in resources if r["dataset"] == dataset and r["detector"] == detector), {})


def material_map(rows: list[dict[str, Any]], dataset: str) -> dict[str, Any]:
    return next((r for r in rows if r["dataset"] == dataset), {})


def rmdx_material_status(raw: str) -> str:
    if raw == "READY_EXISTING_BANK":
        return "READY_EXISTING_MATERIAL_BANK"
    if raw == "READY_CAN_BUILD_FROM_SUPPORT":
        return "READY_FOR_BATCH24_MATERIAL_BANK"
    if raw == "NEEDS_SUPPORT_MAPPING":
        return "NEEDS_SUPPORT_BANK_CONSTRUCTION"
    if raw == "ONLY_TRANSFER_BANK_AVAILABLE":
        return "NEEDS_TRANSFER_BANK_DESIGN"
    if raw == "BLOCKED_NO_SUPPORT_SOURCE":
        return "BLOCKED_NO_SUPPORT_SOURCE"
    return "UNKNOWN_NEEDS_MANUAL_REVIEW"


def decide_row(ds: str, inv: dict[str, Any]) -> dict[str, Any]:
    dinfo = inv["datasets"][ds]
    gd = det_map(inv["detector_resources"], ds, "groundingDINO")
    mat = material_map(inv["material_bank_readiness"], ds)
    cov = inv["prototype_coverage"].get(ds, {})
    annotation_ok = dinfo.get("annotation_exists") and dinfo.get("annotation_valid")
    images_ok = dinfo.get("test_dir_exists") and dinfo.get("test_images_cover_annotation_file_name")
    proto_ok = inv["prototypes"].get("exists") and inv["prototypes"].get("can_load_light")
    coverage_ratio = cov.get("coverage_ratio")
    proto_coverage_ok = coverage_ratio is not None and coverage_ratio >= 0.8
    initial_ok = bool(gd.get("initial_detection_json_valid_full") or gd.get("recovered_detection_valid"))
    initial_missing_or_bad = not initial_ok
    with_masks_ok = bool(gd.get("with_masks_json_valid_light") or gd.get("known_final_uncertanty_valid"))
    known_unc_ok = bool(gd.get("known_final_uncertanty_valid"))

    if known_unc_ok and annotation_ok and images_ok:
        readiness = "READY_EVALUATE_EXISTING_KNOWN_FINAL"
        action = "evaluate_existing_known_final_in_batch23"
        block = "none"
        effort = "low"
    elif not annotation_ok:
        readiness = "BLOCKED_MISSING_ANNOTATION"
        action = "exclude_from_multidataset_main_table"
        block = "annotation missing or invalid"
        effort = "blocked"
    elif not images_ok:
        readiness = "BLOCKED_MISSING_TEST_IMAGES"
        action = "restore_test_images_before_batch23"
        block = "test images missing or do not cover annotation file_name"
        effort = "blocked"
    elif not proto_ok or not proto_coverage_ok:
        readiness = "BLOCKED_PROTOTYPE_COVERAGE_LOW"
        action = "manual_review_required"
        block = f"prototype coverage low or unavailable: {coverage_ratio}"
        effort = "medium"
    elif initial_ok and with_masks_ok:
        readiness = "READY_FOR_BATCH23_RAXO_BASELINE"
        action = "proceed_to_batch23_raxo_baseline"
        block = "none"
        effort = "low"
    elif gd.get("with_masks_json_valid_light") and initial_missing_or_bad:
        readiness = "READY_NEEDS_DETECTION_RECOVERY"
        action = "recover_detection_from_with_masks"
        block = "initial/recovered detection missing or invalid"
        effort = "medium"
    elif gd.get("initial_detection_json_valid_full") and not gd.get("with_masks_json_valid_light"):
        readiness = "NEEDS_MASK_EXTRACTION"
        action = "run_mask_extraction"
        block = "with_masks missing or invalid"
        effort = "high"
    elif not initial_ok and not gd.get("with_masks_json_valid_light"):
        readiness = "BLOCKED_MISSING_DETECTION"
        action = "exclude_from_multidataset_main_table"
        block = "groundingDINO initial/recovered/with_masks unavailable"
        effort = "blocked"
    else:
        readiness = "UNKNOWN_NEEDS_MANUAL_REVIEW"
        action = "manual_review_required"
        block = "resource state ambiguous"
        effort = "medium"

    material_status = rmdx_material_status(mat.get("material_bank_readiness_status", ""))
    if readiness.startswith("READY") and material_status in {"NEEDS_TRANSFER_BANK_DESIGN", "BLOCKED_NO_SUPPORT_SOURCE"}:
        notes = "RAXO may proceed, but RMD-X material bank needs design."
    else:
        notes = ""
    return {
        "dataset": ds,
        "annotation_status": "valid" if annotation_ok else "missing_or_invalid",
        "test_image_status": "complete" if images_ok else "missing_or_incomplete",
        "image_count": dinfo.get("image_count"),
        "category_count": dinfo.get("category_count"),
        "groundingDINO_initial": yesno(gd.get("initial_detection_json_valid_full") or gd.get("recovered_detection_valid")),
        "groundingDINO_with_masks": yesno(gd.get("with_masks_json_valid_light")),
        "groundingDINO_known_final_uncertanty": yesno(gd.get("known_final_uncertanty_valid")),
        "raxo_groundingDINO_readiness": readiness,
        "material_bank_readiness": material_status,
        "recommended_next_action": action,
        "blocking_reason": block,
        "estimated_effort_level": effort,
        "notes": notes,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = ["# Multidataset Readiness Matrix", "", "| " + " | ".join(FIELDS) + " |", "|" + "|".join(["---"] * len(FIELDS)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in FIELDS) + " |")
    path.write_text("\n".join(lines) + "\n")


def run_cmd(args: list[str]) -> str:
    proc = subprocess.run(args, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    return proc.stdout.strip()


def detector_summary(inv: dict[str, Any], detector: str) -> str:
    parts = []
    for ds in DATASETS:
        r = det_map(inv["detector_resources"], ds, detector)
        parts.append(
            f"{ds}: initial={yesno(r.get('initial_detection_exists'))}, "
            f"with_masks={yesno(r.get('with_masks_exists'))}, "
            f"known_unc={yesno(r.get('known_final_uncertanty_exists'))}"
        )
    return "; ".join(parts)


def write_report(rows: list[dict[str, Any]], inv: dict[str, Any]) -> None:
    ready = [r["dataset"] for r in rows if r["raxo_groundingDINO_readiness"] in {"READY_FOR_BATCH23_RAXO_BASELINE", "READY_EVALUATE_EXISTING_KNOWN_FINAL"}]
    need_recovery = [r["dataset"] for r in rows if r["raxo_groundingDINO_readiness"] == "READY_NEEDS_DETECTION_RECOVERY"]
    need_masks = [r["dataset"] for r in rows if r["raxo_groundingDINO_readiness"] == "NEEDS_MASK_EXTRACTION"]
    material_design = [r["dataset"] for r in rows if r["material_bank_readiness"] in {"NEEDS_TRANSFER_BANK_DESIGN", "NEEDS_SUPPORT_BANK_CONSTRUCTION", "BLOCKED_NO_SUPPORT_SOURCE"}]
    blocked = [r["dataset"] for r in rows if r["raxo_groundingDINO_readiness"].startswith("BLOCKED")]

    if len(ready) >= 3:
        next_suggestion = "Proceed to Batch23 multi-dataset RAXO baseline."
    elif ready == ["DET-COMPASS"]:
        next_suggestion = (
            "Only DET-COMPASS is ready; current multi-dataset expansion is blocked by missing test images "
            "for PIXray, pidray, CLCXray, HiXray, and DvXray."
        )
    elif need_recovery:
        next_suggestion = "Run Batch22.1 detection recovery before Batch23."
    elif need_masks:
        next_suggestion = "Run Batch22.2 multi-dataset mask extraction before Batch23."
    elif ready and material_design:
        next_suggestion = "Run Batch23 RAXO baseline for ready datasets while designing material banks for Batch24."
    else:
        next_suggestion = "Current multi-dataset expansion is blocked; restore missing dataset resources first."

    git_base = ["git", "-c", f"safe.directory={ROOT}"]
    git_status = run_cmd(git_base + ["status", "--short"])
    diff_raxo = run_cmd(git_base + ["diff", "--", "raxo/"])
    diff_db = run_cmd(git_base + ["diff", "--", "run_method_database_branch_v2.sh"])
    diff_web = run_cmd(git_base + ["diff", "--", "run_method_web_branch_v2.sh"])

    lines = [
        "# Batch22 Multidataset Resource Check Report",
        "",
        "## Goal",
        "",
        "Check official RAXO multi-dataset resources for future Batch23/24/25 runs. No experiment was run.",
        "",
        "## Datasets",
        "",
        "`" + "`, `".join(DATASETS) + "`",
        "",
        "## Annotation and Test Images",
        "",
        "| dataset | annotation | images | ann images | test files | missing images | categories |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for ds in DATASETS:
        d = inv["datasets"][ds]
        lines.append(
            f"| {ds} | {d.get('annotation_valid')} | {d.get('test_images_cover_annotation_file_name')} | "
            f"{d.get('image_count')} | {d.get('test_image_count')} | {d.get('missing_images_count')} | {d.get('category_count')} |"
        )
    lines.extend(
        [
            "",
            "Note: `images=False` here means the `test/` directory does not cover annotation `file_name` entries. "
            "It is the primary blocker for non-DET-COMPASS datasets in this check.",
        ]
    )
    lines.extend(["", "## Prototype Coverage", "", "| dataset | coverage | matched | missing | missing sample |", "|---|---:|---:|---:|---|"])
    for ds in DATASETS:
        c = inv["prototype_coverage"].get(ds, {})
        lines.append(
            f"| {ds} | {c.get('coverage_ratio')} | {c.get('matched_category_count')} | "
            f"{c.get('missing_category_count')} | `{c.get('missing_category_names_sample')}` |"
        )
    lines.extend(["", "## groundingDINO Resources", "", "| dataset | initial | with_masks | known_final_uncertanty |", "|---|---|---|---|"])
    for ds in DATASETS:
        gd = det_map(inv["detector_resources"], ds, "groundingDINO")
        lines.append(
            f"| {ds} | {gd.get('initial_detection_exists')} | {gd.get('with_masks_exists')} | {gd.get('known_final_uncertanty_exists')} |"
        )
    lines.extend(
        [
            "",
            "Resource columns above report file existence. The readiness matrix uses stricter validity checks "
            "where feasible; very large JSON files may have full validation skipped and are recorded in "
            "`multidataset_file_health.json`.",
        ]
    )
    lines.extend(["", "## VLDet / CoDet Auxiliary Summary", "", f"- VLDet: {detector_summary(inv, 'VLDet')}", f"- CoDet: {detector_summary(inv, 'CoDet')}"])
    lines.extend(["", "## Material Bank Readiness", "", "| dataset | status | source | coverage |", "|---|---|---|---:|"])
    for row in inv["material_bank_readiness"]:
        lines.append(
            f"| {row['dataset']} | {row['material_bank_readiness_status']} | `{row['possible_support_source']}` | {row['expected_category_coverage']} |"
        )
    lines.extend(["", "## Readiness Matrix", "", MATRIX_MD.read_text()])
    lines.extend(
        [
            "",
            "## Planning Buckets",
            "",
            f"- can enter Batch23: `{', '.join(ready) if ready else 'none'}`",
            f"- need detection recovery: `{', '.join(need_recovery) if need_recovery else 'none'}`",
            f"- need mask extraction: `{', '.join(need_masks) if need_masks else 'none'}`",
            f"- need material bank design: `{', '.join(material_design) if material_design else 'none'}`",
            f"- currently blocked: `{', '.join(blocked) if blocked else 'none'}`",
            "",
            "## Constraints Audit",
            "",
            "- modified RAXO source code: `no`",
            "- modified official run scripts: `no`",
            "- ran DINOv2: `no`",
            "- ran AP evaluation: `no`",
            "- ran RAXO method: `no`",
            "- ran mask extraction: `no`",
            "- built cache: `no`",
            "- used full_test material bank: `no`",
            "",
            "## Git / Original Code Diff Check",
            "",
            "### git status --short",
            "```text",
            git_status,
            "```",
            "",
            "### git diff -- raxo/",
            "```text",
            diff_raxo,
            "```",
            "",
            "### git diff -- run_method_database_branch_v2.sh",
            "```text",
            diff_db,
            "```",
            "",
            "### git diff -- run_method_web_branch_v2.sh",
            "```text",
            diff_web,
            "```",
            "",
            "## Next Recommendation",
            "",
            next_suggestion,
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    inv = json.load(INV.open())
    rows = [decide_row(ds, inv) for ds in DATASETS]
    write_csv(MATRIX_CSV, rows)
    write_md(MATRIX_MD, rows)
    write_report(rows, inv)
    print(f"matrix_csv={MATRIX_CSV}")
    print(f"matrix_md={MATRIX_MD}")
    print(f"report={REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
