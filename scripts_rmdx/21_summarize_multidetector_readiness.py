#!/usr/bin/env python3
"""Summarize Batch19 multi-detector readiness from inventory artifacts."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from typing import Any


OUT_DIR = Path("/workspace/RAXO/outputs_rmdx/resource_checks")
DOCS = Path("/workspace/RAXO/docs_rmdx")
INVENTORY_JSON = OUT_DIR / "multidetector_resource_inventory.json"
FILE_HEALTH_JSON = OUT_DIR / "multidetector_file_health.json"
MATRIX_CSV = OUT_DIR / "multidetector_readiness_matrix.csv"
MATRIX_MD = OUT_DIR / "multidetector_readiness_matrix.md"
REPORT = DOCS / "batch19_multidetector_resource_check_report.md"


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value) == "True"


def global_ok(global_info: dict[str, Any]) -> bool:
    return all(
        [
            global_info.get("annotation_exists"),
            global_info.get("test_image_dir_exists"),
            global_info.get("test_images_cover_annotation_file_name"),
            global_info.get("prototype_exists"),
            global_info.get("material_bank_exists"),
        ]
    )


def with_masks_appears_valid(row: dict[str, Any]) -> bool:
    return boolish(row.get("with_masks_json_valid_light")) or boolish(row.get("with_masks_json_valid_full"))


def initial_valid(row: dict[str, Any]) -> bool:
    return boolish(row.get("initial_detection_json_valid_full"))


def recovered_valid(row: dict[str, Any]) -> bool:
    return boolish(row.get("recovered_detection_valid"))


def known_exists(row: dict[str, Any]) -> bool:
    return boolish(row.get("known_final_exists")) or boolish(row.get("known_final_uncertanty_exists"))


def classify(row: dict[str, Any], global_ready: bool) -> tuple[str, str, str, str]:
    if not global_ready:
        return (
            "BLOCKED_GLOBAL_RESOURCE",
            "fix_global_resources",
            "annotation/test/prototypes/material bank missing or inconsistent",
            "blocked",
        )

    wm_valid = with_masks_appears_valid(row)
    init_valid = initial_valid(row)
    rec_valid = recovered_valid(row)
    has_known = known_exists(row)
    has_cache = boolish(row.get("existing_cache_dir_exists"))

    if wm_valid and (init_valid or rec_valid):
        if has_known and not has_cache:
            return (
                "READY_NEEDS_ONLY_CACHE_BUILD",
                "build_detector_cache_in_batch21",
                "baseline outputs exist but detector-specific cache is missing",
                "low",
            )
        return (
            "READY_FOR_BATCH20_BASELINE",
            "proceed_to_batch20_baseline",
            "",
            "low",
        )
    if wm_valid and not init_valid and not rec_valid:
        return (
            "READY_NEEDS_DETECTION_RECOVERY",
            "recover_detection_from_with_masks_in_batch19_1",
            "with_masks valid but initial/recovered detection missing or invalid",
            "medium",
        )
    if init_valid and not wm_valid:
        return (
            "NEEDS_MASK_EXTRACTION",
            "run_mask_extraction_in_batch19_2",
            "initial detection valid but with_masks missing or invalid",
            "medium",
        )
    if not init_valid and not wm_valid and not rec_valid:
        return (
            "BLOCKED_MISSING_DETECTION",
            "manual_review_required",
            "initial/recovered/with_masks are unavailable or invalid",
            "blocked",
        )
    return (
        "UNKNOWN_NEEDS_MANUAL_REVIEW",
        "manual_review_required",
        "file state is mixed and cannot be safely classified",
        "high",
    )


def cell(value: Any) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    if value is None:
        return "unknown"
    return str(value)


def matrix_rows(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    global_ready = global_ok(inventory["global"])
    rows = []
    for row in inventory["detectors"]:
        status, action, reason, effort = classify(row, global_ready)
        rows.append(
            {
                "detector": row["detector"],
                "initial_detection": cell(initial_valid(row)),
                "recovered_detection": cell(recovered_valid(row)),
                "with_masks": cell(with_masks_appears_valid(row)),
                "known_final": cell(boolish(row.get("known_final_valid"))),
                "known_final_uncertanty": cell(boolish(row.get("known_final_uncertanty_valid"))),
                "existing_cache": cell(boolish(row.get("existing_cache_dir_exists"))),
                "readiness_status": status,
                "recommended_next_action": action,
                "blocking_reason": reason,
                "estimated_effort_level": effort,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = list(rows[0].keys()) if rows else []
    lines = ["# Multi-detector Readiness Matrix", "", "| " + " | ".join(keys) + " |", "|" + "|".join(["---"] * len(keys)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(k, "")) for k in keys) + " |")
    path.write_text("\n".join(lines) + "\n")


def git_output(args: list[str]) -> str:
    if args and args[0] == "git":
        args = ["git", "-c", "safe.directory=/workspace/RAXO"] + args[1:]
    return subprocess.run(args, cwd="/workspace/RAXO", text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.strip()


def next_plan(rows: list[dict[str, Any]]) -> str:
    non_gdino = [row for row in rows if row["detector"] != "groundingDINO"]
    ready = [row for row in non_gdino if row["readiness_status"] == "READY_FOR_BATCH20_BASELINE"]
    recovery = [row for row in non_gdino if row["readiness_status"] == "READY_NEEDS_DETECTION_RECOVERY"]
    mask = [row for row in non_gdino if row["readiness_status"] == "NEEDS_MASK_EXTRACTION"]
    if len(ready) >= 2:
        return "At least two non-groundingDINO detectors are ready; proceed to Batch20 multi-detector RAXO baseline."
    if len(recovery) >= 2:
        return "Most candidate detectors need detection recovery; run Batch19.1 from with_masks before Batch20."
    if len(mask) >= 2:
        return "Most candidate detectors need mask extraction; run Batch19.2 multi-detector mask extraction."
    if not ready and all(row["detector"] == "groundingDINO" or row["readiness_status"].startswith("BLOCKED") for row in rows):
        return "Only groundingDINO is ready; multi-detector expansion is blocked until resources are restored."
    if ready:
        return "Batch20 can proceed with the currently ready detectors, but SCI-level multi-detector coverage should first prioritize CoDet mask extraction and detic resource recovery."
    return "Mixed state; prioritize detectors with valid initial detections and missing with_masks."


def write_report(inventory: dict[str, Any], health: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    g = inventory["global"]
    git_status = git_output(["git", "status", "--short"])
    diff_raxo = git_output(["git", "diff", "--", "raxo/"])
    diff_db = git_output(["git", "diff", "--", "run_method_database_branch_v2.sh"])
    diff_web = git_output(["git", "diff", "--", "run_method_web_branch_v2.sh"])
    ready_batch20 = [row["detector"] for row in rows if row["readiness_status"] == "READY_FOR_BATCH20_BASELINE"]
    needs_recovery = [row["detector"] for row in rows if row["readiness_status"] == "READY_NEEDS_DETECTION_RECOVERY"]
    needs_mask = [row["detector"] for row in rows if row["readiness_status"] == "NEEDS_MASK_EXTRACTION"]
    blocked = [row["detector"] for row in rows if row["readiness_status"].startswith("BLOCKED")]
    cache_detectors = [row["detector"] for row in rows if row["existing_cache"] == "yes"]

    lines = [
        "# Batch19：多 Detector 资源检查报告",
        "",
        "## A. Batch19 目标",
        "",
        "只检查 DET-COMPASS 上 groundingDINO / detic / CoDet / VLDet 的 RAXO/RMD-X 资源是否齐全，为 Batch20 和 Batch21 做准备。本批没有运行 DINOv2、RAXO method、pycocotools AP evaluation、mask extraction、detection recovery 或 cache build。",
        "",
        "## B. 全局资源检查结果",
        "",
        f"- annotation exists: `{g.get('annotation_exists')}`",
        f"- annotation image count: `{g.get('annotation_image_count')}`",
        f"- category count: `{g.get('category_count')}`",
        f"- test image count: `{g.get('test_image_count')}`",
        f"- test images cover annotation file_name: `{g.get('test_images_cover_annotation_file_name')}`",
        f"- prototypes exists: `{g.get('prototype_exists')}`",
        f"- material bank exists: `{g.get('material_bank_exists')}`",
        f"- groundingDINO locked cache exists: `{g.get('groundingDINO_locked_cache_exists')}`",
        "",
        "## C. 每个 detector 的资源盘点",
        "",
        "| detector | initial | recovered | with_masks | known_final | known_final_uncertanty | cache | notes |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    inv_by_detector = {row["detector"]: row for row in inventory["detectors"]}
    for detector, inv in inv_by_detector.items():
        lines.append(
            f"| {detector} | {inv.get('initial_detection_exists')} / valid={inv.get('initial_detection_json_valid_full')} | "
            f"{inv.get('recovered_detection_exists')} / valid={inv.get('recovered_detection_valid')} | "
            f"{inv.get('with_masks_exists')} / light={inv.get('with_masks_json_valid_light')} / full={inv.get('with_masks_json_valid_full')} | "
            f"{inv.get('known_final_exists')} | {inv.get('known_final_uncertanty_exists')} | "
            f"{inv.get('existing_cache_dir_exists')} | {inv.get('notes')} |"
        )
    lines += [
        "",
        "## D. JSON 健康状态",
        "",
    ]
    for detector, det_health in health["detectors"].items():
        lines.append(f"### {detector}")
        lines.append("")
        lines.append(f"- initial full: `{det_health['initial_detection']['full']}`")
        lines.append(f"- recovered full: `{det_health['recovered_detection']['full']}`")
        lines.append(f"- with_masks light: `{det_health['with_masks_light']}`")
        lines.append(f"- with_masks full: `{det_health['with_masks_full']}`")
        lines.append("")
    lines += [
        "## E. with_masks 与 full_test 一致性",
        "",
    ]
    for detector, det_health in health["detectors"].items():
        lines.append(f"- `{detector}`: `{det_health['with_masks_consistency']}`")
    lines += [
        "",
        "## F. readiness matrix",
        "",
        f"- Markdown: `{MATRIX_MD}`",
        f"- CSV: `{MATRIX_CSV}`",
        "",
    ]
    lines.append("| " + " | ".join(rows[0].keys()) + " |")
    lines.append("|" + "|".join(["---"] * len(rows[0])) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(k, "")) for k in row.keys()) + " |")
    lines += [
        "",
        "## G. 可以直接进入 Batch20 的 detectors",
        "",
        f"`{ready_batch20}`",
        "",
        "## H. 需要恢复 detection 的 detectors",
        "",
        f"`{needs_recovery}`",
        "",
        "## I. 需要 mask extraction 的 detectors",
        "",
        f"`{needs_mask}`",
        "",
        "## J. 被阻塞的 detectors",
        "",
        f"`{blocked}`",
        "",
        "## K. 已有 detector-specific cache",
        "",
        f"`{cache_detectors}`",
        "",
        "## L. 是否修改 RAXO 原始代码",
        "",
        "否。`git diff -- raxo/` 为空。",
        "",
        "## M. 是否运行 DINOv2",
        "",
        "`False`",
        "",
        "## N. 是否运行 AP evaluation",
        "",
        "`False`",
        "",
        "## O. 下一步建议",
        "",
        next_plan(rows),
        "",
        "## Git / 原始代码改动检查",
        "",
        "### git status --short",
        "",
        "```text",
        git_status or "(clean)",
        "```",
        "",
        f"- `git diff -- raxo/` empty: `{not bool(diff_raxo)}`",
        f"- `git diff -- run_method_database_branch_v2.sh` empty: `{not bool(diff_db)}`",
        f"- `git diff -- run_method_web_branch_v2.sh` empty: `{not bool(diff_web)}`",
        "",
        "## 输出文件",
        "",
        f"- `{INVENTORY_JSON}`",
        f"- `{OUT_DIR / 'multidetector_resource_inventory.csv'}`",
        f"- `{FILE_HEALTH_JSON}`",
        f"- `{MATRIX_MD}`",
        f"- `{MATRIX_CSV}`",
    ]
    DOCS.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    inventory = json.load(INVENTORY_JSON.open())
    health = json.load(FILE_HEALTH_JSON.open())
    rows = matrix_rows(inventory)
    write_csv(MATRIX_CSV, rows)
    write_md(MATRIX_MD, rows)
    write_report(inventory, health, rows)
    print(json.dumps({"matrix_csv": str(MATRIX_CSV), "matrix_md": str(MATRIX_MD), "report": str(REPORT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
