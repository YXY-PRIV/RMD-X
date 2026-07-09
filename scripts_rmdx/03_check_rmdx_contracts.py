#!/usr/bin/env python3
"""Smoke-check RMD-X data contracts without modifying RAXO baseline files."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rmdx.io.coco_io import summarize_coco_dataset
from rmdx.io.paths import default_paths
from rmdx.io.prototype_io import summarize_prototypes
from rmdx.io.result_io import compare_result_counts, summarize_result_list
from rmdx.validation.validators import validate_contracts


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def ok_text(ok: bool) -> str:
    return "OK" if ok else "FAIL"


def main() -> int:
    paths = default_paths()
    paths.contracts_dir.mkdir(parents=True, exist_ok=True)
    paths.smoke_dir.mkdir(parents=True, exist_ok=True)
    paths.rmdx_results_dir.mkdir(parents=True, exist_ok=True)
    paths.docs_dir.mkdir(parents=True, exist_ok=True)

    prototype_summary = summarize_prototypes(paths.prototypes_path)
    annotation_summary = summarize_coco_dataset(paths.det_compass_annotation)
    with_masks_summary = summarize_coco_dataset(paths.with_masks_path, reference_path=paths.det_compass_annotation)
    known_final_summary = summarize_result_list(paths.known_final_path)
    known_final_uncertanty_summary = summarize_result_list(paths.known_final_uncertanty_path)
    comparison = compare_result_counts(known_final_summary, known_final_uncertanty_summary)

    write_json(paths.contracts_dir / "prototype_summary.json", prototype_summary)
    write_json(paths.contracts_dir / "annotation_summary.json", annotation_summary)
    write_json(paths.contracts_dir / "with_masks_summary.json", with_masks_summary)
    write_json(paths.contracts_dir / "known_final_summary.json", known_final_summary)
    write_json(paths.contracts_dir / "known_final_uncertanty_summary.json", known_final_uncertanty_summary)
    write_json(paths.contracts_dir / "known_final_to_uncertanty_comparison.json", comparison)

    checks = validate_contracts(
        prototype_summary=prototype_summary,
        annotation_summary=annotation_summary,
        with_masks_summary=with_masks_summary,
        known_final_summary=known_final_summary,
        image_dir=paths.det_compass_image_dir,
        annotation_path=paths.det_compass_annotation,
    )
    write_json(paths.contracts_dir / "validator_results.json", checks)

    new_files = [
        "rmdx/__init__.py",
        "rmdx/schemas.py",
        "rmdx/io/__init__.py",
        "rmdx/io/paths.py",
        "rmdx/io/coco_io.py",
        "rmdx/io/prototype_io.py",
        "rmdx/io/result_io.py",
        "rmdx/validation/__init__.py",
        "rmdx/validation/validators.py",
        "rmdx/docs/README.md",
        "scripts_rmdx/03_check_rmdx_contracts.py",
    ]

    report = []
    report.append("# Batch7：RMD-X 数据契约与外挂结构报告\n")
    report.append("## A. RMD-X 目录结构\n")
    report.append("```text")
    report.append("rmdx/")
    report.append("  __init__.py")
    report.append("  schemas.py")
    report.append("  io/")
    report.append("    __init__.py")
    report.append("    paths.py")
    report.append("    coco_io.py")
    report.append("    prototype_io.py")
    report.append("    result_io.py")
    report.append("  validation/")
    report.append("    __init__.py")
    report.append("    validators.py")
    report.append("  docs/")
    report.append("    README.md")
    report.append("scripts_rmdx/")
    report.append("  03_check_rmdx_contracts.py")
    report.append("outputs_rmdx/")
    report.append("  contracts/")
    report.append("  smoke/")
    report.append("  rmdx_results/")
    report.append("```\n")

    report.append("## B. 新增文件路径\n")
    for rel in new_files:
        report.append(f"- `{paths.repo_root / rel}`")
    report.append("")

    report.append("## C. Prototype Summary\n")
    report.append(f"- path: `{prototype_summary['path']}`")
    report.append(f"- prototype_count: `{prototype_summary['prototype_count']}`")
    report.append(f"- supercategory_count: `{prototype_summary['supercategory_count']}`")
    report.append(f"- feature_dim: `{prototype_summary['feature_dim']}`")
    report.append(f"- negative_prototype_count: `{prototype_summary['negative_prototype_count']}`")
    report.append(f"- output: `{paths.contracts_dir / 'prototype_summary.json'}`\n")

    report.append("## D. with_masks Summary\n")
    report.append(f"- path: `{with_masks_summary['path']}`")
    report.append(f"- images: `{with_masks_summary['image_count']}`")
    report.append(f"- annotations: `{with_masks_summary['annotation_count']}`")
    report.append(f"- categories: `{with_masks_summary['category_count']}`")
    report.append(f"- has_mask_field: `{with_masks_summary['has_mask_field']}`")
    report.append(f"- annotations_per_image: `{with_masks_summary['annotations_per_image']}`")
    report.append(f"- output: `{paths.contracts_dir / 'with_masks_summary.json'}`\n")

    report.append("## E. full_test Annotation Summary\n")
    report.append(f"- path: `{annotation_summary['path']}`")
    report.append(f"- images: `{annotation_summary['image_count']}`")
    report.append(f"- annotations: `{annotation_summary['annotation_count']}`")
    report.append(f"- categories: `{annotation_summary['category_count']}`")
    report.append(f"- output: `{paths.contracts_dir / 'annotation_summary.json'}`\n")

    report.append("## F. known_final Summary\n")
    report.append(f"- path: `{known_final_summary['path']}`")
    report.append(f"- detections: `{known_final_summary['detection_count']}`")
    report.append(f"- images: `{known_final_summary['image_count']}`")
    report.append(f"- category_ids: `{known_final_summary['category_id_count']}`")
    report.append(f"- score_range: `{known_final_summary['score_range']}`")
    report.append(f"- dist_cosine_inter_range: `{known_final_summary['dist_cosine_inter_range']}`")
    report.append(f"- output: `{paths.contracts_dir / 'known_final_summary.json'}`\n")

    report.append("## G. known_final_uncertanty Summary\n")
    report.append(f"- path: `{known_final_uncertanty_summary['path']}`")
    report.append(f"- detections: `{known_final_uncertanty_summary['detection_count']}`")
    report.append(f"- images: `{known_final_uncertanty_summary['image_count']}`")
    report.append(f"- category_ids: `{known_final_uncertanty_summary['category_id_count']}`")
    report.append(f"- score_range: `{known_final_uncertanty_summary['score_range']}`")
    report.append(f"- dist_cosine_inter_range: `{known_final_uncertanty_summary['dist_cosine_inter_range']}`")
    report.append(f"- count_change: `{comparison}`")
    report.append(f"- output: `{paths.contracts_dir / 'known_final_uncertanty_summary.json'}`\n")

    report.append("## H. Validator 检查结果\n")
    for item in checks:
        report.append(f"- {ok_text(item['ok'])}: `{item['name']}` - {item['detail']}")
    all_ok = all(item["ok"] for item in checks)
    report.append(f"\nvalidator overall: `{ok_text(all_ok)}`\n")

    report.append("## I. 当前是否可以进入 Batch8\n")
    report.append("可以进入 Batch8。" if all_ok else "暂不建议进入 Batch8，需先修复 validator 失败项。")
    report.append("")

    report.append("## J. Batch8 建议任务\n")
    report.append("- 创建 RMD-X wrapper smoke，不实现真实算法，只复制 known_final 并追加占位字段。")
    report.append("- 定义 reliability/material/risk sidecar JSON schema。")
    report.append("- 为 5-image smoke 构造 RMD-X 输出路径，不覆盖 RAXO baseline。")
    report.append("- 增加 `cocoapi_2.py` 评估 wrapper 输出的 smoke 检查。")
    report.append("")

    report.append("## K. 是否修改 RAXO 原始代码\n")
    report.append("否。未修改 `raxo/*.py`、`run_method_database_branch_v2.sh`、`run_method_web_branch_v2.sh`。")
    report.append("本批次只新增 `rmdx/`、`scripts_rmdx/03_check_rmdx_contracts.py`、`outputs_rmdx/contracts/` summary 和本报告。")
    report.append("")

    report_path = paths.docs_dir / "batch7_rmdx_contracts_report.md"
    report_path.write_text("\n".join(report) + "\n")
    print(f"wrote {report_path}")
    print(f"validator overall: {ok_text(all_ok)}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

