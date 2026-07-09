#!/usr/bin/env python3
"""Build a colour-only material bank from COCO GT boxes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rmdx.material import build_material_bank  # noqa: E402


DEFAULT_ANN = Path("/datasets/xray-datasets/DET-COMPASS/annotations/full_test.json")
DEFAULT_IMAGE_DIR = Path("/datasets/xray-datasets/DET-COMPASS/test")
DEFAULT_OUTPUT = Path(
    "/workspace/RAXO/outputs_rmdx/material/material_colour_bank_bbox_only.json"
)
SUMMARY_PATH = Path("/workspace/RAXO/outputs_rmdx/material/material_colour_bank_summary.json")
REPORT_PATH = Path("/workspace/RAXO/docs_rmdx/material_colour_bank_report.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build material colour-only bank")
    parser.add_argument("--annotation", default=str(DEFAULT_ANN))
    parser.add_argument("--image-dir", default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--summary-output", default=str(SUMMARY_PATH))
    parser.add_argument("--mode", choices=["bbox_only"], default="bbox_only")
    parser.add_argument("--bank-name", default="full_test")
    parser.add_argument("--target-annotation", default=None)
    parser.add_argument("--report-output", default=str(REPORT_PATH))
    return parser.parse_args()


def load_annotation(path: Path) -> dict:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"{path} is not a COCO annotation dict")
    return data


def write_report(args: argparse.Namespace, output_path: Path, summary_path: Path, summary: dict) -> None:
    dist = summary["sample_count_distribution"]
    lines = [
        "# Material Colour Bank 报告",
        "",
        "## 输入",
        "",
        f"- annotation: `{args.annotation}`",
        f"- image_dir: `{args.image_dir}`",
        f"- mode: `{args.mode}`",
        f"- bank_name: `{args.bank_name}`",
        f"- target_annotation: `{args.target_annotation}`",
        "",
        "## 输出",
        "",
        f"- material bank: `{output_path}`",
        f"- summary: `{summary_path}`",
        "",
        "## 构建方式",
        "",
        "使用输入 annotation 的 GT bbox 在 Colour X-ray 图像上提取 bbox crop colour feature，并按 category_id 求均值 prototype。若提供 target_annotation，则按 category name 映射到评估用 category_id。",
        "",
        "## 类别与样本分布",
        "",
        f"- category_count: `{summary['category_count']}`",
        f"- source_category_count: `{summary['source_category_count']}`",
        f"- target_category_count: `{summary['target_category_count']}`",
        f"- mapped_source_category_count: `{summary['mapped_source_category_count']}`",
        f"- unmapped_source_category_count: `{summary['unmapped_source_category_count']}`",
        f"- gt_annotation_count: `{summary['gt_annotation_count']}`",
        f"- used_annotation_count: `{summary['used_annotation_count']}`",
        f"- skipped_annotation_count: `{summary['skipped_annotation_count']}`",
        f"- skipped_unmapped_annotation_count: `{summary['skipped_unmapped_annotation_count']}`",
        f"- sample_count min/max/mean: `{dist['min']} / {dist['max']} / {dist['mean']}`",
        f"- sample_count histogram: `{dist['histogram']}`",
        "",
        "## Feature Names",
        "",
        "```text",
        "\n".join(summary["feature_names"]),
        "```",
        "",
        "## 重要说明",
        "",
        "如果该 bank 使用 full_test GT 构建，则只适合作为 exploratory MVP；如果使用 support/prototype split 构建，则泄漏风险低于 full_test bank，但仍需确认 split 定义和类别覆盖率。",
        "",
    ]
    report_path = Path(args.report_output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))


def main() -> int:
    args = parse_args()
    annotation_path = Path(args.annotation)
    output_path = Path(args.output)
    summary_path = Path(args.summary_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotation = load_annotation(annotation_path)
    target_annotation = load_annotation(Path(args.target_annotation)) if args.target_annotation else None
    bank, summary = build_material_bank(
        annotation,
        args.image_dir,
        mode=args.mode,
        target_annotation=target_annotation,
        bank_name=args.bank_name,
    )
    with output_path.open("w") as f:
        json.dump(bank, f, indent=2, sort_keys=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    write_report(args, output_path, summary_path, summary)
    print(json.dumps({"output": str(output_path), "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
