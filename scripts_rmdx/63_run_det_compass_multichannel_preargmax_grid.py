#!/usr/bin/env python3
"""Run Batch28 DET-COMPASS multi-channel pre-argmax ablation grid."""

from __future__ import annotations

import csv
import gc
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rmdx.cache_fusion import (
    detections_from_fusion,
    fuse_from_cache,
    load_json,
    load_jsonl,
    material_class_to_prototype,
    save_json,
)
from rmdx.preargmax_fusion import (
    fuse_similarity_material,
    fuse_similarity_material_reliability,
    fuse_similarity_visual_only,
)


CACHE_DIR = REPO_ROOT / "outputs_rmdx/cache/det_compass_groundingDINO_100_0_full"
SIM_DIR = REPO_ROOT / "outputs_rmdx/material_multichannel_det_compass/material_similarity"
OUT_ROOT = REPO_ROOT / "outputs_rmdx/preargmax_multichannel_det_compass"
RESULT_DIR = REPO_ROOT / "outputs_rmdx/rmdx_results"
LOG_DIR = REPO_ROOT / "outputs_rmdx/logs"
ANNOTATION = Path("/datasets/xray-datasets/DET-COMPASS/annotations/full_test.json")
if not ANNOTATION.exists():
    ANNOTATION = Path("/mnt/data/raxo/datasets/xray-datasets/DET-COMPASS/annotations/full_test.json")

RAXO_BASELINE = {"AP": 0.480, "AP50": 0.544, "AP75": 0.488}
LEGACY_BEST_AP = 0.501

BANKS = [
    ("legacy_colour_bank", "colour", 22, "legacy reference"),
    ("new_colour_bank", "colour", 22, "Batch27 controlled rebuild"),
    ("grey_bank", "grey", 18, "single channel"),
    ("high_bank", "high", 18, "single channel"),
    ("low_bank", "low", 18, "single channel"),
    ("density_bank", "density", 18, "single channel"),
    ("all_channel_bank", "colour+grey+high+low+density+cross", 110, "multi-channel"),
]

MATERIAL_CONFIGS = [
    ("material_preargmax", 0.01, 0.0),
    ("material_preargmax", 0.02, 0.0),
    ("material_preargmax", 0.03, 0.0),
    ("material_preargmax", 0.05, 0.0),
    ("material_preargmax", 0.07, 0.0),
    ("material_preargmax", 0.10, 0.0),
    ("material_reliability_preargmax", 0.03, 0.25),
    ("material_reliability_preargmax", 0.03, 0.50),
    ("material_reliability_preargmax", 0.05, 0.25),
    ("material_reliability_preargmax", 0.05, 0.50),
    ("material_reliability_preargmax", 0.07, 0.25),
    ("material_reliability_preargmax", 0.07, 0.50),
]


def slug(mode: str, lm: float, lr: float) -> str:
    return f"{mode}_lm{lm:.2f}_lr{lr:.2f}".replace(".", "p")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    lines = [
        f"# {title}",
        "",
        "说明：表中 AP 来自 pycocotools 实际评估日志；本批次不重跑 DINOv2、不重建 visual cache、不使用 full_test GT bbox 构建 bank。",
        "",
    ]
    if rows:
        headers = list(rows[0].keys())
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in rows:
            lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    path.write_text("\n".join(lines) + "\n")


def parse_ap(text: str) -> tuple[float | None, float | None, float | None]:
    ap = ap50 = ap75 = None
    for line in text.splitlines():
        if "Average Precision" not in line:
            continue
        match = re.search(r"\]\s*=\s*([-+]?\d+\.\d+)", line)
        if not match:
            continue
        value = float(match.group(1))
        if "IoU=0.50:0.95" in line and "area=   all" in line and "maxDets=100" in line and ap is None:
            ap = value
        elif "IoU=0.50      " in line and "area=   all" in line and ap50 is None:
            ap50 = value
        elif "IoU=0.75      " in line and "area=   all" in line and ap75 is None:
            ap75 = value
    return ap, ap50, ap75


def run_cmd(cmd: list[str], log_path: Path) -> tuple[int, str]:
    proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(proc.stdout)
    return proc.returncode, proc.stdout


def count_json_list(path: Path) -> int:
    with path.open() as f:
        data = json.load(f)
    return len(data) if isinstance(data, list) else -1


def stats_from_existing_detection(path: Path) -> dict[str, Any]:
    with path.open() as f:
        detections = json.load(f)
    changed_category_count = sum(
        1
        for det in detections
        if det.get("rmdx_original_category_id") is not None
        and int(det["category_id"]) != int(det["rmdx_original_category_id"])
    )
    return {
        "detection_count_before_uncertanty": len(detections),
        "changed_category_count": changed_category_count,
        "changed_category_ratio": changed_category_count / max(len(detections), 1),
    }


def eval_detection(output_path: Path, eval_tag: str) -> tuple[bool, Path, int, float | None, float | None, float | None, str]:
    uncert_path = Path(str(output_path).replace(".json", "_uncertanty.json"))
    uncert_log = LOG_DIR / f"batch28_uncertanty_{eval_tag}.log"
    eval_log = LOG_DIR / f"batch28_eval_{eval_tag}.log"
    if uncert_path.exists() and eval_log.exists():
        after_count = count_json_list(uncert_path)
        out = eval_log.read_text()
        ap, ap50, ap75 = parse_ap(out)
        ok = ap is not None and ap50 is not None and ap75 is not None
        return ok, uncert_path, after_count, ap, ap50, ap75, "" if ok else out[-4000:]
    code, out = run_cmd(["python3", "raxo/uncertanty_estimation_fixed.py", "--dets", str(output_path)], uncert_log)
    if code != 0:
        return False, uncert_path, -1, None, None, None, out[-4000:]
    after_count = count_json_list(uncert_path) if uncert_path.exists() else -1
    code, out = run_cmd(
        ["python3", "raxo/cocoapi_2.py", "--cocoGt", str(ANNOTATION), "--cocoDt", str(uncert_path)],
        eval_log,
    )
    if code != 0:
        return False, uncert_path, after_count, None, None, None, out[-4000:]
    ap, ap50, ap75 = parse_ap(out)
    ok = ap is not None and ap50 is not None and ap75 is not None
    return ok, uncert_path, after_count, ap, ap50, ap75, "" if ok else out[-4000:]


def load_base_cache() -> dict[str, Any]:
    return {
        "cache_dir": CACHE_DIR,
        "proposal_meta": load_jsonl(CACHE_DIR / "proposal_meta.jsonl"),
        "prototype_meta": load_json(CACHE_DIR / "prototype_meta.json"),
        "category_meta": load_json(CACHE_DIR / "category_meta.json"),
        "summary": load_json(CACHE_DIR / "cache_summary.json"),
        "visual_similarity": torch.load(CACHE_DIR / "visual_similarity.pt", map_location="cpu").float(),
    }


def material_proto_for_bank(bank_name: str, prototype_meta: list[dict[str, Any]]) -> torch.Tensor:
    path = SIM_DIR / f"{bank_name}_material_class_similarity.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    material_class = torch.load(path, map_location="cpu").float()
    return material_class_to_prototype(material_class, prototype_meta).float()


def generate_detections(
    cache: dict[str, Any],
    material_proto: torch.Tensor,
    *,
    mode: str,
    lambda_material: float,
    lambda_reliability: float,
    output_path: Path,
) -> dict[str, Any]:
    visual = cache["visual_similarity"].float()
    if mode == "visual_only":
        cache_for_output = dict(cache)
        cache_for_output["material_class_similarity"] = torch.zeros(
            (len(cache["proposal_meta"]), len(cache["category_meta"])), dtype=torch.float32
        )
        fused, _, gate = fuse_from_cache(
            cache_for_output,
            mode=mode,
            lambda_material=lambda_material,
            lambda_reliability=lambda_reliability,
        )
        material_for_output = torch.zeros_like(fused)
    else:
        detector_scores = torch.tensor(
            [float(row["detector_score"]) for row in cache["proposal_meta"]],
            dtype=torch.float32,
        )
        if mode == "material_preargmax":
            gate = torch.ones(visual.shape[0], dtype=torch.float32)
            fused = fuse_similarity_material(visual, material_proto, lambda_material)
        else:
            fused, gate = fuse_similarity_material_reliability(
                visual,
                material_proto,
                detector_scores,
                lambda_material,
                lambda_reliability,
            )
        material_for_output = material_proto
    detections = detections_from_fusion(
        {
            "visual_similarity": visual,
            "proposal_meta": cache["proposal_meta"],
            "prototype_meta": cache["prototype_meta"],
            "category_meta": cache["category_meta"],
        },
        fused.float(),
        material_for_output.float(),
        gate.float(),
        mode=mode,
        lambda_material=lambda_material,
        lambda_reliability=lambda_reliability,
    )
    save_json(output_path, detections)
    changed_category_count = sum(
        1
        for det in detections
        if det.get("rmdx_original_category_id") is not None
        and int(det["category_id"]) != int(det["rmdx_original_category_id"])
    )
    return {
        "detection_count_before_uncertanty": len(detections),
        "changed_category_count": changed_category_count,
        "changed_category_ratio": changed_category_count / max(len(detections), 1),
    }


def result_row(
    *,
    bank_name: str,
    channel_set: str,
    feature_dim: int,
    mode: str,
    lm: float,
    lr: float,
    output_path: Path,
    status: str,
    failure_reason: str = "",
    before_count: int | str = "",
    after_count: int | str = "",
    changed_count: int | str = "",
    changed_ratio: float | str = "",
    ap: float | None = None,
    ap50: float | None = None,
    ap75: float | None = None,
) -> dict[str, Any]:
    return {
        "bank_name": bank_name,
        "channel_set": channel_set,
        "feature_dim": feature_dim,
        "mode": mode,
        "lambda_material": lm,
        "lambda_reliability": lr,
        "AP": "" if ap is None else round(ap, 3),
        "AP50": "" if ap50 is None else round(ap50, 3),
        "AP75": "" if ap75 is None else round(ap75, 3),
        "delta_AP_vs_raxo": "" if ap is None else round(ap - RAXO_BASELINE["AP"], 3),
        "delta_AP50_vs_raxo": "" if ap50 is None else round(ap50 - RAXO_BASELINE["AP50"], 3),
        "delta_AP75_vs_raxo": "" if ap75 is None else round(ap75 - RAXO_BASELINE["AP75"], 3),
        "delta_AP_vs_legacy_colour_best": "" if ap is None else round(ap - LEGACY_BEST_AP, 3),
        "detection_count_before_uncertanty": before_count,
        "detection_count_after_uncertanty": after_count,
        "changed_category_count": changed_count,
        "changed_category_ratio": "" if changed_ratio == "" else round(float(changed_ratio), 6),
        "low_sample_warning": "91 categories sample_count < 3",
        "output_path": str(output_path),
        "uncertanty_output_path": str(output_path).replace(".json", "_uncertanty.json"),
        "status": status,
        "failure_reason": failure_reason,
    }


def main() -> int:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    print("[Batch28] loading locked visual cache", flush=True)
    cache = load_base_cache()
    print(f"[Batch28] proposals={len(cache['proposal_meta'])} visual_shape={tuple(cache['visual_similarity'].shape)}", flush=True)

    # Global visual-only reproduction.
    visual_dir = OUT_ROOT / "visual_only"
    visual_dir.mkdir(parents=True, exist_ok=True)
    visual_output = visual_dir / "visual_only.json"
    try:
        if visual_output.exists():
            stats = stats_from_existing_detection(visual_output)
        else:
            zero_material = torch.zeros_like(cache["visual_similarity"])
            stats = generate_detections(
                cache,
                zero_material,
                mode="visual_only",
                lambda_material=0.0,
                lambda_reliability=0.0,
                output_path=visual_output,
            )
        ok, uncert_path, after_count, ap, ap50, ap75, error = eval_detection(visual_output, "visual_only")
        rows.append(
            result_row(
                bank_name="visual_only",
                channel_set="none",
                feature_dim=0,
                mode="visual_only",
                lm=0.0,
                lr=0.0,
                output_path=visual_output,
                status="SUCCESS" if ok else "FAILED",
                failure_reason=error,
                before_count=stats["detection_count_before_uncertanty"],
                after_count=after_count,
                changed_count=stats["changed_category_count"],
                changed_ratio=stats["changed_category_ratio"],
                ap=ap,
                ap50=ap50,
                ap75=ap75,
            )
        )
    except Exception as exc:  # noqa: BLE001
        rows.append(
            result_row(
                bank_name="visual_only",
                channel_set="none",
                feature_dim=0,
                mode="visual_only",
                lm=0.0,
                lr=0.0,
                output_path=visual_output,
                status="FAILED",
                failure_reason=repr(exc),
            )
        )

    for bank_name, channel_set, feature_dim, _notes in BANKS:
        print(f"[Batch28] loading material proto for {bank_name}", flush=True)
        material_proto = None

        for mode, lm, lr in MATERIAL_CONFIGS:
            config_slug = slug(mode, lm, lr)
            output_dir = OUT_ROOT / bank_name
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{config_slug}.json"
            print(f"[Batch28] {bank_name} {config_slug}", flush=True)
            try:
                if output_path.exists():
                    stats = stats_from_existing_detection(output_path)
                else:
                    if material_proto is None:
                        material_proto = material_proto_for_bank(bank_name, cache["prototype_meta"])
                    stats = generate_detections(
                        cache,
                        material_proto,
                        mode=mode,
                        lambda_material=lm,
                        lambda_reliability=lr,
                        output_path=output_path,
                    )
                ok, _uncert_path, after_count, ap, ap50, ap75, error = eval_detection(
                    output_path,
                    f"{bank_name}_{config_slug}",
                )
                rows.append(
                    result_row(
                        bank_name=bank_name,
                        channel_set=channel_set,
                        feature_dim=feature_dim,
                        mode=mode,
                        lm=lm,
                        lr=lr,
                        output_path=output_path,
                        status="SUCCESS" if ok else "FAILED",
                        failure_reason=error,
                        before_count=stats["detection_count_before_uncertanty"],
                        after_count=after_count,
                        changed_count=stats["changed_category_count"],
                        changed_ratio=stats["changed_category_ratio"],
                        ap=ap,
                        ap50=ap50,
                        ap75=ap75,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                rows.append(
                    result_row(
                        bank_name=bank_name,
                        channel_set=channel_set,
                        feature_dim=feature_dim,
                        mode=mode,
                        lm=lm,
                        lr=lr,
                        output_path=output_path,
                        status="FAILED",
                        failure_reason=repr(exc),
                    )
                )
        if material_proto is not None:
            del material_proto
        gc.collect()

    write_csv(RESULT_DIR / "det_compass_multichannel_preargmax_results.csv", rows)
    write_markdown(RESULT_DIR / "det_compass_multichannel_preargmax_results.md", rows, "DET-COMPASS 多通道 pre-argmax 全量结果")
    success_rows = [row for row in rows if row["status"] == "SUCCESS" and row["bank_name"] != "visual_only"]
    best_rows = []
    for bank_name, channel_set, feature_dim, _notes in BANKS:
        bank_rows = [row for row in success_rows if row["bank_name"] == bank_name and row["AP"] != ""]
        if not bank_rows:
            continue
        best = max(bank_rows, key=lambda row: float(row["AP"]))
        best_rows.append(
            {
                "bank_name": bank_name,
                "channel_set": channel_set,
                "feature_dim": feature_dim,
                "best_mode": best["mode"],
                "best_lambda_material": best["lambda_material"],
                "best_lambda_reliability": best["lambda_reliability"],
                "best_AP": best["AP"],
                "best_AP50": best["AP50"],
                "best_AP75": best["AP75"],
                "delta_AP_vs_raxo": best["delta_AP_vs_raxo"],
                "delta_AP_vs_legacy_colour_best": best["delta_AP_vs_legacy_colour_best"],
                "beats_legacy_colour_best": float(best["AP"]) > LEGACY_BEST_AP,
                "low_sample_warning": "91 categories sample_count < 3",
                "notes": "由 Batch28 grid 自动选择该 bank AP 最高配置。",
            }
        )
    write_csv(RESULT_DIR / "det_compass_multichannel_best_results.csv", best_rows)
    write_markdown(RESULT_DIR / "det_compass_multichannel_best_results.md", best_rows, "DET-COMPASS 多通道 material bank 最优结果")
    print(json.dumps({"rows": len(rows), "best_rows": best_rows}, indent=2), flush=True)
    return 0 if all(row["status"] == "SUCCESS" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
