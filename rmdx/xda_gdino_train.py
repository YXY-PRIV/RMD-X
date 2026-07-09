"""Training-entrypoint discovery for XDA-GDINO.

The official GroundingDINO source bundled in this workspace is inference/demo
oriented. This module intentionally refuses to fabricate a training loop when a
supported fine-tuning entrypoint is absent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rmdx.xda_gdino_env import get_assets


TRAIN_NAME_PATTERNS = ("train", "finetune", "fine_tune", "lora", "adapter")


def discover_training_entrypoints() -> list[dict[str, Any]]:
    source = get_assets()["source"]
    hits: list[dict[str, Any]] = []
    if not source.exists():
        return hits
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        lower = path.name.lower()
        if any(pat in lower for pat in TRAIN_NAME_PATTERNS) and path.suffix in {".py", ".sh"}:
            hits.append({"path": str(path), "name": path.name, "relative_path": str(path.relative_to(source))})
    return hits


def training_status(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    hits = discover_training_entrypoints()
    source = get_assets()["source"]
    if not source.exists():
        status = "BLOCKED_TRAINING_RUNTIME_ERROR"
        reason = "GroundingDINO source path missing"
    elif not hits:
        status = "BLOCKED_NO_TRAINING_ENTRYPOINT"
        reason = "No LoRA/adapter/fine-tuning/training entrypoint was found in the local GroundingDINO source."
    else:
        status = "READY_SMOKE_ONLY"
        reason = "Training-like files exist, but Batch56 does not auto-invent a fine-tuning recipe; manual review required before GPU training."
    row = {
        "final_status": status,
        "failure_reason": reason,
        "training_entrypoint_count": len(hits),
        "training_entrypoints_json": json.dumps(hits, ensure_ascii=False),
        "checkpoint_path": "",
        "checkpoint_ready": False,
        "max_steps": 0,
        "gpu_used": "",
        "full_test_used_for_training": False,
        "notes": "No detector fine-tuning was run unless an explicit supported entrypoint is available.",
    }
    return row

