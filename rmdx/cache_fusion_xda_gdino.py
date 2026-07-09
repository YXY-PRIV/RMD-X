"""Status helpers for XDA-GDINO + RMD-X cache/fusion attempts."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def blocked_cache_status(reason: str, proposals_path: Path | None = None) -> dict[str, Any]:
    return {
        "final_status": "BLOCKED_NO_ADAPTED_PROPOSALS",
        "failure_reason": reason,
        "proposals_path": str(proposals_path or ""),
        "with_masks_path": "",
        "cache_dir": "",
        "rmdx_result_path": "",
        "AP": "",
        "AP50": "",
        "AP75": "",
        "notes": "RMD-X cache/fusion is skipped until a real adapted GroundingDINO proposal file exists.",
    }

