"""Schema helpers for calibration detector proposals."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_coco_result_list(path: str | Path, valid_image_ids: set[int], valid_category_ids: set[int]) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {"valid": False, "count": 0, "error": "file_missing"}
    try:
        data = load_json(path)
    except Exception as exc:
        return {"valid": False, "count": 0, "error": f"{type(exc).__name__}: {exc}"}
    if not isinstance(data, list):
        return {"valid": False, "count": 0, "error": "top_level_not_list"}
    bad = []
    for idx, det in enumerate(data[:1000]):
        try:
            image_id = int(det["image_id"])
            category_id = int(det["category_id"])
            bbox = det["bbox"]
            score = float(det["score"])
            if image_id not in valid_image_ids:
                bad.append((idx, "bad_image_id"))
            if category_id not in valid_category_ids:
                bad.append((idx, "bad_category_id"))
            if not isinstance(bbox, list) or len(bbox) != 4 or any(float(v) < 0 for v in bbox[2:]):
                bad.append((idx, "bad_bbox"))
            if not (0.0 <= score <= 1.0):
                bad.append((idx, "bad_score"))
        except Exception as exc:
            bad.append((idx, f"{type(exc).__name__}: {exc}"))
    return {"valid": not bad, "count": len(data), "error": ";".join(map(str, bad[:10]))}
