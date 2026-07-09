"""Calib-VTC policy structures.

Batch45 keeps this module small on purpose: policies must be learned or
selected from non-test calibration data.  If no calibration proposal cache is
available, downstream scripts create BLOCKED policy records instead of falling
back to full_test tuning.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class CalibVTCPolicy:
    policy_id: str
    policy_type: str
    train_datasets: list[str]
    detector: str
    topk: int
    model_params: dict[str, Any]
    selected_by_calibration: bool
    uses_full_test_gt: bool
    calibration_metric: str
    calibration_result: dict[str, Any]
    expected_risk: str
    status: str = "READY"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def save_policy(policy: CalibVTCPolicy, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(policy.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_policy(path: str | Path) -> CalibVTCPolicy:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return CalibVTCPolicy(**data)

