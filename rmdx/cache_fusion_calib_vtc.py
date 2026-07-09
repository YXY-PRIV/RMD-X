#!/usr/bin/env python3
"""Apply a frozen Calib-VTC policy to a full_test cache.

This CLI is intentionally conservative.  It refuses to run BLOCKED policies
and delegates actual top-K fusion to future READY policies only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rmdx.calib_vtc_fusion import load_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply Calib-VTC policy")
    parser.add_argument("--policy", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-path", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    policy = load_policy(args.policy)
    if policy.status != "READY":
        raise RuntimeError(
            f"Policy {policy.policy_id} is not READY: status={policy.status}; "
            "refusing to run full_test fusion."
        )
    raise NotImplementedError(
        "READY Calib-VTC policy application is not implemented until a valid "
        "non-test calibration proposal cache and trained policy are available."
    )


if __name__ == "__main__":
    raise SystemExit(main())

