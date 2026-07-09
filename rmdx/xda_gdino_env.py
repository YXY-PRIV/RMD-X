"""Environment helpers for Batch56 XDA-GDINO-RMD-X.

This module only inspects or repairs the GroundingDINO runtime. It does not
modify RAXO code or existing experiment outputs.
"""

from __future__ import annotations

import csv
import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
AUDIT_CSV = ROOT / "outputs_rmdx/calib_vtc/detector_audit/groundingdino_runtime_assets_audit.csv"
SOURCE_DEFAULT = Path("/mnt/data/raxo/external_artifacts/groundingdino/source/GroundingDINO")
CONFIG_DEFAULT = Path("/mnt/data/raxo/external_artifacts/groundingdino/config/GroundingDINO_SwinT_OGC.py")
CHECKPOINT_DEFAULT = Path("/mnt/data/raxo/external_artifacts/groundingdino/checkpoint/groundingdino_swint_ogc.pth")
REQUIRED_MODULES = ["torch", "torchvision", "timm", "supervision", "addict", "yapf", "groundingdino", "groundingdino.util.inference"]
INSTALL_PACKAGES = ["timm", "supervision", "addict", "yapf"]


def read_first_csv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else {}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_assets() -> dict[str, Path]:
    row = read_first_csv(AUDIT_CSV)
    return {
        "source": Path(row.get("source_path") or SOURCE_DEFAULT),
        "config": Path(row.get("config_path") or CONFIG_DEFAULT),
        "checkpoint": Path(row.get("checkpoint_path") or CHECKPOINT_DEFAULT),
    }


def module_status(source_path: Path | None = None) -> dict[str, dict[str, Any]]:
    if source_path and str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))
    rows: dict[str, dict[str, Any]] = {}
    for name in REQUIRED_MODULES:
        try:
            mod = importlib.import_module(name)
            rows[name] = {
                "ok": True,
                "version": str(getattr(mod, "__version__", "")),
                "path": str(getattr(mod, "__file__", "")),
                "error": "",
            }
        except Exception as exc:  # noqa: BLE001 - record exact runtime blocker
            rows[name] = {
                "ok": False,
                "version": "",
                "path": "",
                "error": f"{type(exc).__name__}: {exc}",
            }
    return rows


def pip_freeze_key_packages() -> str:
    proc = subprocess.run([sys.executable, "-m", "pip", "freeze"], text=True, capture_output=True, check=False, timeout=120)
    keys = ("torch", "torchvision", "timm", "supervision", "addict", "yapf", "grounding")
    lines = [line for line in proc.stdout.splitlines() if line.lower().startswith(keys)]
    return "\n".join(lines)


def install_missing(mods: dict[str, dict[str, Any]], log_path: Path) -> tuple[bool, str, str]:
    missing_top = {name.split(".")[0] for name, row in mods.items() if not row["ok"]}
    packages = [pkg for pkg in INSTALL_PACKAGES if pkg in missing_top]
    if not packages:
        return True, "", "no_install_needed"
    cmd = [sys.executable, "-m", "pip", "install", "-U"] + packages
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False, timeout=3600)
    text = "$ " + " ".join(cmd) + "\n\n" + proc.stdout + "\n" + proc.stderr
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(text, encoding="utf-8")
    return proc.returncode == 0, " ".join(cmd), text[-3000:]


def runtime_repair(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    assets = get_assets()
    before = module_status(assets["source"])
    install_ok, install_cmd, install_tail = install_missing(before, output_dir / "pip_install_groundingdino_deps.log")
    after = module_status(assets["source"]) if install_ok else before

    source_ok = assets["source"].exists()
    config_ok = assets["config"].exists()
    ckpt_ok = assets["checkpoint"].exists()
    import_ok = bool(after.get("groundingdino.util.inference", {}).get("ok"))
    if not install_ok:
        status = "BLOCKED_DEPENDENCY_INSTALL_FAILED"
    elif not source_ok:
        status = "BLOCKED_GROUNDINGDINO_IMPORT_FAILED"
    elif not import_ok:
        status = "BLOCKED_GROUNDINGDINO_IMPORT_FAILED"
    elif not config_ok:
        status = "BLOCKED_CONFIG_MISSING"
    elif not ckpt_ok:
        status = "BLOCKED_CHECKPOINT_MISSING"
    else:
        status = "READY_GROUNDINGDINO_RUNTIME"

    row = {
        "final_status": status,
        "python": sys.executable,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "source_path": str(assets["source"]),
        "source_exists": source_ok,
        "config_path": str(assets["config"]),
        "config_exists": config_ok,
        "checkpoint_path": str(assets["checkpoint"]),
        "checkpoint_exists": ckpt_ok,
        "checkpoint_size": assets["checkpoint"].stat().st_size if ckpt_ok else 0,
        "checkpoint_sha256": sha256_file(assets["checkpoint"]) if ckpt_ok else "",
        "install_ok": install_ok,
        "install_command": install_cmd,
        "install_log_tail": install_tail.replace("\n", "\\n"),
        "pip_freeze_key_packages": pip_freeze_key_packages().replace("\n", "; "),
        "module_status_json": json.dumps(after, ensure_ascii=False),
    }
    return row

