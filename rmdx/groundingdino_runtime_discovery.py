"""GroundingDINO runtime discovery helpers for Batch45.1a.

The helpers only inspect local files and Python imports. They do not download
models, install packages, or run detector inference.
"""

from __future__ import annotations

import importlib
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SEARCH_ROOTS = [
    Path("."),
    Path("/mnt/data/yxy/program"),
    Path("/home/xinxueshi/yxy/program"),
    Path("/mnt/data/raxo"),
    Path("/home/xinxueshi/yxy/data"),
]


def module_status(module_name: str) -> dict[str, str]:
    try:
        module = importlib.import_module(module_name)
        return {
            "module": module_name,
            "available": "True",
            "version": str(getattr(module, "__version__", "")),
            "path": str(getattr(module, "__file__", "")),
            "error": "",
        }
    except Exception as exc:
        return {
            "module": module_name,
            "available": "False",
            "version": "",
            "path": "",
            "error": f"{type(exc).__name__}: {exc}",
        }


def container_module_status(container: str, module_name: str) -> dict[str, str]:
    code = (
        "import importlib, json\n"
        f"m={module_name!r}\n"
        "try:\n"
        "    mod=importlib.import_module(m)\n"
        "    print(json.dumps({'available':'True','version':str(getattr(mod,'__version__','')),'path':str(getattr(mod,'__file__','')),'error':''}))\n"
        "except Exception as exc:\n"
        "    print(json.dumps({'available':'False','version':'','path':'','error':type(exc).__name__+': '+str(exc)}))\n"
    )
    try:
        proc = subprocess.run(
            ["docker", "exec", "-w", "/tmp", container, "python3", "-c", code],
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )
        import json

        data = json.loads(proc.stdout.strip().splitlines()[-1])
        return {"container": container, "module": module_name, **{k: str(v) for k, v in data.items()}}
    except Exception as exc:
        return {
            "container": container,
            "module": module_name,
            "available": "False",
            "version": "",
            "path": "",
            "error": f"{type(exc).__name__}: {exc}",
        }


def find_paths(patterns: list[str], roots: list[Path] | None = None, max_results: int = 200) -> list[str]:
    roots = roots or SEARCH_ROOTS
    found: list[str] = []
    for root in roots:
        root = root.resolve() if root.exists() else root
        if not root.exists():
            continue
        for pattern in patterns:
            try:
                for path in root.rglob(pattern):
                    found.append(str(path))
                    if len(found) >= max_results:
                        return sorted(set(found))
            except PermissionError:
                continue
            except OSError:
                continue
    return sorted(set(found))


def find_text_hits(patterns: list[str], files: list[Path], max_hits: int = 200) -> list[str]:
    hits: list[str] = []
    for file in files:
        if not file.exists() or not file.is_file():
            continue
        try:
            lines = file.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue
        for idx, line in enumerate(lines, start=1):
            lower = line.lower()
            if any(pattern.lower() in lower for pattern in patterns):
                try:
                    rel = file.relative_to(REPO_ROOT)
                except ValueError:
                    rel = file
                hits.append(f"{rel}:{idx}: {line.strip()}")
                if len(hits) >= max_hits:
                    return hits
    return hits


def derive_runtime_status(package_ok: bool, config_paths: list[str], checkpoint_paths: list[str]) -> tuple[str, str]:
    missing = []
    if not package_ok:
        missing.append("package")
    if not config_paths:
        missing.append("config")
    if not checkpoint_paths:
        missing.append("checkpoint")
    if not missing:
        return "READY", ""
    if missing == ["package"]:
        return "BLOCKED_NO_PACKAGE", "groundingDINO Python package/API is unavailable"
    if missing == ["config"]:
        return "BLOCKED_NO_CONFIG", "GroundingDINO config file was not found"
    if missing == ["checkpoint"]:
        return "BLOCKED_NO_CHECKPOINT", "GroundingDINO checkpoint was not found"
    return "BLOCKED_NO_GROUNDINGDINO_RUNTIME", "missing required runtime components: " + ",".join(missing)


def csv_safe_list(values: list[str], limit: int = 20) -> str:
    return " ; ".join(values[:limit])
