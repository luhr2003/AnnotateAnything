from __future__ import annotations

import shutil
import sys
import uuid
from pathlib import Path

from isaacsim import SimulationApp

_CURRENT_RUNTIME_ROOT: Path | None = None


def _runtime_cache_base() -> Path:
    return (Path(__file__).resolve().parent / ".cache" / "isaacsim_runtime").resolve()


def create_simulation_app(*, headless: bool, script_name: str | None = None) -> SimulationApp:
    global _CURRENT_RUNTIME_ROOT

    base = _runtime_cache_base()
    base.mkdir(parents=True, exist_ok=True)

    label = script_name or Path(sys.argv[0]).stem or "isaacsim"
    runtime_root = (base / f"{label}_{uuid.uuid4().hex[:12]}").resolve()
    runtime_root.mkdir(parents=True, exist_ok=True)
    _CURRENT_RUNTIME_ROOT = runtime_root

    original_argv = list(sys.argv)
    sys.argv = [sys.argv[0]]
    try:
        return SimulationApp(
            {
                "headless": bool(headless),
                "extra_args": ["--portable-root", str(runtime_root)],
            }
        )
    finally:
        sys.argv = original_argv


def cleanup_simulation_runtime_cache() -> None:
    global _CURRENT_RUNTIME_ROOT

    runtime_root = _CURRENT_RUNTIME_ROOT
    _CURRENT_RUNTIME_ROOT = None
    if runtime_root is None:
        return
    shutil.rmtree(runtime_root, ignore_errors=True)
