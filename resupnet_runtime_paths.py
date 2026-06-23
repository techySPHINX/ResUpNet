"""Runtime paths for ResUpNet experiments.

Keep generated caches and default run outputs away from the C: drive.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _default_runtime_root() -> Path:
    configured = os.environ.get("RESUPNET_RUNTIME_ROOT")
    if configured:
        return Path(configured)
    e_root = Path("E:/")
    if e_root.exists():
        return e_root / "ResUpNet"
    raise RuntimeError(
        "E: drive is required for ResUpNet runtime cache/output. "
        "Set RESUPNET_RUNTIME_ROOT to another drive only if you intentionally want that."
    )


def configure_runtime_paths() -> dict[str, str]:
    sys.dont_write_bytecode = True
    repo_root = Path(__file__).resolve().parent
    root = _default_runtime_root()
    cache_root = root / "cache"
    temp_root = root / "tmp"
    runs_root = root / "runs"
    paths = {
        "RESUPNET_RUNTIME_ROOT": root,
        "RESUPNET_CACHE_ROOT": cache_root,
        "RESUPNET_RUNS_ROOT": runs_root,
        "TMP": temp_root,
        "TEMP": temp_root,
        "TMPDIR": temp_root,
        "MPLCONFIGDIR": cache_root / "matplotlib",
        "JOBLIB_TEMP_FOLDER": temp_root / "joblib",
        "PIP_CACHE_DIR": cache_root / "pip",
        "KAGGLE_CONFIG_DIR": repo_root / "data" / ".kaggle",
    }
    for value in paths.values():
        Path(value).mkdir(parents=True, exist_ok=True)
    for key, value in paths.items():
        os.environ.setdefault(key, str(value))
    return {key: str(value) for key, value in paths.items()}


def default_runs_root() -> Path:
    configure_runtime_paths()
    return Path(os.environ["RESUPNET_RUNS_ROOT"])
