"""Run directories with full provenance, so every reported number traces to a real run."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("TBSHIFT_DATA", ROOT / "data"))
# Override on cloud GPU machines so runs land on persistent storage.
RUNS_ROOT = Path(os.environ.get("TBSHIFT_RUNS", ROOT / "outputs" / "runs"))
RAW_ROOT = DATA_ROOT / "raw"


def resolve_data_path(p: str | Path) -> Path:
    """Manifest paths are relative to data/raw; absolute paths are returned unchanged."""
    p = Path(p)
    return p if p.is_absolute() else RAW_ROOT / p


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Required for deterministic cuBLAS kernels; must be set before CUDA initialises.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import torch
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unavailable"


def cache_provenance(directory: Path, params: dict) -> None:
    """Record which code state produced a derived-data cache (prepared images, masks)."""
    dirty = _git("status", "--porcelain")
    write_json(Path(directory) / "_provenance.json", {
        "created_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "git_commit": _git("rev-parse", "HEAD"), "git_dirty": bool(dirty) and dirty != "unavailable",
        "command": sys.argv, **params})


def new_run(name: str, config: dict) -> Path:
    """Create outputs/runs/<utc-timestamp>_<name>/ with config, git state and environment."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = RUNS_ROOT / f"{stamp}_{name}"
    run.mkdir(parents=True, exist_ok=False)
    dirty = _git("status", "--porcelain")
    meta = {
        "run_id": run.name,
        "started_utc": stamp,
        "command": sys.argv,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(dirty) and dirty != "unavailable",
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }
    (run / "config.json").write_text(json.dumps(config, indent=2, default=str))
    (run / "meta.json").write_text(json.dumps(meta, indent=2))
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    (run / "pip_freeze.txt").write_text(freeze.stdout)
    return run


def write_json(path: Path, obj) -> None:
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    Path(path).write_text(json.dumps(obj, indent=2, default=default))
