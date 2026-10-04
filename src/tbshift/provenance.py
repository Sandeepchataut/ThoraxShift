"""Run directories with full provenance, so every reported number traces to a real run."""
from __future__ import annotations

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
RUNS_ROOT = ROOT / "outputs" / "runs"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
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
