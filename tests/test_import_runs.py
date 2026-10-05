"""scripts/import_runs.py accepts clean published runs and rejects everything else."""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HEAD = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
IDENT = {"model_family": "deep", "model": "densenet121", "protocol": "cross", "mask": "lung",
         "source": "shenzhen", "target": "montgomery", "source_subsample": 0, "input_size": 384}


def _run(root: Path, name: str, commit=HEAD, dirty=False, metrics=True, recipe_modified=False,
         sessions=None):
    d = root / name
    (d / "ckpt").mkdir(parents=True)
    (d / "ckpt" / "best.pt").write_bytes(b"x")
    for f in ("config.json", "pip_freeze.txt"):
        (d / f).write_text("{}")
    (d / "meta.json").write_text(json.dumps({"git_commit": commit, "git_dirty": dirty}))
    pd.DataFrame({"image_id": ["a", "b"], "label": [0, 1], "score": [0.1, 0.9], "repeat": [0, 0]}) \
        .to_csv(d / "predictions.csv", index=False)
    if metrics:
        (d / "metrics.json").write_text(json.dumps({**IDENT, "recipe_modified": recipe_modified}))
    if sessions:
        (d / "sessions.jsonl").write_text("\n".join(json.dumps(s) for s in sessions) + "\n")


def _import(src: Path, dest: Path):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "import_runs.py"), str(src), "--dest", str(dest)],
                          capture_output=True, text=True)


def test_import_accepts_clean_run_and_drops_checkpoints(tmp_path):
    src, dest = tmp_path / "in", tmp_path / "out"
    _run(src, "ok_run")
    r = _import(src, dest)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (dest / "ok_run" / "metrics.json").exists() and not (dest / "ok_run" / "ckpt").exists()


def test_import_rejects_dirty_unknown_smoke_and_in_progress(tmp_path):
    src, dest = tmp_path / "in", tmp_path / "out"
    _run(src, "dirty", dirty=True)
    _run(src, "unknown", commit="0" * 40)
    _run(src, "smoke", recipe_modified=True)
    _run(src, "inprogress", metrics=False)
    _run(src, "dirty_session", sessions=[{"git_commit": HEAD, "git_dirty": True}])
    r = _import(src, dest)
    assert r.returncode != 0
    out = r.stdout
    for name in ("dirty", "unknown", "smoke", "dirty_session"):
        assert f"REJECTED     {name}" in out
    assert "IN PROGRESS  inprogress" in out
    assert not dest.exists() or not any(dest.iterdir())


def test_import_never_overwrites(tmp_path):
    src, dest = tmp_path / "in", tmp_path / "out"
    _run(src, "same")
    _run(dest, "same")
    r = _import(src, dest)
    assert "already exists" in r.stdout
