"""Paired comparisons between saved runs (repeat 0), importable by joblib workers."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from tbshift.eval import metrics as M


@lru_cache(maxsize=None)
def scores(run_dir: str, repeat: int = 0) -> pd.DataFrame:
    p = pd.read_csv(Path(run_dir) / "predictions.csv", dtype={"image_id": str})
    p = p[p["repeat"] == repeat]
    if p.image_id.duplicated().any():
        raise ValueError(f"duplicate image ids in {run_dir}")
    return p[["image_id", "label", "score"]].set_index("image_id")


def aligned(*run_dirs: Path) -> tuple[np.ndarray, list[np.ndarray]]:
    """Labels and repeat-0 scores of several runs on exactly the same images (same order)."""
    frames = [scores(str(d)) for d in run_dirs]
    base = frames[0]
    for d, f in zip(run_dirs[1:], frames[1:]):
        if len(f) != len(base) or not f.index.isin(base.index).all():
            raise ValueError(f"image sets differ: {run_dirs[0].name} vs {d.name}")
    ids = base.index
    y = base.label.to_numpy()
    for d, f in zip(run_dirs[1:], frames[1:]):
        if not (f.loc[ids].label.to_numpy() == y).all():
            raise ValueError(f"labels disagree: {run_dirs[0].name} vs {d.name}")
    return y, [f.loc[ids].score.to_numpy() for f in frames]


# ------------------------------------------------------------------------------- comparisons
def _cmp_auc(a_dir, b_dir, n_boot, seed):
    y, (sa, sb) = aligned(a_dir, b_dir)
    return M.auc_difference(y, sa, sb, n_boot=n_boot, seed=seed)


def _cmp_gap(in_dir, cross_dir, n_boot, seed):
    y, (si, sc) = aligned(in_dir, cross_dir)
    return M.transfer_gap(y, si, sc, n_boot=n_boot, seed=seed)


def _cmp_gapdiff(in_a, cross_a, in_b, cross_b, n_boot, seed):
    y, (ia, ca, ib, cb) = aligned(in_a, cross_a, in_b, cross_b)
    return M.gap_difference(y, ia, ca, ib, cb, n_boot=n_boot, seed=seed)


def _safe(fn, args):
    try:
        return fn(*args)
    except Exception as e:  # noqa: BLE001 - reported in the table as an error row
        return e


def run_parallel(tasks: list[tuple], n_jobs: int) -> list:
    """tasks: (fn, args) tuples; returns results in order (exceptions are returned, not raised)."""
    return Parallel(n_jobs=n_jobs)(delayed(_safe)(fn, args) for fn, args in tasks)
