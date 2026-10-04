"""Shared run-output contract for every model family.

Every experiment run directory contains:
  config.json, meta.json, pip_freeze.txt      (written by provenance.new_run)
  predictions.csv                              columns: image_id, label, score, repeat[, fold]
  metrics.json                                 must contain the RUN_KEYS below

The aggregation step identifies runs only through RUN_KEYS, so it does not care which model
family produced a run.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tbshift.eval import metrics as M

RUN_KEYS = ("model_family", "model", "protocol", "mask", "source", "target",
            "source_subsample", "input_size")


def run_identity(model_family: str, model: str, protocol: str, mask: str, source: str,
                 target: str, source_subsample: int | None, input_size: int) -> dict:
    """protocol 'indomain': source == target == the dataset."""
    if protocol not in ("indomain", "cross"):
        raise ValueError(protocol)
    if protocol == "indomain" and source != target:
        raise ValueError("in-domain runs must have source == target")
    return {"model_family": model_family, "model": model, "protocol": protocol, "mask": mask,
            "source": source, "target": target, "source_subsample": source_subsample or 0,
            "input_size": input_size}


def summarise(y, s, threshold=None) -> dict:
    out = {"auc_delong": M.auc_delong_ci(y, s).as_dict()}
    if threshold is not None:
        out["at_source_threshold"] = M.threshold_metrics(y, s, threshold)
    return out


def size_matched_n(n_source: int, n_target: int, folds: int = 5) -> int:
    """Training-set size of one in-domain CV fold on the target: floor((folds-1)/folds * n_target).

    Raises if this is not smaller than the source, because 'size-matched' would then be a
    full-source run under another name.
    """
    n = (folds - 1) * n_target // folds
    if n >= n_source:
        raise ValueError(f"size-matched n={n} is not smaller than the source (n={n_source}); "
                         "a size-matched run is not defined for this pair")
    return n


def stratified_subsample(y: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Indices of a class-stratified random subsample of size n (n = 0 means all images)."""
    y = np.asarray(y).astype(int)
    if not n:
        return np.arange(len(y))
    if n >= len(y):
        raise ValueError(f"subsample n={n} must be smaller than the source size {len(y)}")
    rng = np.random.default_rng(seed)
    out = []
    for c in (0, 1):
        idx = np.flatnonzero(y == c)
        k = int(round(n * len(idx) / len(y)))
        out.append(rng.choice(idx, k, replace=False))
    return np.sort(np.concatenate(out))


def indomain_metrics(identity: dict, y: np.ndarray, per_repeat: list[dict]) -> dict:
    aucs = [p["auc_delong"]["auc"] for p in per_repeat]
    return {**identity, "n": int(len(y)), "n_pos": int(np.sum(y)),
            "primary_repeat": 0,
            "auc_mean_over_repeats": float(np.mean(aucs)),
            "auc_sd_over_repeats": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
            "repeats": per_repeat}


def cross_metrics(identity: dict, yt: np.ndarray, per_repeat: list[dict]) -> dict:
    aucs = [p["target_metrics"]["auc_delong"]["auc"] for p in per_repeat]
    return {**identity, "n_target": int(len(yt)), "n_target_pos": int(np.sum(yt)),
            "primary_repeat": 0,
            "target_auc_mean_over_repeats": float(np.mean(aucs)),
            "target_auc_sd_over_repeats": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
            "repeats": per_repeat}


def predictions_frame(ids, y, scores, repeat: int, fold=None) -> pd.DataFrame:
    df = pd.DataFrame({"image_id": ids, "label": np.asarray(y).astype(int),
                       "score": np.asarray(scores, dtype=float), "repeat": repeat})
    if fold is not None:
        df["fold"] = fold
    return df
