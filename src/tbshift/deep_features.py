"""Cache of frozen-backbone features (one directory per backbone x dataset x mask).

    DATA_ROOT/deep_features/<backbone>/<dataset>_mask-<mask>_s<size>/
        image_id.npy   (N,)        manifest order
        label.npy      (N,)
        pooled.npy     (N, D)      float32
        tokens.npy     (N, T, D)   float16 (memory-mapped on load)
        meta.json                  backbone provenance, shapes, array hashes, git commit

Extraction is per image, deterministic and unsupervised (frozen weights, no augmentation), so
caching before cross-validation is not leakage. Everything fitted across images (scaling,
heads) happens later inside the CV folds.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from tbshift.provenance import DATA_ROOT

FEATURE_ROOT = DATA_ROOT / "deep_features"


def cache_dir(backbone: str, dataset: str, mask: str, size: int) -> Path:
    return FEATURE_ROOT / backbone / f"{dataset}_mask-{mask}_s{size}"


def exists(backbone: str, dataset: str, mask: str, size: int) -> bool:
    return (cache_dir(backbone, dataset, mask, size) / "meta.json").exists()


def load(backbone: str, dataset: str, mask: str, size: int, kind: str = "pooled",
         mmap: bool = True):
    """(X, y, image_ids). kind: 'pooled' -> (N, D) float32; 'tokens' -> (N, T, D) float16."""
    d = cache_dir(backbone, dataset, mask, size)
    if not (d / "meta.json").exists():
        raise FileNotFoundError(f"no feature cache at {d}; run scripts/extract_deep_features.py")
    if kind not in ("pooled", "tokens"):
        raise ValueError(kind)
    X = np.load(d / f"{kind}.npy", mmap_mode="r" if (mmap and kind == "tokens") else None)
    y = np.load(d / "label.npy").astype(int)
    ids = np.load(d / "image_id.npy", allow_pickle=False).astype(str)
    if len(X) != len(y) or len(y) != len(ids):
        raise ValueError(f"{d}: inconsistent array lengths")
    if kind == "pooled" and not np.isfinite(X).all():
        raise ValueError(f"{d}: non-finite pooled features")
    return X, y, ids


def meta(backbone: str, dataset: str, mask: str, size: int) -> dict:
    return json.loads((cache_dir(backbone, dataset, mask, size) / "meta.json").read_text())


class CacheWriter:
    """Writes into <dir>.tmp and renames on commit, so a crash never leaves a half cache that
    looks complete (meta.json is the completion marker and is written last)."""

    def __init__(self, backbone: str, dataset: str, mask: str, size: int, n: int, d: int, t: int):
        self.final = cache_dir(backbone, dataset, mask, size)
        self.tmp = self.final.with_name(self.final.name + ".tmp")
        if self.tmp.exists():
            shutil.rmtree(self.tmp)
        self.tmp.mkdir(parents=True)
        self.pooled = np.lib.format.open_memmap(self.tmp / "pooled.npy", "w+", np.float32, (n, d))
        self.tokens = np.lib.format.open_memmap(self.tmp / "tokens.npy", "w+", np.float16, (n, t, d))

    def commit(self, image_ids: np.ndarray, labels: np.ndarray, meta_obj: dict) -> Path:
        self.pooled.flush(); self.tokens.flush()
        np.save(self.tmp / "image_id.npy", np.asarray(image_ids).astype(str))
        np.save(self.tmp / "label.npy", np.asarray(labels).astype(np.int64))
        del self.pooled, self.tokens
        (self.tmp / "meta.json").write_text(json.dumps(meta_obj, indent=2, default=str))
        if self.final.exists():
            shutil.rmtree(self.final)
        self.tmp.rename(self.final)
        return self.final
