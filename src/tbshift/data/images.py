"""Access to the cached, dataset-independent inputs shared by every model family.

On-disk layout (all under DATA_ROOT, default <repo>/data, override with TBSHIFT_DATA):

    manifests/<dataset>.csv                 one row per image: dataset, image_id, label, path[, mask_paths]
    prepared/<dataset>_s<size>/<id>.png     8-bit output of tbshift.preprocess.prepare (scripts/prepare_images.py)
    masks/<dataset>_s<size>/<id>.png        automatic lung mask, 0/255 (scripts/segment_lungs.py)

label: 1 = TB-related abnormality, 0 = normal. The manifest is the single source of truth for
which images exist and their labels; every model family reads images in manifest order.

NOTE: this module was missing from the repository because the old .gitignore rule "data/" also
matched src/tbshift/data/. It is reconstructed from its call sites. If an original copy exists,
diff it against this file before replacing it: the paths above must match the existing caches.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from tbshift import preprocess
from tbshift.provenance import DATA_ROOT

REQUIRED_COLUMNS = ("image_id", "label")


def manifest_path(dataset: str) -> Path:
    return DATA_ROOT / "manifests" / f"{dataset}.csv"


@lru_cache(maxsize=None)
def _manifest_cached(dataset: str) -> pd.DataFrame:
    path = manifest_path(dataset)
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    df = pd.read_csv(path, dtype={"image_id": str})
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    if df.image_id.duplicated().any():
        raise ValueError(f"{path}: duplicate image ids")
    if not set(df.label.unique()) <= {0, 1}:
        raise ValueError(f"{path}: labels must be 0/1, found {sorted(df.label.unique())}")
    if "dataset" not in df.columns:
        df.insert(0, "dataset", dataset)
    df["label"] = df["label"].astype(int)
    return df.reset_index(drop=True)


def manifest(dataset: str) -> pd.DataFrame:
    """Manifest rows in canonical (file) order. Returns a copy."""
    return _manifest_cached(dataset).copy()


def prepared_dir(dataset: str, size: int) -> Path:
    return DATA_ROOT / "prepared" / f"{dataset}_s{size}"


def mask_dir(dataset: str, size: int) -> Path:
    return DATA_ROOT / "masks" / f"{dataset}_s{size}"


def read_prepared(dataset: str, image_id: str, size: int) -> np.ndarray:
    """Prepared image as float64 in [0, 1], shape (size, size)."""
    p = prepared_dir(dataset, size) / f"{image_id}.png"
    a = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if a is None:
        raise FileNotFoundError(p)
    if a.shape != (size, size):
        raise ValueError(f"{p}: expected {size}x{size}, got {a.shape}")
    return a.astype(np.float64) / 255.0


def read_mask(dataset: str, image_id: str, size: int) -> np.ndarray | None:
    """Automatic lung mask (bool, (size, size)), or None if the mask is empty."""
    p = mask_dir(dataset, size) / f"{image_id}.png"
    a = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if a is None:
        raise FileNotFoundError(p)
    if a.shape != (size, size):
        raise ValueError(f"{p}: expected {size}x{size}, got {a.shape}")
    m = a > 127
    return m if m.any() else None


def load_input(dataset: str, image_id: str, size: int, mask_mode: str) -> np.ndarray:
    """Model input on the common grid. mask_mode 'lung': zero outside the dilated lung mask
    (preprocess.THORACIC_DILATION_PX); 'none': the whole prepared image (shortcut ablation).

    An empty lung mask is an error, never a silent fallback to the unmasked image.
    """
    img = read_prepared(dataset, image_id, size)
    if mask_mode == "none":
        return img
    if mask_mode != "lung":
        raise ValueError(f"unknown mask mode {mask_mode!r}")
    mask = read_mask(dataset, image_id, size)
    if mask is None:
        raise ValueError(f"empty lung mask for {dataset}/{image_id}")
    return preprocess.apply_region(img, mask)


def check_caches(dataset: str, size: int, need_masks: bool = True) -> dict:
    """Count manifest images with a prepared image (and a mask). Raises if any are missing."""
    df = manifest(dataset)
    ids = df.image_id.tolist()
    miss_img = [i for i in ids if not (prepared_dir(dataset, size) / f"{i}.png").exists()]
    miss_mask = ([i for i in ids if not (mask_dir(dataset, size) / f"{i}.png").exists()]
                 if need_masks else [])
    if miss_img or miss_mask:
        raise FileNotFoundError(
            f"{dataset}: {len(miss_img)} prepared images and {len(miss_mask)} masks missing "
            f"(first: {(miss_img + miss_mask)[:3]}). Run scripts/prepare_images.py and "
            f"scripts/segment_lungs.py first.")
    return {"dataset": dataset, "n": len(df), "n_pos": int(df.label.sum())}
