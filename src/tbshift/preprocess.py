"""Image loading and normalisation, applied identically to every dataset.

Important for the domain-shift question: preprocessing must not differ by dataset, or the
preprocessing itself becomes a shortcut feature. Manual masks (Montgomery only) are therefore
used only to evaluate the automatic lung segmenter, never as model inputs.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

DEFAULT_SIZE = 512
THORACIC_DILATION_PX = 15  # lung mask is grown by this much to form the "thoracic region"


def load_gray(path: str | Path) -> np.ndarray:
    """Read an image of any bit depth as a float64 grayscale array (raw intensities)."""
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
    if arr is None:
        raise FileNotFoundError(path)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY if arr.shape[2] == 3 else cv2.COLOR_BGRA2GRAY)
    return arr.astype(np.float64)


def normalise(img: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0) -> np.ndarray:
    """Per-image percentile window to [0, 1]. Removes bit-depth and exposure offsets."""
    lo, hi = np.percentile(img, [lo_pct, hi_pct])
    if hi <= lo:
        return np.zeros_like(img)
    return np.clip((img - lo) / (hi - lo), 0.0, 1.0)


def pad_to_square(img: np.ndarray, value: float = 0.0) -> np.ndarray:
    """Centre-pad to a square, so that resizing preserves the aspect ratio.

    Without this, portrait and landscape frames are stretched differently per dataset, which
    distorts the shape-context angle bins.
    """
    h, w = img.shape
    s = max(h, w)
    top, left = (s - h) // 2, (s - w) // 2
    return cv2.copyMakeBorder(img, top, s - h - top, left, s - w - left,
                              cv2.BORDER_CONSTANT, value=value)


def resize(img: np.ndarray, size: int = DEFAULT_SIZE) -> np.ndarray:
    """Pad to square, then resize to size x size (aspect ratio preserved)."""
    sq = pad_to_square(img.astype(np.float32))
    return cv2.resize(sq, (size, size), interpolation=cv2.INTER_AREA).astype(np.float64)


def resize_mask(mask: np.ndarray, size: int = DEFAULT_SIZE) -> np.ndarray:
    sq = pad_to_square(mask.astype(np.uint8))
    return cv2.resize(sq, (size, size), interpolation=cv2.INTER_NEAREST).astype(bool)


def clahe(img: np.ndarray, clip: float = 2.0, tiles: int = 8) -> np.ndarray:
    op = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles))
    return op.apply((np.clip(img, 0, 1) * 255).astype(np.uint8)).astype(np.float64) / 255.0


def prepare(path: str | Path, size: int = DEFAULT_SIZE, use_clahe: bool = True) -> np.ndarray:
    """Load -> percentile normalise -> resize -> optional CLAHE. Output float in [0, 1]."""
    img = resize(normalise(load_gray(path)), size)
    return clahe(img) if use_clahe else img


def load_mask(paths: list[str | Path], size: int = DEFAULT_SIZE) -> np.ndarray:
    """Union of one or more binary mask images (e.g. Montgomery left + right lung)."""
    out = None
    for p in paths:
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(p)
        m = m > 127
        out = m if out is None else (out | m)
    return resize_mask(out, size)


def thoracic_region(mask: np.ndarray, dilation_px: int = THORACIC_DILATION_PX) -> np.ndarray:
    """Lung mask dilated into the surrounding thoracic region (ribs, pleura, mediastinal edge)."""
    from scipy import ndimage
    m = mask.astype(bool)
    return ndimage.binary_dilation(m, iterations=dilation_px) if dilation_px > 0 else m


def apply_region(img: np.ndarray, mask: np.ndarray | None,
                 dilation_px: int = THORACIC_DILATION_PX) -> np.ndarray:
    """Zero everything outside the thoracic region. mask=None returns the image unchanged."""
    if mask is None:
        return img
    return img * thoracic_region(mask, dilation_px)
