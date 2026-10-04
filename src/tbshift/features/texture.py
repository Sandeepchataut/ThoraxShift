"""Hand-crafted texture descriptors computed inside the lung region.

LBP (uniform, rotation-invariant), GLCM statistics, a Gabor filter bank, and HOG. Each function
returns a fixed-length 1-D float vector, so images with different masks are comparable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from skimage import feature, filters


@dataclass(frozen=True)
class TextureParams:
    lbp_radii: tuple[int, ...] = (1, 2, 3)
    glcm_levels: int = 32
    glcm_distances: tuple[int, ...] = (1, 2, 4)
    glcm_angles: tuple[float, ...] = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
    gabor_frequencies: tuple[float, ...] = (0.05, 0.1, 0.2, 0.3)
    gabor_n_orient: int = 4
    hog_size: int = 128
    hog_ppc: int = 16
    hog_orientations: int = 9
    families: tuple[str, ...] = field(default=("lbp", "glcm", "gabor", "hog"))


def _region(mask: np.ndarray | None, shape) -> np.ndarray:
    return np.ones(shape, bool) if mask is None else mask.astype(bool)


def _bbox_crop(img: np.ndarray, region: np.ndarray) -> np.ndarray:
    rows, cols = np.nonzero(region)
    if len(rows) == 0:
        return img
    return img[rows.min():rows.max() + 1, cols.min():cols.max() + 1]


def lbp_features(img: np.ndarray, mask, p: TextureParams) -> np.ndarray:
    region = _region(mask, img.shape)
    img8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    out = []
    for r in p.lbp_radii:
        n_pts = 8 * r
        codes = feature.local_binary_pattern(img8, n_pts, r, method="uniform")
        hist, _ = np.histogram(codes[region], bins=n_pts + 2, range=(0, n_pts + 2))
        out.append(hist / max(hist.sum(), 1))
    return np.concatenate(out)


def glcm_features(img: np.ndarray, mask, p: TextureParams) -> np.ndarray:
    region = _region(mask, img.shape)
    crop = _bbox_crop(img, region)
    q = np.minimum((np.clip(crop, 0, 1) * p.glcm_levels).astype(np.uint8), p.glcm_levels - 1)
    glcm = feature.graycomatrix(q, p.glcm_distances, p.glcm_angles, levels=p.glcm_levels,
                                symmetric=True, normed=True)
    props = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM")
    # Average over angles (approximate rotation invariance); keep distances separate.
    return np.concatenate([np.nan_to_num(feature.graycoprops(glcm, pr).mean(axis=1))
                           for pr in props])


def gabor_features(img: np.ndarray, mask, p: TextureParams) -> np.ndarray:
    region = _region(mask, img.shape)
    out = []
    for f in p.gabor_frequencies:
        for k in range(p.gabor_n_orient):
            real, imag = filters.gabor(img, frequency=f, theta=k * np.pi / p.gabor_n_orient)
            mag = np.hypot(real, imag)[region]
            out.extend([mag.mean(), mag.std()])
    return np.asarray(out)


def hog_features(img: np.ndarray, mask, p: TextureParams) -> np.ndarray:
    region = _region(mask, img.shape)
    crop = _bbox_crop(img * region, region)
    crop = cv2.resize(crop.astype(np.float32), (p.hog_size, p.hog_size),
                      interpolation=cv2.INTER_AREA)
    return feature.hog(crop, orientations=p.hog_orientations,
                       pixels_per_cell=(p.hog_ppc, p.hog_ppc), cells_per_block=(2, 2),
                       feature_vector=True)


_FUNCS = {"lbp": lbp_features, "glcm": glcm_features, "gabor": gabor_features,
          "hog": hog_features}


def extract(img: np.ndarray, mask, p: TextureParams) -> dict[str, np.ndarray]:
    """Dict family -> feature vector, for each family in p.families."""
    return {name: _FUNCS[name](img, mask, p).astype(np.float64) for name in p.families}
