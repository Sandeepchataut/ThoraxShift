"""Automatic lung segmentation, applied identically to every dataset.

Model: the pretrained torchxrayvision PSPNet anatomical segmenter (ChestX-Det). It is an
external pretrained model, not trained on any of our evaluation sets. Its accuracy is checked
against the Montgomery manual masks (scripts/segment_lungs.py --evaluate).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

LEFT, RIGHT = "Left Lung", "Right Lung"


def _largest_component(m: np.ndarray) -> np.ndarray:
    lab, n = ndimage.label(m)
    if n <= 1:
        return m
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1)


def postprocess(prob_left: np.ndarray, prob_right: np.ndarray, thr: float = 0.5) -> np.ndarray:
    """Threshold each lung, keep its largest component, fill holes, take the union."""
    out = np.zeros(prob_left.shape, bool)
    for p in (prob_left, prob_right):
        m = _largest_component(p >= thr)
        out |= ndimage.binary_fill_holes(m)
    return out


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else 1.0


class LungSegmenter:
    def __init__(self, cache_dir: str | None = None, threads: int | None = None):
        import torch
        import torchxrayvision as xrv

        if threads:
            torch.set_num_threads(threads)
        self.torch = torch
        self.model = xrv.baseline_models.chestx_det.PSPNet(cache_dir=cache_dir).eval()
        self.idx = (self.model.targets.index(LEFT), self.model.targets.index(RIGHT))

    def __call__(self, img01: np.ndarray, thr: float = 0.5) -> np.ndarray:
        """img01: float image in [0, 1] (normalised, before CLAHE), any square size.
        Returns a boolean lung mask with the same shape."""
        x = (img01.astype(np.float32) * 2048.0 - 1024.0)[None, None]
        with self.torch.no_grad():
            logits = self.model(self.torch.from_numpy(x))
            probs = self.torch.sigmoid(logits)[0].numpy()
        pl, pr = probs[self.idx[0]], probs[self.idx[1]]
        if pl.shape != img01.shape:
            import cv2
            pl = cv2.resize(pl, img01.shape[::-1], interpolation=cv2.INTER_LINEAR)
            pr = cv2.resize(pr, img01.shape[::-1], interpolation=cv2.INTER_LINEAR)
        return postprocess(pl, pr, thr)
