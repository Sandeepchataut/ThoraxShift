"""The study design in one place, shared by the orchestrator and the aggregation step so the
runs that are launched and the runs that are analysed can never drift apart.
Mirrors docs/analysis_plan.md; change both together (and report it as a deviation once frozen).
"""
from __future__ import annotations

from tbshift.models import backbones as bb
from tbshift.models import heads as hd
from tbshift.runs import size_matched_n

DATASETS = ("shenzhen", "montgomery", "tbx11k")

# Holm family for the primary endpoint = PRIMARY_PAIRS x primary backbones (fixed size).
PRIMARY_PAIRS = (("montgomery", "shenzhen"), ("tbx11k", "shenzhen"),
                 ("shenzhen", "tbx11k"), ("montgomery", "tbx11k"))
DESCRIPTIVE_PAIRS = (("shenzhen", "montgomery"), ("tbx11k", "montgomery"))
CROSS_PAIRS = PRIMARY_PAIRS + DESCRIPTIVE_PAIRS

PRIMARY_MASK = "lung"
SHORTCUT_MASK = "none"
PREPARED_SIZE = 512

HC_FAMILY = "handcrafted"
HC_FEATURES = ("sc", "lbp", "glcm", "gabor", "hog")
HC_PRIMARY = "+".join(HC_FEATURES)          # FUSION
HC_INPUT_SIZE = PREPARED_SIZE

FROZEN_FAMILY = "frozen"
PRIMARY_HEAD = hd.PRIMARY_HEAD               # svm_rbf: identical classifier to FUSION
HEADS = tuple(hd.HEADS)
BACKBONES = tuple(bb.BACKBONES)

# In-domain nested-CV repeats. Repeat 0 is the pre-specified primary repeat for every paired
# test; the rest only give mean +/- SD. Large datasets get fewer repeats (compute; the
# repeat-to-repeat SD is small at n ~ 8k).
REPEATS = 5
REPEATS_LARGE = 2
LARGE_N = 2000


def indomain_repeats(n_images: int) -> int:
    return REPEATS_LARGE if n_images > LARGE_N else REPEATS


def size_matched_pairs(sizes: dict[str, int]) -> list[tuple[str, str, int]]:
    """(source, target, n) for every cross pair where a size-matched run is defined."""
    out = []
    for s, t in CROSS_PAIRS:
        try:
            out.append((s, t, size_matched_n(sizes[s], sizes[t])))
        except ValueError:
            pass
    return out


def frozen_model(backbone: str, head: str, hybrid: bool = False) -> str:
    return f"{backbone}{'+fusion' if hybrid else ''}/{head}"


def parse_frozen_model(model: str) -> tuple[str, str, bool]:
    left, head = model.split("/")
    hybrid = left.endswith("+fusion")
    return left.removesuffix("+fusion"), head, hybrid
