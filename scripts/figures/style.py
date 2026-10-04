"""Shared figure style: vector output, consistent identity colours, mock-up watermark."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "outputs" / "figures"

# Identity colours (CVD-validated pairs). Colour follows the entity in every figure.
FAMILY = {"handcrafted": "#2a78d6", "deep": "#eb6834"}
FAMILY_LABEL = {"handcrafted": "Hand-crafted (FUSION)", "deep": "Deep (DenseNet-121)"}
MASK = {"lung": "#1baf7a", "none": "#4a3aa7"}
MASK_LABEL = {"lung": "Lung-masked", "none": "Unmasked"}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DATASET_LABEL = {"shenzhen": "Shenzhen", "montgomery": "Montgomery", "tbx11k": "TBX11K"}

# IEEE two-column widths (inches)
COL, FULL = 3.5, 7.16


def apply() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
        "lines.linewidth": 1.4, "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def mock_watermark(fig) -> None:
    fig.text(0.5, 0.5, "LAYOUT MOCK-UP — NOT RESULTS", ha="center", va="center", rotation=25,
             fontsize=16, color="#e34948", alpha=0.35, weight="bold", zorder=1000)


def save(fig, name: str, mock: bool = False) -> list[Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    stem = name + ("_MOCK" if mock else "")
    if mock:
        mock_watermark(fig)
    paths = [OUT / f"{stem}.pdf", OUT / f"{stem}.svg"]
    for p in paths:
        fig.savefig(p)
    plt.close(fig)
    print("wrote", *paths)
    return paths
