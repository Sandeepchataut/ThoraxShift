"""TBX11K: locate (or fetch from Kaggle) the raw images and write data/manifests/tbx11k.csv.

Usage:
    python scripts/download_tbx11k.py [--sick exclude|negative]

TBX11K has no public direct download (Google Drive / Baidu / Kaggle, all behind a login), so
the raw data is expected under data/raw/tbx11k/ in either layout:

    original release (mmcheng.net/tb):  .../imgs/{health,sick,tb}/*.png
    Kaggle "tbx11k-simplified":         .../data.csv + .../images/*.png

If neither is there and the kaggle CLI is configured (~/.kaggle/kaggle.json), the simplified
version is downloaded. Only train + val images are used (test labels are not public).

Labels (TB-related abnormality vs normal, docs/analysis_plan.md): tb (active + latent) = 1,
health = 0, sick-but-non-TB excluded by default (--sick negative counts it as 0). The manifest
keeps the source category in a "category" column.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tbshift.provenance import DATA_ROOT, RAW_ROOT  # noqa: E402

KAGGLE_SLUG = "vbookshelf/tbx11k-simplified"
CATEGORIES = ("health", "sick", "tb")


def _rel(p: Path) -> str:
    p = p.resolve()
    try:
        return str(p.relative_to(RAW_ROOT.resolve()))
    except ValueError:
        return str(p)


def _category(value: str) -> str:
    v = str(value).strip().lower()
    if v.startswith("health"):
        return "health"
    if v.startswith("sick"):
        return "sick"
    if v in ("tb", "active_tb", "latent_tb", "active_tb&latent_tb", "active&latent_tb"):
        return "tb"
    raise ValueError(f"unknown TBX11K category {value!r}")


def _from_original(root: Path) -> pd.DataFrame | None:
    imgs = next((p for p in root.rglob("imgs") if p.is_dir() and (p / "health").is_dir()), None)
    if imgs is None:
        return None
    rows = [{"image_id": f"{c}_{p.stem}", "category": c, "path": _rel(p)}
            for c in CATEGORIES for p in sorted((imgs / c).glob("*.png"))]
    return pd.DataFrame(rows)


def _from_simplified(root: Path) -> pd.DataFrame | None:
    csv = next(iter(sorted(root.rglob("data.csv"))), None)
    if csv is None:
        return None
    df = pd.read_csv(csv)
    fcol = next((c for c in ("fname", "filename", "file_name", "image") if c in df.columns), None)
    if fcol is None or "image_type" not in df.columns:
        raise ValueError(f"{csv}: unrecognised columns {list(df.columns)}")
    if "source" in df.columns:
        df = df[df.source.astype(str).str.lower().isin(["train", "val"])]
    df = df.drop_duplicates(fcol)  # one row per bounding box in the CSV
    img_dir = next((d for d in (csv.parent / "images", csv.parent / "imgs") if d.is_dir()), csv.parent)
    rows = []
    for r in df.itertuples():
        c = _category(getattr(r, "image_type"))
        p = img_dir / str(getattr(r, fcol))
        if not p.exists():
            raise FileNotFoundError(p)
        rows.append({"image_id": f"{c}_{p.stem}", "category": c, "path": _rel(p)})
    return pd.DataFrame(rows)


def _kaggle_download(root: Path) -> None:
    exe = shutil.which("kaggle")
    if exe is None or not (Path.home() / ".kaggle" / "kaggle.json").exists():
        sys.exit(f"TBX11K not found under {root} and the kaggle CLI is not configured.\n"
                 f"Copy the dataset there (original 'imgs/{{health,sick,tb}}' layout or Kaggle "
                 f"'{KAGGLE_SLUG}'), or install kaggle and add ~/.kaggle/kaggle.json, then re-run.")
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run([exe, "datasets", "download", "-d", KAGGLE_SLUG, "-p", str(root), "--unzip"],
                   check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=RAW_ROOT / "tbx11k")
    ap.add_argument("--sick", choices=["exclude", "negative"], default="exclude",
                    help="how to treat sick-but-non-TB images")
    args = ap.parse_args()

    df = _from_original(args.data_root) if args.data_root.exists() else None
    if df is None and args.data_root.exists():
        df = _from_simplified(args.data_root)
    if df is None:
        _kaggle_download(args.data_root)
        df = _from_original(args.data_root)
        if df is None:
            df = _from_simplified(args.data_root)
    if df is None or df.empty:
        sys.exit(f"no TBX11K images found under {args.data_root}")

    if args.sick == "exclude":
        df = df[df.category != "sick"]
    df = df.assign(dataset="tbx11k", label=(df.category == "tb").astype(int))
    df = df[["dataset", "image_id", "label", "path", "category"]].reset_index(drop=True)
    if df.image_id.duplicated().any():
        raise ValueError("duplicate TBX11K image ids")

    manifests = DATA_ROOT / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    out = manifests / "tbx11k.csv"
    df.to_csv(out, index=False)
    print(f"tbx11k: {len(df)} images, {int(df.label.sum())} TB, {int((df.label == 0).sum())} normal "
          f"(sick-but-non-TB: {args.sick}) -> {out}")


if __name__ == "__main__":
    main()
