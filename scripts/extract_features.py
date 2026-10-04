"""Extract hand-crafted features for one dataset and cache them.

Per-image extraction is unsupervised and deterministic (the seed is derived from image_id), so
caching it before CV is not leakage. Anything fitted across images (scaling, codebooks) happens
later, inside the CV folds.

Usage:
    python scripts/extract_features.py --dataset montgomery --mask lung
"""
import argparse
import json
import sys
import zlib
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tbshift import preprocess  # noqa: E402
from tbshift.data import images  # noqa: E402
from tbshift.features import shape_context as sc  # noqa: E402
from tbshift.features import texture as tx  # noqa: E402
from tbshift.provenance import DATA_ROOT, _git  # noqa: E402


def _one(row: dict, mask_mode: str, size: int, scp, txp) -> dict:
    # Same cached pixels and masks as every other model family (scripts/prepare_images.py and
    # scripts/segment_lungs.py). Manual masks are never model inputs.
    img = images.read_prepared(row["dataset"], row["image_id"], size)
    mask = images.read_mask(row["dataset"], row["image_id"], size) if mask_mode == "lung" else None
    if mask_mode == "lung" and mask is None:
        raise ValueError(f"empty lung mask for {row['dataset']}/{row['image_id']}")
    seed = zlib.crc32(row["image_id"].encode())
    s = sc.extract(img, mask, scp, seed)
    t = tx.extract(img, mask, txp)
    return {"image_id": row["image_id"], "sc_pooled": s["pooled"],
            "sc_edge_density": s["edge_density"], "sc_n_edge_px": s["n_edge_px"],
            "sc_descriptors": s["descriptors"], **{f"tx_{k}": v for k, v in t.items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--mask", required=True, choices=["lung", "none"],
                    help="lung = automatic lung mask (primary); none = whole image (shortcut ablation)")
    ap.add_argument("--size", type=int, default=preprocess.DEFAULT_SIZE)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N images only")
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args()

    df = pd.read_csv(DATA_ROOT / "manifests" / f"{args.dataset}.csv")
    if args.limit:
        df = df.head(args.limit)
    scp, txp = sc.ShapeContextParams(), tx.TextureParams()
    rows = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(_one)(r, args.mask, args.size, scp, txp) for r in df.to_dict("records"))

    out = DATA_ROOT / "features" / f"{args.dataset}_mask-{args.mask}_s{args.size}"
    out.mkdir(parents=True, exist_ok=True)
    arrays = {"image_id": np.array([r["image_id"] for r in rows]),
              "label": df["label"].to_numpy()}
    for key in rows[0]:
        if key in ("image_id", "sc_descriptors"):
            continue
        arrays[key] = np.vstack([np.atleast_1d(r[key]) for r in rows])
    np.savez_compressed(out / "features.npz", **arrays)
    np.savez_compressed(out / "sc_descriptors.npz",
                        **{r["image_id"]: r["sc_descriptors"] for r in rows})
    (out / "params.json").write_text(json.dumps({
        "dataset": args.dataset, "mask": args.mask, "size": args.size, "n_images": len(rows),
        "shape_context": asdict(scp), "texture": asdict(txp), "git_commit": _git("rev-parse", "HEAD"),
    }, indent=2, default=str))
    print(f"wrote {out} :: " + ", ".join(f"{k}{v.shape}" for k, v in arrays.items()))


if __name__ == "__main__":
    main()
