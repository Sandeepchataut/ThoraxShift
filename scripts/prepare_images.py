"""Cache preprocessed images (8-bit PNG) for one dataset.

    python scripts/prepare_images.py --dataset shenzhen

Deterministic and per-image (nothing is fitted across images), so caching is not leakage.
The cache plus data/masks and data/manifests is all a GPU machine needs for the deep models.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tbshift import preprocess  # noqa: E402
from tbshift.data.images import manifest, prepared_dir  # noqa: E402
from tbshift.provenance import cache_provenance, resolve_data_path  # noqa: E402


def _one(path: str, dest: Path, size: int) -> None:
    if dest.exists():
        return
    img = preprocess.prepare(resolve_data_path(path), size=size)
    cv2.imwrite(str(dest), np.round(img * 255).astype(np.uint8))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--size", type=int, default=preprocess.DEFAULT_SIZE)
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args()
    df = manifest(args.dataset)
    out = prepared_dir(args.dataset, args.size)
    out.mkdir(parents=True, exist_ok=True)
    Parallel(n_jobs=args.jobs, verbose=2)(
        delayed(_one)(r["path"], out / f"{r['image_id']}.png", args.size)
        for r in df.to_dict("records"))
    cache_provenance(out, {"dataset": args.dataset, "size": args.size, "n_images": len(df)})
    print(f"{len(list(out.glob('*.png')))} prepared images in {out}")


if __name__ == "__main__":
    main()
