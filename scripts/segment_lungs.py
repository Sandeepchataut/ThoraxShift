"""Segment lungs for every image in a dataset and cache the masks as PNGs.

    python scripts/segment_lungs.py --dataset montgomery --evaluate   # also Dice vs manual masks
    python scripts/segment_lungs.py --dataset shenzhen

Masks go to data/masks/<dataset>_s<size>/<image_id>.png. With --evaluate (Montgomery only, since
it has manual masks), a run directory is created holding per-image Dice values.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tbshift import preprocess  # noqa: E402
from tbshift.provenance import DATA_ROOT, cache_provenance, new_run, resolve_data_path, write_json  # noqa: E402
from tbshift.segmentation import LungSegmenter, dice  # noqa: E402


def mask_dir(dataset: str, size: int) -> Path:
    return DATA_ROOT / "masks" / f"{dataset}_s{size}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--size", type=int, default=preprocess.DEFAULT_SIZE)
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(DATA_ROOT / "manifests" / f"{args.dataset}.csv")
    if args.limit:
        df = df.head(args.limit)
    out = mask_dir(args.dataset, args.size)
    out.mkdir(parents=True, exist_ok=True)
    seg = LungSegmenter()
    print(f"segmenter weights sha256: {seg.weights_sha256}")
    rows = []
    for r in tqdm(df.to_dict("records"), desc=f"segment {args.dataset}"):
        dest = out / f"{r['image_id']}.png"
        if dest.exists():
            pred = cv2.imread(str(dest), cv2.IMREAD_GRAYSCALE) > 127
        else:
            img = preprocess.resize(preprocess.normalise(preprocess.load_gray(resolve_data_path(r["path"]))), args.size)
            pred = seg(img)
            cv2.imwrite(str(dest), pred.astype(np.uint8) * 255)
        row = {"image_id": r["image_id"], "label": r["label"], "mask_area_frac": float(pred.mean())}
        if args.evaluate:
            if not isinstance(r.get("mask_paths"), str) or not r["mask_paths"]:
                raise SystemExit(f"{args.dataset} has no manual masks; --evaluate not possible")
            gt = preprocess.load_mask([resolve_data_path(m) for m in r["mask_paths"].split(";")], args.size)
            row["dice"] = dice(pred, gt)
            inter = float((pred & gt).sum())
            row["precision"] = inter / max(pred.sum(), 1)       # fraction of predicted lung that is lung
            row["recall"] = inter / max(gt.sum(), 1)            # fraction of manual lung recovered
            row["area_ratio"] = float(pred.sum() / max(gt.sum(), 1))
        rows.append(row)

    res = pd.DataFrame(rows)
    cache_provenance(out, {"dataset": args.dataset, "size": args.size, "n_masks": len(res),
                           "empty_masks": int((res.mask_area_frac == 0).sum()),
                           "segmenter_weights_sha256": seg.weights_sha256})
    print(f"{len(res)} masks in {out}; empty masks: {(res.mask_area_frac == 0).sum()}")
    if args.evaluate:
        run = new_run(f"segmentation_dice_{args.dataset}",
                      {**vars(args), "segmenter_weights_sha256": seg.weights_sha256})
        res.to_csv(run / "per_image.csv", index=False)
        summary = {"n": len(res), "dice_mean": res.dice.mean(), "dice_median": res.dice.median(),
                   "dice_min": res.dice.min(), "dice_q05": res.dice.quantile(0.05),
                   "precision_mean": res.precision.mean(), "recall_mean": res.recall.mean(),
                   "area_ratio_mean": res.area_ratio.mean(), "area_ratio_median": res.area_ratio.median(),
                   "empty_masks": int((res.mask_area_frac == 0).sum()),
                   "by_label": res.groupby("label").dice.describe().to_dict()}
        write_json(run / "metrics.json", summary)
        print(f"Dice mean {summary['dice_mean']:.4f}, min {summary['dice_min']:.4f} -> {run}")


if __name__ == "__main__":
    main()
