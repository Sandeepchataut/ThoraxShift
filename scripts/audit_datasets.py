"""Per-image acquisition properties (bit depth, size, intensity range) for each dataset.

These are potential shortcut cues: if they differ by dataset, a model can identify the source
without looking at disease. Output: outputs/runs/<id>/ with per_image.csv and a summary.

    python scripts/audit_datasets.py --datasets montgomery shenzhen
"""
import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tbshift.provenance import DATA_ROOT, new_run, resolve_data_path, write_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    args = ap.parse_args()
    rows = []
    for name in args.datasets:
        df = pd.read_csv(DATA_ROOT / "manifests" / f"{name}.csv")
        for r in tqdm(df.to_dict("records"), desc=name):
            a = cv2.imread(str(resolve_data_path(r["path"])), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
            rows.append({"dataset": name, "image_id": r["image_id"], "label": r["label"],
                         "dtype": str(a.dtype), "channels": 1 if a.ndim == 2 else a.shape[2],
                         "height": a.shape[0], "width": a.shape[1],
                         "min": int(a.min()), "max": int(a.max())})
    res = pd.DataFrame(rows)
    run = new_run("audit_" + "-".join(args.datasets), vars(args))
    res.to_csv(run / "per_image.csv", index=False)
    summary = {}
    for name, g in res.groupby("dataset"):
        summary[name] = {
            "n": len(g), "n_abnormal": int(g.label.sum()),
            "dtype_counts": g.dtype.value_counts().to_dict(),
            "channels_counts": g.channels.value_counts().to_dict(),
            "height_range": [int(g.height.min()), int(g.height.max())],
            "width_range": [int(g.width.min()), int(g.width.max())],
            "aspect_hw_range": [float((g.height / g.width).min()), float((g.height / g.width).max())],
            "max_value_range": [int(g["max"].min()), int(g["max"].max())],
        }
    write_json(run / "summary.json", summary)
    for k, v in summary.items():
        print(k, v)
    print(f"-> {run}")


if __name__ == "__main__":
    main()
