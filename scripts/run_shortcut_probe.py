"""Shortcut control: dataset-identification probe within each label class.

    python scripts/run_shortcut_probe.py --datasets montgomery shenzhen

Representations: every hand-crafted family and their fusion (from the feature caches), plus frozen
ImageNet DenseNet-121 features (global-average-pooled, computed here and cached). Each is probed
on lung-masked and unmasked inputs, separately for normal and abnormal images.
"""
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_classical import FEATURE_ORDER, load  # noqa: E402

from tbshift.data.images import manifest  # noqa: E402
from tbshift.eval.shortcut import PROBE_C, dataset_id_probe  # noqa: E402
from tbshift.provenance import DATA_ROOT, new_run, write_json  # noqa: E402


def frozen_cnn_features(dataset: str, mask: str, prepared_size: int, input_size: int, device: str):
    """(features, labels, ids) from an ImageNet DenseNet-121 with no training (cached)."""
    cache = DATA_ROOT / "features" / f"frozen-densenet121_{dataset}_mask-{mask}_in{input_size}.npz"
    if cache.exists():
        f = np.load(cache)
        return f["X"], f["label"], f["image_id"]
    import torch
    import torchvision.models as tvm

    from tbshift.models.deep import CXRDataset, DeepRecipe, pick_device

    dev = pick_device(device)
    m = tvm.densenet121(weights=tvm.DenseNet121_Weights.IMAGENET1K_V1)
    m.classifier = torch.nn.Identity()
    m = m.eval().to(dev)
    df = manifest(dataset)
    rows = [(dataset, r.image_id, int(r.label)) for r in df.itertuples()]
    recipe = DeepRecipe(input_size=input_size, num_workers=0)
    ds = CXRDataset(rows, prepared_size, mask, recipe, train=False)
    feats = []
    with torch.no_grad():
        for i in range(0, len(ds), 16):
            x = torch.stack([ds[j][0] for j in range(i, min(i + 16, len(ds)))]).to(dev)
            feats.append(m(x).cpu().numpy())
    X = np.vstack(feats)
    y = np.array([r[2] for r in rows]); ids = np.array([r[1] for r in rows])
    np.savez_compressed(cache, X=X, label=y, image_id=ids)
    return X, y, ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--masks", nargs="+", default=["lung", "none"])
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--cnn-input-size", type=int, default=384)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-cnn", action="store_true")
    args = ap.parse_args()

    reps = [[f] for f in FEATURE_ORDER] + [FEATURE_ORDER]
    run = new_run("shortcut_probe_" + "-".join(args.datasets), {**vars(args), "probe_C": PROBE_C})
    rows = []
    for mask in args.masks:
        data = {}
        for ds in args.datasets:
            for rep in reps:
                X, y, _, blocks, _ = load(ds, mask, args.size, rep)
                data[(ds, "+".join(rep))] = (X, y, blocks if len(rep) > 1 else None)
            if not args.no_cnn:
                X, y, _ = frozen_cnn_features(ds, mask, args.size, args.cnn_input_size, args.device)
                data[(ds, "frozen-densenet121")] = (X, y, None)
        rep_names = sorted({k[1] for k in data})
        for a, b in itertools.combinations(args.datasets, 2):
            for rep in rep_names:
                Xa, ya, blocks = data[(a, rep)]
                Xb, yb, _ = data[(b, rep)]
                for cls, cname in ((0, "normal"), (1, "abnormal")):
                    res = dataset_id_probe(Xa[ya == cls], Xb[yb == cls], repeats=args.repeats,
                                           seed=args.seed, block_sizes=blocks)
                    rows.append({"dataset_a": a, "dataset_b": b, "representation": rep, "mask": mask,
                                 "label_class": cname, **{k: v for k, v in res.items()
                                                          if k != "balanced_accuracy_per_repeat"}})
                    print(f"{mask:4s} {cname:8s} {rep:24s} BA {res['balanced_accuracy_mean']:.3f}")
    table = pd.DataFrame(rows)
    table.to_csv(run / "probe.csv", index=False)
    write_json(run / "metrics.json", {"model_family": "probe", "protocol": "dataset_id_probe",
                                      "datasets": args.datasets, "rows": rows})
    print(f"run saved: {run}")


if __name__ == "__main__":
    main()
