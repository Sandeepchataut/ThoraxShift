"""Shortcut control: dataset-identification probe within each label class.

    python scripts/run_shortcut_probe.py --datasets montgomery shenzhen

Representations: every hand-crafted family and their fusion (from the feature caches), plus the
pooled features of every frozen backbone (from scripts/extract_deep_features.py). Each is probed
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

from tbshift import deep_features as dfc  # noqa: E402
from tbshift.models import backbones as bb  # noqa: E402
from tbshift.eval.shortcut import PROBE_C, dataset_id_probe  # noqa: E402
from tbshift.provenance import new_run, write_json  # noqa: E402


def frozen_features(backbone: str, dataset: str, mask: str, size: int):
    """(pooled features, labels, ids) of a frozen backbone, from the deep-feature cache."""
    X, y, ids = dfc.load(backbone, dataset, mask, size, kind="pooled")
    return np.asarray(X, dtype=np.float64), y, ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--masks", nargs="+", default=["lung", "none"])
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--backbones", nargs="+", default=["all"])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-deep", action="store_true")
    ap.add_argument("--n-perm", type=int, default=200, help="label-permutation null size")
    ap.add_argument("--jobs", type=int, default=-1, help="parallel workers for the permutation null")
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
            if not args.no_deep:
                for spec in bb.resolve(args.backbones):
                    X, y, _ = frozen_features(spec.name, ds, mask, args.size)
                    data[(ds, f"frozen-{spec.name}")] = (X, y, None)
        rep_names = sorted({k[1] for k in data})
        for a, b in itertools.combinations(args.datasets, 2):
            for rep in rep_names:
                Xa, ya, blocks = data[(a, rep)]
                Xb, yb, _ = data[(b, rep)]
                for cls, cname in ((0, "normal"), (1, "abnormal")):
                    res = dataset_id_probe(Xa[ya == cls], Xb[yb == cls], repeats=args.repeats,
                                           seed=args.seed, block_sizes=blocks, n_perm=args.n_perm,
                                           n_jobs=args.jobs)
                    rows.append({"dataset_a": a, "dataset_b": b, "representation": rep, "mask": mask,
                                 "label_class": cname, **{k: v for k, v in res.items()
                                                          if k != "balanced_accuracy_per_repeat"}})
                    print(f"{mask:4s} {cname:8s} {rep:24s} BA {res['balanced_accuracy_mean']:.3f} "
                          f"(null q95 {res.get('null_ba_q95', float('nan')):.3f}, p {res.get('perm_p', float('nan')):.4f})")
    table = pd.DataFrame(rows)
    table.to_csv(run / "probe.csv", index=False)
    write_json(run / "metrics.json", {"model_family": "probe", "protocol": "dataset_id_probe",
                                      "datasets": args.datasets, "rows": rows})
    print(f"run saved: {run}")


if __name__ == "__main__":
    main()
