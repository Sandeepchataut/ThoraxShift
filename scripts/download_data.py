"""Download the NLM Shenzhen and/or Montgomery sets and write manifests.

Usage:
    python scripts/download_data.py --datasets shenzhen montgomery

Raw size is roughly 4 GB (the PNGs are about 5-6 MB each). Re-running skips files already present.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tbshift.data import nlm  # noqa: E402
from tbshift.provenance import DATA_ROOT, RAW_ROOT  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["montgomery", "shenzhen"],
                    choices=sorted(nlm.DATASETS))
    ap.add_argument("--data-root", type=Path, default=RAW_ROOT)
    args = ap.parse_args()
    manifests = DATA_ROOT / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        nlm.download(name, args.data_root)
        df = nlm.build_manifest(name, args.data_root)
        df.to_csv(manifests / f"{name}.csv", index=False)
        print(f"{name}: {len(df)} images, {int(df.label.sum())} abnormal, "
              f"{int((df.label == 0).sum())} normal -> {manifests / f'{name}.csv'}")


if __name__ == "__main__":
    main()
