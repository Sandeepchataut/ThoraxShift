"""Extract and cache frozen-backbone features for every (backbone, dataset, mask).

    python scripts/extract_deep_features.py --datasets shenzhen montgomery tbx11k
    python scripts/extract_deep_features.py --backbones rad_dino dinov2_vitb14 --datasets shenzhen
    python scripts/extract_deep_features.py --prefetch          # download all weights, then exit

Inputs are the shared prepared images and automatic lung masks (identical to every other model
family). Existing caches are skipped unless --overwrite. Inference is fp32 by default (half
precision would coarsen features); --amp trades exactness for speed.
"""
import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tbshift import deep_features as dfc  # noqa: E402
from tbshift.data import images  # noqa: E402
from tbshift.models import backbones as bb  # noqa: E402
from tbshift.provenance import _git  # noqa: E402


class _Images:
    """Torch-style dataset over manifest rows: (1, S, S) float32 in [0, 1]."""

    def __init__(self, dataset: str, ids: list[str], size: int, mask: str):
        self.dataset, self.ids, self.size, self.mask = dataset, ids, size, mask

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        import torch
        img = images.load_input(self.dataset, self.ids[i], self.size, self.mask).astype(np.float32)
        return torch.from_numpy(img)[None], i


def pick_device(requested: str):
    import torch
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def extract_one(model, spec, dataset: str, mask: str, size: int, args, device) -> Path:
    import torch
    df = images.manifest(dataset)
    ids = df.image_id.tolist()
    loader = torch.utils.data.DataLoader(_Images(dataset, ids, size, mask), batch_size=args.batch_size,
                                         shuffle=False, num_workers=args.num_workers,
                                         pin_memory=device.type == "cuda")
    writer = None
    t0 = time.time()
    with torch.no_grad():
        for x, idx in loader:
            x = x.to(device, non_blocking=True)
            ctx = (torch.autocast("cuda", dtype=torch.float16) if args.amp and device.type == "cuda"
                   else contextlib.nullcontext())
            with ctx:
                pooled, tokens = model(x)
            pooled, tokens = pooled.float().cpu().numpy(), tokens.float().cpu().numpy()
            if writer is None:
                writer = dfc.CacheWriter(spec.name, dataset, mask, size, len(ids),
                                         pooled.shape[1], tokens.shape[1])
            if not np.isfinite(pooled).all() or not np.isfinite(tokens).all():
                raise ValueError(f"non-finite features for {spec.name} {dataset} batch {idx[:3].tolist()}")
            i = idx.numpy()
            writer.pooled[i] = pooled
            writer.tokens[i] = tokens.astype(np.float16)
    if writer is None:
        raise ValueError(f"{dataset}: empty manifest")
    pooled_arr = np.asarray(writer.pooled)
    meta = {"backbone": spec.name, "dataset": dataset, "mask": mask, "prepared_size": size,
            "n_images": len(ids), "pooled_shape": list(writer.pooled.shape),
            "tokens_shape": list(writer.tokens.shape), "token_grid": args.token_grid,
            "amp": bool(args.amp and device.type == "cuda"), "device": str(device),
            "pooled_sha256": bb.fingerprint_array(pooled_arr),
            "git_commit": _git("rev-parse", "HEAD"), "git_dirty": bool(_git("status", "--porcelain")),
            "seconds": round(time.time() - t0, 1), "command": sys.argv, **model.provenance}
    out = writer.commit(np.array(ids), df.label.to_numpy(), meta)
    print(f"  {spec.name} {dataset} mask={mask}: pooled {meta['pooled_shape']} tokens "
          f"{meta['tokens_shape']} in {meta['seconds']}s -> {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+", default=["all"])
    ap.add_argument("--datasets", nargs="+", default=[])
    ap.add_argument("--masks", nargs="+", default=["lung", "none"], choices=["lung", "none"])
    ap.add_argument("--size", type=int, default=512, help="prepared image size (the shared cache)")
    ap.add_argument("--token-grid", type=int, default=14)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--prefetch", action="store_true", help="only download/instantiate every backbone")
    args = ap.parse_args()

    import torch
    torch.manual_seed(0)
    device = pick_device(args.device)
    specs = bb.resolve(args.backbones)
    if not args.prefetch:
        if not args.datasets:
            raise SystemExit("--datasets is required (or use --prefetch)")
        for ds in args.datasets:
            print(images.check_caches(ds, args.size, need_masks=True))
    for spec in specs:
        todo = [(ds, m) for ds in args.datasets for m in args.masks
                if args.overwrite or not dfc.exists(spec.name, ds, m, args.size)]
        if not args.prefetch and not todo:
            print(f"{spec.name}: all caches present, skipping")
            continue
        print(f"{spec.name}: loading {spec.source}:{spec.ref}")
        model = bb.build(spec, token_grid=args.token_grid).to(device).eval()
        print(f"  params {model.provenance['n_params'] / 1e6:.1f}M  weights sha256 "
              f"{model.provenance['weights_sha256'][:16]}")
        if args.prefetch:
            with torch.no_grad():
                p, t = model(torch.zeros(1, 1, args.size, args.size, device=device))
            print(f"  ok: pooled {tuple(p.shape)} tokens {tuple(t.shape)}")
        else:
            for ds, m in todo:
                extract_one(model, spec, ds, m, args.size, args, device)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(json.dumps({"done": [s.name for s in specs]}))


if __name__ == "__main__":
    main()
