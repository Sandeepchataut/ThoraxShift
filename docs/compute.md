# Running on a GPU machine

The hand-crafted pipeline and the sklearn heads run on CPU (they use every core). Feature
extraction and the MLP / attention heads use a CUDA GPU when one is visible.

1. Copy `data/manifests/`, `data/prepared/` and `data/masks/` to the machine (or build them there)
   and set `TBSHIFT_DATA` to that data root. Set `TBSHIFT_RUNS` to persistent storage.
2. `pip install -r requirements.txt` then `pip install -e . --no-deps`.
3. Download every backbone once (Hugging Face Hub, timm and TorchXRayVision caches):
   `python scripts/extract_deep_features.py --prefetch`. For an offline node, run this on a
   connected machine and copy `~/.cache/huggingface` and `~/.torchxrayvision`, then set
   `HF_HUB_OFFLINE=1`.
4. One machine, one GPU: `python scripts/run_experiments.py`. It skips finished work, so rerun the
   same command after an interruption.
5. Several GPUs or machines sharing `TBSHIFT_DATA` and `TBSHIFT_RUNS`:

   ```bash
   python scripts/run_experiments.py --stages check hc_features deep_features
   CUDA_VISIBLE_DEVICES=0 python scripts/run_experiments.py --stages handcrafted frozen --shard 0/4 &
   CUDA_VISIBLE_DEVICES=1 python scripts/run_experiments.py --stages handcrafted frozen --shard 1/4 &
   CUDA_VISIBLE_DEVICES=2 python scripts/run_experiments.py --stages handcrafted frozen --shard 2/4 &
   CUDA_VISIBLE_DEVICES=3 python scripts/run_experiments.py --stages handcrafted frozen --shard 3/4 &
   wait
   python scripts/run_experiments.py --stages probe aggregate figures
   ```

   Each shard's sklearn runs use all CPU cores, so on one machine expect CPU contention; spread
   the shards over machines when possible.
6. `outputs/runs/<stamp>_aggregate_benchmark/summary.md` is the first thing to read.

Disk: the token caches dominate (fp16; roughly 2–3 GB per ViT-B backbone and mask for TBX11K,
about 35 GB in total for 12 backbones × 2 masks).
