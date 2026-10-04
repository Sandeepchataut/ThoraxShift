# Running the deep models on a GPU

The hand-crafted pipeline runs on CPU. The deep models need a CUDA GPU (e.g. Kaggle, Colab or a
cloud/HPC node).

1. On the CPU machine, build the shared inputs:
   `scripts/download_data.py`, `scripts/segment_lungs.py`, `scripts/prepare_images.py`.
2. Copy `data/manifests/`, `data/prepared/` and `data/masks/` (a few hundred MB) to the GPU
   machine, preserving the layout under a data root. Set `TBSHIFT_DATA` to that root.
3. Set `TBSHIFT_RUNS` to persistent storage, so run directories and checkpoints survive
   session limits.
4. Install with `pip install -r requirements.txt` (a CUDA build of torch 2.3.1 is selected
   automatically on Linux), then `pip install -e . --no-deps`.
5. Run `scripts/run_deep.py ...`. If the session ends, rerun the same command with
   `--resume $TBSHIFT_RUNS/<run_id>`; finished folds and seeds are skipped and training
   resumes from the last epoch checkpoint.
6. Copy the run directories back into `outputs/runs/` before running `scripts/aggregate.py`.
