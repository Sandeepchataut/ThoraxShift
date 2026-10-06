"""NLM Shenzhen and Montgomery tuberculosis CXR sets: download and manifest.

Files come from the public NLM index (one PNG per request) and land in

    <root>/<dataset>/CXR_png/<file>.png
    <root>/montgomery/ManualMask/{leftMask,rightMask}/<file>.png

File names are <prefix>_<4 digits>_<label>.png with label 1 = TB manifestation, 0 = normal.
Manifest paths are relative to RAW_ROOT (see provenance.resolve_data_path) when the files live
under it, absolute otherwise.
"""
from __future__ import annotations

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

from tbshift.provenance import RAW_ROOT

BASE = "https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets"
# dataset -> {local subfolder: remote folder}; the first entry holds the images.
DATASETS = {
    "montgomery": {
        "CXR_png": "Montgomery-County-CXR-Set/MontgomerySet/CXR_png",
        "ManualMask/leftMask": "Montgomery-County-CXR-Set/MontgomerySet/ManualMask/leftMask",
        "ManualMask/rightMask": "Montgomery-County-CXR-Set/MontgomerySet/ManualMask/rightMask",
    },
    "shenzhen": {
        "CXR_png": "Shenzhen-Hospital-CXR-Set/CXR_png",
    },
}
EXPECTED = {"montgomery": (138, 58), "shenzhen": (662, 336)}  # (images, TB) per the NLM release
_NAME = re.compile(r"^[A-Z]+_\d{4}_([01])\.png$")
_HEADERS = {"User-Agent": "Mozilla/5.0 (tbshift dataset download)"}


def _list(remote: str) -> list[str]:
    r = requests.get(f"{BASE}/{remote}/index.html", headers=_HEADERS, timeout=60)
    r.raise_for_status()
    names = re.findall(r"href=['\"]([^'\"/]+\.png)['\"]", r.text)
    if not names:
        raise RuntimeError(f"no PNG files listed at {BASE}/{remote}/")
    return sorted(set(names))


def _fetch(url: str, dest: Path, retries: int = 5) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    tmp = dest.with_name(f"{dest.name}.{os.getpid()}.part")  # per process: two jobs may share data/
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, headers=_HEADERS, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            tmp.replace(dest)
            return
        except (requests.RequestException, OSError):
            if attempt == retries:
                raise
            time.sleep(2 * attempt)


def download(name: str, root: Path = RAW_ROOT, workers: int = 8) -> None:
    """Fetch every file of one dataset into root/<name>/. Files already present are skipped."""
    jobs = []
    for local, remote in DATASETS[name].items():
        out = Path(root) / name / local
        out.mkdir(parents=True, exist_ok=True)
        jobs += [(f"{BASE}/{remote}/{f}", out / f) for f in _list(remote)]
    todo = [j for j in jobs if not (j[1].exists() and j[1].stat().st_size > 0)]
    print(f"{name}: {len(jobs)} files, {len(todo)} to download")
    with ThreadPoolExecutor(workers) as ex:
        for _ in ex.map(lambda j: _fetch(*j), todo):
            pass


def _rel(p: Path) -> str:
    p = p.resolve()
    try:
        return str(p.relative_to(RAW_ROOT.resolve()))
    except ValueError:
        return str(p)


def build_manifest(name: str, root: Path = RAW_ROOT) -> pd.DataFrame:
    """One row per image: dataset, image_id, label, path[, mask_paths (left;right)]."""
    img_dir = Path(root) / name / "CXR_png"
    rows = []
    for p in sorted(img_dir.glob("*.png")):
        m = _NAME.match(p.name)
        if m is None:
            continue
        row = {"dataset": name, "image_id": p.stem, "label": int(m.group(1)), "path": _rel(p)}
        if name == "montgomery":
            masks = [Path(root) / name / "ManualMask" / side / p.name
                     for side in ("leftMask", "rightMask")]
            missing = [str(q) for q in masks if not q.exists()]
            if missing:
                raise FileNotFoundError(f"manual masks missing: {missing}")
            row["mask_paths"] = ";".join(_rel(q) for q in masks)
        rows.append(row)
    df = pd.DataFrame(rows)
    n, n_pos = EXPECTED[name]
    if len(df) != n or int(df.label.sum()) != n_pos:
        raise RuntimeError(f"{name}: found {len(df)} images / {int(df.label.sum()) if len(df) else 0} TB "
                           f"in {img_dir}, expected {n} / {n_pos}. Re-run the download.")
    return df
