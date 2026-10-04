"""Deep baselines: ImageNet-initialised CNNs, fully fine-tuned with one fixed recipe.

The recipe is fixed in advance (no per-dataset tuning). Early stopping and the operating
threshold use a validation split carved out of the TRAINING data only.
"""
from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from tbshift.data.images import load_input


@dataclass(frozen=True)
class DeepRecipe:
    arch: str = "densenet121"
    pretrained: bool = True            # ImageNet initialisation
    input_size: int = 384              # network input (resized from the 512 prepared image)
    epochs: int = 30
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 5                  # early stopping on validation AUC
    val_fraction: float = 0.15         # of the training data, stratified
    rotate_deg: float = 5.0
    translate: float = 0.05
    scale: tuple[float, float] = (0.9, 1.1)
    brightness: float = 0.1
    contrast: float = 0.1
    num_workers: int = 2
    amp: bool = True                   # mixed precision on CUDA only
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def pick_device(requested: str = "auto"):
    import torch
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(arch: str, pretrained: bool):
    """Backbone with a single-logit head."""
    import torch.nn as nn
    import torchvision.models as tvm

    if arch == "densenet121":
        m = tvm.densenet121(weights=tvm.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier = nn.Linear(m.classifier.in_features, 1)
    elif arch == "resnet50":
        m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, 1)
    elif arch == "efficientnet_b0":
        m = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 1)
    else:
        raise ValueError(f"unknown arch {arch}")
    return m


class CXRDataset:
    """Torch dataset over (dataset, image_id, label) rows; grey image replicated to 3 channels,
    ImageNet-normalised. Augmentation only when train=True."""

    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def __init__(self, rows: list[tuple[str, str, int]], prepared_size: int, mask_mode: str,
                 recipe: DeepRecipe, train: bool):
        import torch
        from torchvision.transforms import v2 as T

        self.rows, self.prepared_size, self.mask_mode, self.train = rows, prepared_size, mask_mode, train
        self.torch = torch
        aug = [T.RandomAffine(degrees=recipe.rotate_deg, translate=(recipe.translate, recipe.translate),
                              scale=recipe.scale),
               T.ColorJitter(brightness=recipe.brightness, contrast=recipe.contrast)] if train else []
        # No horizontal flip: left/right anatomy is not symmetric (heart, aortic knob).
        self.tf = T.Compose([T.Resize((recipe.input_size, recipe.input_size), antialias=True), *aug,
                             T.Normalize(self.MEAN, self.STD)])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        ds, image_id, label = self.rows[i]
        img = load_input(ds, image_id, self.prepared_size, self.mask_mode).astype(np.float32)
        x = self.torch.from_numpy(img)[None].repeat(3, 1, 1)
        return self.tf(x), self.torch.tensor(float(label))


def _loader(ds, recipe: DeepRecipe, shuffle: bool, seed: int, device):
    import torch
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.utils.data.DataLoader(ds, batch_size=recipe.batch_size, shuffle=shuffle,
                                       num_workers=recipe.num_workers, generator=g,
                                       pin_memory=device.type == "cuda",
                                       persistent_workers=recipe.num_workers > 0)


def predict(model, rows, prepared_size: int, mask_mode: str, recipe: DeepRecipe, device) -> np.ndarray:
    """Logits (higher = abnormal) for rows, in order."""
    import torch
    model.eval()
    out = []
    dl = _loader(CXRDataset(rows, prepared_size, mask_mode, recipe, train=False), recipe, False, 0, device)
    with torch.no_grad():
        for x, _ in dl:
            with torch.autocast(device.type, enabled=recipe.amp and device.type == "cuda"):
                out.append(model(x.to(device, non_blocking=True)).float().squeeze(1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def train(train_rows, val_rows, prepared_size: int, mask_mode: str, recipe: DeepRecipe, seed: int,
          device, ckpt_dir: Path, log=print):
    """Fine-tune with early stopping on validation AUC. Resumable from ckpt_dir/last.pt.

    Returns (best model, history list). The best state is also saved as ckpt_dir/best.pt.
    """
    import torch
    from sklearn.metrics import roc_auc_score

    from tbshift.provenance import set_seed

    set_seed(seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(recipe.arch, recipe.pretrained).to(device)
    y_tr = np.array([r[2] for r in train_rows])
    pos_weight = torch.tensor([(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)], device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=recipe.lr, weight_decay=recipe.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=recipe.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=recipe.amp and device.type == "cuda")
    y_val = np.array([r[2] for r in val_rows])

    start, best_auc, best_state, stale, history = 0, -np.inf, None, 0, []
    last = ckpt_dir / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        start, best_auc, stale, history = ck["epoch"] + 1, ck["best_auc"], ck["stale"], ck["history"]
        best_state = torch.load(ckpt_dir / "best.pt", map_location="cpu") if (ckpt_dir / "best.pt").exists() else None
        log(f"resumed from epoch {ck['epoch']}")

    dl = _loader(CXRDataset(train_rows, prepared_size, mask_mode, recipe, train=True), recipe, True,
                 seed, device)
    for epoch in range(start, recipe.epochs):
        if stale >= recipe.patience:
            break
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for x, y in dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=recipe.amp and device.type == "cuda"):
                loss = loss_fn(model(x).squeeze(1), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += float(loss) * len(y); n += len(y)
        sched.step()
        val_scores = predict(model, val_rows, prepared_size, mask_mode, recipe, device)
        val_auc = float(roc_auc_score(y_val, val_scores))
        if val_auc > best_auc:
            best_auc, stale = val_auc, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            torch.save(best_state, ckpt_dir / "best.pt")
        else:
            stale += 1
        history.append({"epoch": epoch, "train_loss": total / max(n, 1), "val_auc": val_auc,
                        "seconds": round(time.time() - t0, 1)})
        log(json.dumps(history[-1]))
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "best_auc": best_auc,
                    "stale": stale, "history": history}, last)

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history
