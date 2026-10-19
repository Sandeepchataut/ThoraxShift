"""Trainable heads on top of frozen backbone features.

All heads are scikit-learn estimators (fit / decision_function, cloneable), so they run through
exactly the same evaluation protocols as the hand-crafted model (tbshift.eval.protocol):
nested CV in-domain, fit-on-source / score-target cross-domain, threshold from the source only.

    logreg    pooled vector  standardise -> L2 logistic regression          (linear probe)
    svm_rbf   pooled vector  standardise -> RBF-SVM, SAME pipeline and grid as FUSION
    mlp       pooled vector  standardise -> 1 hidden layer (GELU, dropout)  (non-linear probe)
    attn_mil  token grid     standardise -> gated-attention pooling (Ilse et al., 2018) -> linear

svm_rbf is the primary head: with it, hand-crafted and deep models differ ONLY in the
representation. The sklearn heads choose hyperparameters by inner CV (grids below). The torch
heads use one pre-specified configuration with early stopping on a stratified validation split
of their own training portion (no inner grid; see docs/analysis_plan.md).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tbshift.models.classical import BlockWeighter, svm_grid, svm_rbf

LOGREG_C = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)


@dataclass(frozen=True)
class HeadSpec:
    name: str
    input: str           # "pooled" | "tokens"
    trainer: str         # "sklearn" | "torch"
    label: str
    supports_blocks: bool  # can take the hybrid (deep + hand-crafted) block-weighted input


HEADS = {
    "logreg": HeadSpec("logreg", "pooled", "sklearn", "Linear (logistic)", True),
    "svm_rbf": HeadSpec("svm_rbf", "pooled", "sklearn", "RBF-SVM", True),
    "mlp": HeadSpec("mlp", "pooled", "torch", "MLP", False),
    "attn_mil": HeadSpec("attn_mil", "tokens", "torch", "Attention pooling", False),
}
PRIMARY_HEAD = "svm_rbf"

# Pre-specified torch-head configurations (fixed before any experiment; not tuned).
TORCH_CONFIG = {
    "mlp": dict(arch="mlp", hidden=256, dropout=0.5, lr=1e-3, weight_decay=1e-2, batch_size=64,
                max_epochs=200, patience=25, token_drop=0.0),
    "attn_mil": dict(arch="attn_mil", hidden=256, attn_dim=128, dropout=0.25, token_drop=0.25,
                     lr=2e-4, weight_decay=1e-2, batch_size=32, max_epochs=100, patience=15),
}


def make(head: str, n_features: int, seed: int, block_sizes: tuple[int, ...] | None = None,
         device: str = "auto"):
    """(estimator, grid, n_jobs). The grid is searched by inner CV; a single-point grid means
    the configuration is fixed (the protocol then fits it directly, without an inner search)."""
    if head not in HEADS:
        raise KeyError(f"unknown head {head!r}; choose from {sorted(HEADS)}")
    spec = HEADS[head]
    if block_sizes is not None and not spec.supports_blocks:
        raise ValueError(f"head {head} does not support block-weighted (hybrid) inputs")
    if head == "svm_rbf":
        weighted = block_sizes is not None and len(block_sizes) > 1
        est = svm_rbf(seed, block_sizes if weighted else None)
        return est, svm_grid(n_features, n_blocks=len(block_sizes) if weighted else None), -1
    if head == "logreg":
        est = Pipeline([("scale", StandardScaler()),
                        ("blocks", BlockWeighter(block_sizes if block_sizes and len(block_sizes) > 1 else None)),
                        ("lr", LogisticRegression(C=1.0, class_weight="balanced", max_iter=10000,
                                                  random_state=seed))])
        return est, {"lr__C": list(LOGREG_C)}, -1
    est = TorchHead(**TORCH_CONFIG[head], seed=seed, device=device)
    return est, {"seed": [seed]}, 1   # single point: fixed configuration, no inner search


# ---------------------------------------------------------------------------------------------
# Torch heads
# ---------------------------------------------------------------------------------------------

def _device(requested: str):
    import torch
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _channel_stats(X: np.ndarray, rows: np.ndarray, chunk: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean and SD over the given rows (and over tokens for 3-D input), computed in
    float64 chunks so a memory-mapped fp16 token array is never materialised at once."""
    d = X.shape[-1]
    s = np.zeros(d); ss = np.zeros(d); n = 0
    for i in range(0, len(rows), chunk):
        a = np.asarray(X[np.sort(rows[i:i + chunk])], dtype=np.float64).reshape(-1, d)
        s += a.sum(0); ss += (a * a).sum(0); n += len(a)
    mu = s / n
    sd = np.sqrt(np.maximum(ss / n - mu * mu, 0.0))
    sd[sd < 1e-6] = 1.0
    return mu.astype(np.float32), sd.astype(np.float32)


def _build_net(arch: str, d_in: int, hidden: int, attn_dim: int, dropout: float):
    import torch
    from torch import nn

    if arch == "mlp":
        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.body = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout),
                                          nn.Linear(hidden, 1))

            def forward(self, x, token_drop: float = 0.0):
                return self.body(x).squeeze(-1)
        return MLP()

    if arch == "attn_mil":
        class GatedAttentionMIL(nn.Module):
            """h_t = MLP(x_t); a = softmax_t(w^T (tanh(V h_t) * sigmoid(U h_t))); z = sum_t a_t h_t."""

            def __init__(self):
                super().__init__()
                self.proj = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout))
                self.att_v = nn.Linear(hidden, attn_dim)
                self.att_u = nn.Linear(hidden, attn_dim)
                self.att_w = nn.Linear(attn_dim, 1)
                self.cls = nn.Linear(hidden, 1)

            def attention(self, h):
                return self.att_w(torch.tanh(self.att_v(h)) * torch.sigmoid(self.att_u(h))).squeeze(-1)

            def forward(self, x, token_drop: float = 0.0, return_attention: bool = False):
                h = self.proj(x)                                 # (B, T, H)
                logits = self.attention(h)                       # (B, T)
                if token_drop > 0:
                    keep = torch.rand(logits.shape, device=logits.device) >= token_drop
                    keep[:, 0] |= ~keep.any(dim=1)               # never drop every token
                    logits = logits.masked_fill(~keep, float("-inf"))
                a = torch.softmax(logits, dim=1)
                z = torch.einsum("bt,bth->bh", a, h)
                out = self.cls(z).squeeze(-1)
                return (out, a) if return_attention else out
        return GatedAttentionMIL()

    raise ValueError(f"unknown torch head {arch!r}")


class TorchHead(ClassifierMixin, BaseEstimator):
    """Small torch classifier on frozen features with early stopping on validation AUC.

    X: (N, D) pooled vectors or (N, T, D) token grids (any float dtype; may be memory-mapped).
    Standardisation statistics come from the training portion only. The validation split is a
    stratified `val_fraction` of the data passed to fit (never the outer test fold or target).
    """

    def __init__(self, arch="mlp", hidden=256, attn_dim=128, dropout=0.5, token_drop=0.0,
                 lr=1e-3, weight_decay=1e-2, batch_size=64, max_epochs=200, patience=25,
                 val_fraction=0.15, seed=0, device="auto"):
        self.arch = arch
        self.hidden = hidden
        self.attn_dim = attn_dim
        self.dropout = dropout
        self.token_drop = token_drop
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_fraction = val_fraction
        self.seed = seed
        self.device = device

    # -- helpers
    def _tensor(self, X, rows, device):
        import torch
        a = np.asarray(X[np.sort(rows)] if rows is not None else X)
        dtype = np.float16 if a.ndim == 3 else np.float32         # token grids stay compact on GPU
        return torch.from_numpy(np.ascontiguousarray(a, dtype=dtype)).to(device)

    def _norm(self, xb):
        return (xb.float() - self._mu) / self._sd

    def _scores(self, net, Xt, batch: int = 256) -> np.ndarray:
        import torch
        net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(Xt), batch):
                out.append(net(self._norm(Xt[i:i + batch])).float().cpu().numpy())
        return np.concatenate(out).astype(np.float64)

    def fit(self, X, y):
        import torch
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import train_test_split

        y = np.asarray(y).astype(int)
        if set(np.unique(y)) != {0, 1}:
            raise ValueError("TorchHead needs both classes in y")
        if self.arch == "attn_mil" and np.ndim(X) != 3:
            raise ValueError("attn_mil expects (N, T, D) token grids")
        if self.arch == "mlp" and np.ndim(X) != 2:
            raise ValueError("mlp expects (N, D) pooled vectors")
        idx = np.arange(len(y))
        tr, va = train_test_split(idx, test_size=self.val_fraction, stratify=y, random_state=self.seed)
        tr, va = np.sort(tr), np.sort(va)
        device = _device(self.device)
        mu, sd = _channel_stats(X, tr)
        self.mean_, self.scale_ = mu, sd
        self._mu = torch.from_numpy(mu).to(device)
        self._sd = torch.from_numpy(sd).to(device)

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        net = _build_net(self.arch, X.shape[-1], self.hidden, self.attn_dim, self.dropout).to(device)
        Xtr, Xva = self._tensor(X, tr, device), self._tensor(X, va, device)
        ytr = torch.from_numpy(y[tr].astype(np.float32)).to(device)
        yva = y[va]
        n_pos = float(ytr.sum().item())
        pos_weight = torch.tensor([(len(tr) - n_pos) / max(n_pos, 1.0)], device=device)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        gen = torch.Generator().manual_seed(self.seed)

        best_auc, best_state, best_epoch, stale = -np.inf, None, -1, 0
        self.history_ = []
        for epoch in range(self.max_epochs):
            net.train()
            perm = torch.randperm(len(tr), generator=gen).to(device)
            total = 0.0
            for b in range(0, len(tr), self.batch_size):
                bi = perm[b:b + self.batch_size]
                logits = net(self._norm(Xtr[bi]), token_drop=self.token_drop)
                loss = loss_fn(logits, ytr[bi])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total += float(loss.item()) * len(bi)
            val_auc = float(roc_auc_score(yva, self._scores(net, Xva)))
            self.history_.append({"epoch": epoch, "train_loss": total / len(tr), "val_auc": val_auc})
            if val_auc > best_auc + 1e-6:
                best_auc, best_epoch, stale = val_auc, epoch, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                stale += 1
                if stale >= self.patience:
                    break
        net.load_state_dict(best_state)
        self.net_ = net.cpu().eval()
        self._mu, self._sd = self._mu.cpu(), self._sd.cpu()
        self.best_val_auc_, self.best_epoch_, self.n_epochs_ = best_auc, best_epoch, len(self.history_)
        self.classes_ = np.array([0, 1])
        del Xtr, Xva
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return self

    def decision_function(self, X) -> np.ndarray:
        import torch
        device = _device(self.device)
        net = self.net_.to(device)
        self._mu, self._sd = self._mu.to(device), self._sd.to(device)
        out = []
        chunk = 1024
        for i in range(0, len(X), chunk):
            out.append(self._scores(net, self._tensor(X[i:i + chunk], None, device)))
        self.net_ = net.cpu()
        self._mu, self._sd = self._mu.cpu(), self._sd.cpu()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return np.concatenate(out)

    def predict(self, X):
        return (self.decision_function(X) >= 0).astype(int)

    def attention_maps(self, X) -> np.ndarray:
        """Per-token attention weights (N, T) for the attention head (for figures)."""
        import torch
        if self.arch != "attn_mil":
            raise ValueError("attention maps exist only for attn_mil")
        device = _device(self.device)
        net = self.net_.to(device).eval()
        mu, sd = self._mu.to(device), self._sd.to(device)
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 256):
                xb = self._tensor(X[i:i + 256], None, device)
                _, a = net((xb.float() - mu) / sd, return_attention=True)
                out.append(a.float().cpu().numpy())
        self.net_ = net.cpu()
        return np.concatenate(out)
