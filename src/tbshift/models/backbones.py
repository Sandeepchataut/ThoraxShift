"""Frozen pretrained backbones used as fixed feature extractors.

Every backbone is wrapped so that it takes the SAME input as every other model family (the
cached, prepared 512 x 512 image in [0, 1], lung-masked or not) and returns

    pooled: (B, D)        the backbone's own global representation (CLS token for ViTs,
                          global average pool for CNNs), used by the vector heads;
    tokens: (B, T, D)     its last spatial feature map / patch tokens, average-pooled to at most
                          `token_grid` x `token_grid` cells, used by the attention head.

Only the input conversion differs between backbones and it follows each model's own
pretraining convention (resolution, channel count, normalisation). No weight is ever updated.

The registry is the single place where the set of models is defined. Groups and the
pre-specified pairwise contrasts are documented in docs/analysis_plan.md.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class BackboneSpec:
    name: str            # short id used in run names and tables
    source: str          # "timm" | "xrv" | "hf"
    ref: str             # timm model name / torchxrayvision weights / Hugging Face repo id
    input_size: int      # network input resolution (pretraining resolution)
    group: str           # see GROUPS
    family: str          # architecture family
    pretraining: str     # human-readable pretraining description
    label: str           # display name in figures and tables

    def to_dict(self) -> dict:
        return asdict(self)


GROUPS = {
    "cnn_in1k": "ImageNet-1k supervised CNN",
    "vit_in1k": "ImageNet-1k supervised transformer",
    "ssl_natural": "Self-supervised, natural images",
    "cxr": "Chest X-ray pretrained",
}

_SPECS = [
    BackboneSpec("resnet18", "timm", "resnet18.tv_in1k", 224, "cnn_in1k", "resnet",
                 "ImageNet-1k, supervised", "ResNet-18"),
    BackboneSpec("resnet50", "timm", "resnet50.tv_in1k", 224, "cnn_in1k", "resnet",
                 "ImageNet-1k, supervised", "ResNet-50"),
    BackboneSpec("densenet121", "timm", "densenet121.tv_in1k", 224, "cnn_in1k", "densenet",
                 "ImageNet-1k, supervised", "DenseNet-121"),
    BackboneSpec("efficientnet_b0", "timm", "efficientnet_b0.ra_in1k", 224, "cnn_in1k", "efficientnet",
                 "ImageNet-1k, supervised", "EfficientNet-B0"),
    BackboneSpec("convnext_tiny", "timm", "convnext_tiny.fb_in1k", 224, "cnn_in1k", "convnext",
                 "ImageNet-1k, supervised", "ConvNeXt-T"),
    BackboneSpec("swin_tiny", "timm", "swin_tiny_patch4_window7_224.ms_in1k", 224, "vit_in1k", "swin",
                 "ImageNet-1k, supervised", "Swin-T"),
    BackboneSpec("vit_small", "timm", "vit_small_patch16_224.augreg_in1k", 224, "vit_in1k", "vit",
                 "ImageNet-1k, supervised (AugReg)", "ViT-S/16"),
    BackboneSpec("vit_base", "timm", "vit_base_patch16_224.augreg_in1k", 224, "vit_in1k", "vit",
                 "ImageNet-1k, supervised (AugReg)", "ViT-B/16"),
    BackboneSpec("dinov2_vits14", "timm", "vit_small_patch14_dinov2.lvd142m", 518, "ssl_natural", "vit",
                 "LVD-142M, self-supervised (DINOv2)", "DINOv2 ViT-S/14"),
    BackboneSpec("dinov2_vitb14", "timm", "vit_base_patch14_dinov2.lvd142m", 518, "ssl_natural", "vit",
                 "LVD-142M, self-supervised (DINOv2)", "DINOv2 ViT-B/14"),
    BackboneSpec("xrv_densenet121", "xrv", "densenet121-res224-all", 224, "cxr", "densenet",
                 "Public CXR collections, supervised (TorchXRayVision 'all')", "DenseNet-121 (CXR)"),
    BackboneSpec("rad_dino", "hf", "microsoft/rad-dino", 518, "cxr", "vit",
                 "Public CXR collections, self-supervised (RAD-DINO)", "RAD-DINO ViT-B/14"),
]
BACKBONES: dict[str, BackboneSpec] = {s.name: s for s in _SPECS}

# Pre-specified pairwise contrasts (same head, same data): (id, question, model a, model b).
CONTRASTS = [
    ("pretrain_domain_densenet", "ImageNet vs chest X-ray pretraining (DenseNet-121)",
     "densenet121", "xrv_densenet121"),
    ("supervised_vs_ssl_vitb", "Supervised ImageNet vs self-supervised DINOv2 (ViT-B)",
     "vit_base", "dinov2_vitb14"),
    ("natural_vs_cxr_ssl_vitb", "Self-supervised natural vs chest X-ray images (ViT-B/14)",
     "dinov2_vitb14", "rad_dino"),
    ("cnn_vs_vit_small", "CNN vs plain transformer at ~25M parameters",
     "resnet50", "vit_small"),
    ("convnext_vs_swin", "Modern CNN vs hierarchical transformer at ~28M parameters",
     "convnext_tiny", "swin_tiny"),
    ("scale_resnet", "Scale within family (ResNet-18 vs 50)", "resnet18", "resnet50"),
    ("scale_vit", "Scale within family (ViT-S vs B)", "vit_small", "vit_base"),
    ("scale_dinov2", "Scale within family (DINOv2 S vs B)", "dinov2_vits14", "dinov2_vitb14"),
]


def get(name: str) -> BackboneSpec:
    if name not in BACKBONES:
        raise KeyError(f"unknown backbone {name!r}; choose from {sorted(BACKBONES)}")
    return BACKBONES[name]


def resolve(names: list[str] | None) -> list[BackboneSpec]:
    if not names or names == ["all"]:
        return list(_SPECS)
    return [get(n) for n in names]


# ---------------------------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------------------------

def _state_fingerprint(module) -> str:
    """sha256 over all parameters and buffers (names, shapes, bytes) in state-dict order."""
    h = hashlib.sha256()
    for k, v in module.state_dict().items():
        h.update(k.encode())
        t = v.detach().cpu().contiguous()
        if str(t.dtype) == "torch.bfloat16":
            t = t.float()
        h.update(str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def build(spec: BackboneSpec, token_grid: int = 14, pretrained: bool = True):
    """Instantiate the frozen extractor (downloads weights on first use).
    pretrained=False (timm only) builds the same architecture with random weights, for tests."""
    import torch

    class FrozenBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.spec = spec
            self.token_grid = token_grid
            self.provenance: dict = {"spec": spec.to_dict()}
            if spec.source == "timm":
                import timm
                self.net = timm.create_model(spec.ref, pretrained=pretrained, num_classes=0)
                cfg = timm.data.resolve_model_data_config(self.net)
                if tuple(cfg["input_size"][1:]) != (spec.input_size, spec.input_size):
                    raise ValueError(f"{spec.name}: registry input {spec.input_size} != timm "
                                     f"default {cfg['input_size']}")
                mean, std = cfg["mean"], cfg["std"]
                self.provenance.update({"timm_version": timm.__version__,
                                        "pretrained_cfg": dict(self.net.pretrained_cfg),
                                        "data_config": {k: (list(v) if isinstance(v, tuple) else v)
                                                        for k, v in cfg.items()}})
            elif not pretrained:
                raise ValueError("pretrained=False is only supported for timm backbones")
            elif spec.source == "xrv":
                import torchxrayvision as xrv
                self.net = xrv.models.DenseNet(weights=spec.ref)
                mean, std = None, None
                self.provenance.update({"torchxrayvision_version": xrv.__version__,
                                        "input_convention": "1 channel, [0,1] -> [-1024, 1024]"})
            elif spec.source == "hf":
                import transformers
                from transformers import AutoImageProcessor, AutoModel
                self.net = AutoModel.from_pretrained(spec.ref)
                proc = AutoImageProcessor.from_pretrained(spec.ref)
                mean, std = list(proc.image_mean), list(proc.image_std)
                crop = proc.crop_size["height"] if isinstance(proc.crop_size, dict) else proc.crop_size
                if int(crop) != spec.input_size:
                    raise ValueError(f"{spec.name}: registry input {spec.input_size} != processor {crop}")
                self.provenance.update({"transformers_version": transformers.__version__,
                                        "hf_commit": getattr(self.net.config, "_commit_hash", None),
                                        "image_mean": mean, "image_std": std})
            else:
                raise ValueError(spec.source)
            self.net.eval()
            for p in self.net.parameters():
                p.requires_grad_(False)
            if mean is not None:
                self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1))
                self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, -1, 1, 1))
            self.provenance["weights_sha256"] = _state_fingerprint(self.net)
            self.provenance["n_params"] = int(sum(p.numel() for p in self.net.parameters()))

        # -- input conversion: [0,1] grey (B,1,H,W) -> the backbone's own convention
        def _to_input(self, x01):
            import torch.nn.functional as F
            s = self.spec.input_size
            x = x01
            if x.shape[-1] != s or x.shape[-2] != s:
                x = F.interpolate(x, size=(s, s), mode="bicubic", align_corners=False,
                                  antialias=True).clamp_(0.0, 1.0)
            if self.spec.source == "xrv":
                return x * 2048.0 - 1024.0
            x = x.expand(-1, 3, -1, -1)
            return (x - self.mean) / self.std

        def _grid(self, fmap):
            """(B, D, h, w) -> (B, T, D) with T <= token_grid^2 via adaptive average pooling."""
            import torch.nn.functional as F
            h, w = fmap.shape[-2:]
            g = (min(h, self.token_grid), min(w, self.token_grid))
            if (h, w) != g:
                fmap = F.adaptive_avg_pool2d(fmap, g)
            return fmap.flatten(2).transpose(1, 2)

        def forward(self, x01):
            import torch
            import torch.nn.functional as F
            x = self._to_input(x01)
            if self.spec.source == "timm":
                feats = self.net.forward_features(x)
                pooled = self.net.forward_head(feats, pre_logits=True)
                if feats.ndim == 3:                                  # ViT: (B, prefix + N, D)
                    tok = feats[:, getattr(self.net, "num_prefix_tokens", 1):]
                    gh, gw = self.net.patch_embed.grid_size
                    fmap = tok.transpose(1, 2).reshape(tok.shape[0], tok.shape[2], gh, gw)
                elif str(getattr(self.net, "output_fmt", "NCHW")).endswith("NHWC"):  # Swin
                    fmap = feats.permute(0, 3, 1, 2)
                else:                                                # CNN: (B, D, h, w)
                    fmap = feats
            elif self.spec.source == "xrv":
                fmap = F.relu(self.net.features(x))                  # same as xrv features2
                pooled = fmap.mean(dim=(2, 3))
            else:                                                    # HF DINOv2-style model
                out = self.net(pixel_values=x)
                hs = out.last_hidden_state                           # (B, 1 + N, D), layer-normed
                pooled = hs[:, 0]
                tok = hs[:, 1:]
                n = int(round(tok.shape[1] ** 0.5))
                if n * n != tok.shape[1]:
                    raise ValueError(f"{self.spec.name}: {tok.shape[1]} patch tokens is not a square grid")
                fmap = tok.transpose(1, 2).reshape(tok.shape[0], tok.shape[2], n, n)
            if pooled.ndim != 2:
                raise RuntimeError(f"{self.spec.name}: pooled output has shape {tuple(pooled.shape)}")
            tokens = self._grid(fmap)
            if tokens.shape[-1] != pooled.shape[-1]:
                # Not an error for every architecture, but it is for all registered ones.
                raise RuntimeError(f"{self.spec.name}: token dim {tokens.shape[-1]} != pooled dim "
                                   f"{pooled.shape[-1]}")
            return torch.nan_to_num(pooled.float()), torch.nan_to_num(tokens.float())

    return FrozenBackbone()


def fingerprint_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
