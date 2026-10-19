# Analysis plan (v2)

Status: **draft v2**. Freeze it (git tag `plan-frozen-v2`, pushed to the public repository so the
date is independently timestamped) before any run that feeds a reported number. Every later change
is reported as a deviation. The code mirrors this plan in `src/tbshift/study.py`,
`src/tbshift/models/backbones.py` and `src/tbshift/models/heads.py`; change them together.

## What changed from v1, and why
- **Deep models are frozen pretrained backbones with trainable heads**, not a fully fine-tuned
  DenseNet-121. With 110–530 training images per fold, full fine-tuning mostly measures the
  optimisation recipe and overfitting. Freezing the backbone lets the deep model use **exactly the
  classifier FUSION uses** (standardise → RBF-SVM, same grid), so the comparison isolates the
  representation.
- **Twelve backbones chosen to answer specific questions** (architecture, pretraining data,
  pretraining objective, scale), not one model per family. VGG and Mamba-style models are left
  out: VGG adds nothing these do not already cover, and vision-Mamba weights and kernels are not
  mature enough for a reproducible benchmark.
- **Four heads**, so the result cannot be explained by the choice of classifier.
- **Hybrid analysis** (deep + hand-crafted features): do shape descriptors add information?
- The v1 fine-tuning path is kept as an optional, descriptive reference only.

## Research questions
1. **(Primary)** When a model is trained on one public TB chest X-ray collection and tested on
   another, is the out-of-domain AUC of the hand-crafted thoracic representation (FUSION)
   different from that of frozen pretrained deep representations, **with the classifier held fixed**?
2. Which properties of a deep representation go with better transfer: inductive bias (CNN vs
   transformer), pretraining data (ImageNet vs chest X-rays), pretraining objective (supervised vs
   self-supervised), model size?
3. Does the choice of head (linear, kernel, MLP, attention pooling) change the conclusions?
4. Are hand-crafted shape/texture features complementary to deep features?
5. How much acquisition-site information does each representation carry (shortcut risk), with
   and without lung masking?

## Task and data
- Binary classification of frontal chest radiographs: TB-related abnormality (1) vs normal (0).
- Datasets: Shenzhen and Montgomery (U.S. National Library of Medicine) and TBX11K (train + val
  splits; test labels are not public). Images are the unit of analysis.
- **TBX11K label definition must be written down before the freeze**: which of its categories
  (healthy, sick-but-non-TB, active/latent TB) are 1, which are 0 and which are excluded. The
  manifest `data/manifests/tbx11k.csv` is the record.
- Before any experiment: perceptual-hash check for duplicate images within and across datasets.

## Preprocessing and inputs (identical for every model family)
- Per-image 1st–99th percentile window, pad to square, resize to 512 × 512, CLAHE. Cached once
  (`scripts/prepare_images.py`).
- Automatic lung segmentation with one pretrained model for every dataset (weights hash
  recorded), validated by Dice against the Montgomery manual masks. Manual masks are never inputs.
- **Primary: lung-masked inputs.** Hand-crafted features as in v1 (shape context on edges inside
  the 15 px-dilated lung mask; LBP/GLCM/Gabor/HOG inside the lung). Deep models receive the
  prepared image with everything outside the 15 px-dilated lung mask set to zero.
- Each backbone then converts this common input to its own pretraining convention only:
  bicubic resize to its pretraining resolution, grey replicated to 3 channels and normalised with
  its own mean/std (TorchXRayVision: 1 channel scaled to [−1024, 1024]).
- An empty lung mask is an error, never a silent fallback. Unmasked inputs are used only for the
  shortcut ablation.

## Representations (fixed in advance)
**Hand-crafted (primary): FUSION** = shape context + LBP + GLCM + Gabor + HOG, each block
standardised and scaled by 1/sqrt(block dimension). Individual families are secondary.

**Frozen backbones** (no weight is ever updated; no augmentation; features cached once):

| id | weights | input | group |
|---|---|---|---|
| resnet18 | timm `resnet18.tv_in1k` | 224 | ImageNet-1k CNN |
| resnet50 | timm `resnet50.tv_in1k` | 224 | ImageNet-1k CNN |
| densenet121 | timm `densenet121.tv_in1k` | 224 | ImageNet-1k CNN |
| efficientnet_b0 | timm `efficientnet_b0.ra_in1k` | 224 | ImageNet-1k CNN |
| convnext_tiny | timm `convnext_tiny.fb_in1k` | 224 | ImageNet-1k CNN |
| swin_tiny | timm `swin_tiny_patch4_window7_224.ms_in1k` | 224 | ImageNet-1k transformer |
| vit_small | timm `vit_small_patch16_224.augreg_in1k` | 224 | ImageNet-1k transformer |
| vit_base | timm `vit_base_patch16_224.augreg_in1k` | 224 | ImageNet-1k transformer |
| dinov2_vits14 | timm `vit_small_patch14_dinov2.lvd142m` | 518 | self-supervised, natural images |
| dinov2_vitb14 | timm `vit_base_patch14_dinov2.lvd142m` | 518 | self-supervised, natural images |
| xrv_densenet121 | TorchXRayVision `densenet121-res224-all` | 224 | chest X-ray, supervised |
| rad_dino | Hugging Face `microsoft/rad-dino` | 518 | chest X-ray, self-supervised |

All ImageNet models use ImageNet-1k-only weights, so "ImageNet" means the same data for CNNs and
transformers. Each backbone yields a **pooled vector** (CLS token or global average pool) and a
**token grid** (last feature map / patch tokens, average-pooled to at most 14 × 14).

**Leakage check before the freeze:** confirm from the model cards / papers that neither
chest X-ray model saw Shenzhen, Montgomery or TBX11K during pretraining. RAD-DINO's card lists
MIMIC-CXR, CheXpert, NIH-CXR, PadChest and BRAX. The TorchXRayVision `all` weights must be checked
against its paper; if they include an NLM TB set, they are reported but excluded from the
primary family (deviation).

**Pre-specified contrasts** (same head, same data; model a − model b):
DenseNet-121 ImageNet vs CXR; ViT-B supervised vs DINOv2-B; DINOv2-B vs RAD-DINO;
ResNet-50 vs ViT-S; ConvNeXt-T vs Swin-T; ResNet-18 vs 50; ViT-S vs B; DINOv2-S vs B.
Known confound, reported: the DINOv2 and RAD-DINO models run at 518 px, the others at 224 px
(each at its pretraining resolution).

## Heads (fixed in advance)
| head | input | model | hyperparameters |
|---|---|---|---|
| **svm_rbf (primary)** | pooled | standardise → RBF-SVM, class-balanced | exactly the FUSION grid: C ∈ {0.01 … 1000}, γ ∈ {0.1, 0.3, 1, 3, 10}/d; inner 5-fold CV |
| logreg | pooled | standardise → L2 logistic regression, class-balanced | C ∈ {1e−4 … 10}; inner 5-fold CV |
| mlp | pooled | standardise → Linear(256) → GELU → Dropout 0.5 → Linear(1) | AdamW lr 1e−3, wd 1e−2, batch 64, ≤200 epochs, patience 25 |
| attn_mil | token grid | standardise → Linear(256) → GELU → gated attention pooling → Linear(1), token dropout 0.25 | AdamW lr 2e−4, wd 1e−2, batch 32, ≤100 epochs, patience 15 |

The torch heads use one fixed configuration (no grid). Early stopping uses the AUC of a
stratified 15% validation split of **their own training portion**; standardisation statistics come
from that training portion only. Class-weighted BCE.

## Protocols
1. **In-domain reference** for each dataset: stratified 5-fold outer CV, repeated 5 times
   (seeds 0–4; 2 repeats for TBX11K, n > 2000), hyperparameters by inner 5-fold CV where a head
   has a grid. Reported as the mean ± SD of AUC over repeats.
2. **Cross-domain** S → T for all six ordered pairs: fit on all of S, score every image of T once.
   The decision threshold is Youden's J on S out-of-fold scores (fold split different from the
   hyperparameter search) and is applied unchanged to T. sklearn heads: 1 run (deterministic);
   torch heads: 3 seeds.
3. **Primary repeat.** Every paired test uses repeat/seed 0, pre-specified. The other repeats are
   reported as mean ± SD and as a sensitivity analysis.
4. **Size-matched** S → T with S subsampled (stratified) to floor(4/5 · N_T), where defined
   (Shenzhen→Montgomery, TBX11K→Montgomery, TBX11K→Shenzhen).

## Endpoints
**Primary.** For each primary pair (Montgomery→Shenzhen, TBX11K→Shenzhen, Shenzhen→TBX11K,
Montgomery→TBX11K) and each of the 12 backbones: the paired difference in target AUC,
FUSION − backbone (RBF-SVM head), on identical target images. Paired DeLong test, confirmed by a
paired stratified bootstrap (2,000 resamples). **Holm correction over all 48 comparisons**
(fixed family; a comparison that cannot be computed counts as p = 1).
- A comparison is "significant" only if the Holm-adjusted DeLong p < 0.05 AND the bootstrap 95% CI
  excludes 0; if they disagree, both are reported and the result is called not robust.
- Reporting: the number of significant comparisons in each direction, broken down by backbone
  group and target. No single-sentence verdict that ignores the pattern.
- Montgomery as a target (n = 138) is descriptive only.

**Secondary** (each with its own fixed Holm family where a test is made):
1. *Source-selected deep model vs FUSION.* For each source S, the backbone × head with the highest
   in-domain AUC on S (mean over repeats; source data only) is compared with FUSION on each target
   (Holm over the 4 primary pairs). This is the only legitimate "best deep model": nothing is
   selected using target data.
2. *Transfer gap* G = AUC_in(T) − AUC_cross(S→T), per model, with paired-bootstrap CI; the
   relative gap G / (AUC_in(T) − 0.5) descriptively; the difference in G between FUSION and each
   backbone (paired bootstrap over T).
3. *Backbone contrasts* (8 contrasts × 4 primary pairs, Holm over 32).
4. *Hybrid*: (backbone + FUSION blocks, block-weighted, RBF-SVM) − backbone alone
   (12 × 4, Holm over 48).
5. *Head ablation*: AUC per backbone × head, in-domain and cross-domain. Descriptive.
6. *Mask ablation*: lung-masked − unmasked target AUC per model. Descriptive.
7. *Size-matched transfer gap*. Descriptive.
8. Sensitivity and specificity at the source-chosen threshold, with Wilson 95% CIs.
9. *Optional reference*: v1 fully fine-tuned DenseNet-121 vs its frozen counterpart. Descriptive.

The leaderboard (mean cross-domain AUC and mean rank over the six transfers) is descriptive; a
model that is best on target data is never presented as "selected".

## Shortcut control
- Dataset-identification probe within each label class (normal vs normal, abnormal vs abnormal)
  for each pair of datasets, on the common 512 × 512 grid.
- Fixed-capacity classifier: standardisation + L2 logistic regression, C = 1, class-balanced,
  never tuned. Out-of-fold balanced accuracy and AUC over 5-fold CV × 5 repeats, with a
  200-permutation label-shuffle null per representation.
- Representations: each hand-crafted family, FUSION, and the pooled features of all 12 frozen
  backbones; lung-masked and unmasked. The comparison of interest is masked vs unmasked within a
  representation; representations differ in dimension, so absolute levels are not ranked.

## Rules
- Backbones, heads, grids and torch-head configurations above are fixed at the freeze.
- No outer-test-fold or target-domain result is used to change features, preprocessing,
  backbones, heads, grids or configurations, or to choose which model is reported as best.
- Every reported number comes from a saved run directory (config, commit, environment,
  predictions, metrics) aggregated by `scripts/aggregate_benchmark.py`. Smoke runs are excluded;
  duplicate runs of one identity must be resolved explicitly.

## Compute (for planning)
Feature extraction: 12 backbones × 3 datasets × 2 masks, one forward pass per image.
Runs: about 860 (12 backbones × [4 heads × 12 protocols + 9 unmasked + 9 hybrid RBF-SVM runs],
plus the hand-crafted runs). The RBF-SVM grid on TBX11K dominates CPU time; the torch heads
dominate GPU time. `scripts/run_experiments.py --shard i/n` splits the runs across machines/GPUs.
