# Analysis plan

Status: **draft**. It will be frozen (git tag `plan-frozen`, pushed to the public repository so
the freeze date is independently timestamped) before the main experiments run. Any later
change is reported as a deviation.

## Task and data
- Binary classification of frontal chest radiographs: TB-related abnormality vs normal.
- Datasets: Shenzhen and Montgomery (U.S. National Library of Medicine), and TBX11K (train and
  val splits; its test labels are not public). Images are the unit of analysis.
- Before any experiment, a perceptual-hash check for duplicate images within and across
  datasets.

## Preprocessing and inputs (identical for every dataset and model family)
- Per-image 1st–99th percentile intensity window, pad to square, resize to 512 × 512, CLAHE.
  Cached once (`scripts/prepare_images.py`) and read by every model family.
- Automatic lung segmentation with one pretrained model for every dataset (its weights hash
  is recorded), validated by Dice against the Montgomery manual masks. Manual masks are never
  model inputs.
- **Primary: lung-masked inputs.** Shape context uses edges inside the lung mask dilated by
  15 px ("thoracic region"). Texture descriptors are computed inside the lung mask. Deep
  models see the image with everything outside the 15 px-dilated lung mask set to zero.
- Unmasked inputs are used only as a shortcut ablation.

## Models (fixed in advance)
- **Hand-crafted (primary): FUSION** = shape context + LBP + GLCM + Gabor + HOG. Each block
  is standardised and scaled by 1/sqrt(block dimension), then fed to an RBF-SVM
  (C ∈ {0.01, 0.1, 1, 10, 100, 1000}; γ ∈ {0.1, 0.3, 1, 3, 10} / number of blocks).
- **Deep (primary): DenseNet-121**, ImageNet initialisation, full fine-tuning, one fixed recipe:
  input 384 × 384, AdamW (lr 1e-4, weight decay 1e-4), cosine schedule, at most 30 epochs,
  batch 16, class-weighted BCE, early stopping (patience 5) on the AUC of a stratified 15%
  validation split of the training data, augmentation of ±5° rotation, ±5% translation,
  0.9–1.1 scale and ±0.1 brightness/contrast, no horizontal flip. Seeds 0, 1, 2.
- Secondary: individual feature families; ResNet-50 and EfficientNet-B0 with the same recipe.

## Protocols
1. **In-domain reference** for each dataset.
   - Hand-crafted: stratified 5-fold outer CV × 5 repeats (seeds 0–4), inner 5-fold CV for
     hyperparameters.
   - Deep: stratified 5-fold outer CV for each of the 3 seeds; early stopping uses a
     validation split of each training fold only.
   - Reported as the mean ± SD of AUC over repeats or seeds.
2. **Cross-domain** S → T: train on all of S and score every image in T once. The decision
   threshold is chosen on S only (Youden's J) and applied unchanged to T.
   - Hand-crafted: the threshold uses S out-of-fold scores from a fold split different from the
     one used for the hyperparameter search.
   - Deep: the threshold uses the S validation split (never used for gradient updates).
3. **Primary repeat.** Every paired test uses repeat/seed 0, pre-specified. The other
   repeats/seeds are reported as mean ± SD and as a sensitivity analysis.

## Endpoints
- **Primary:** for each target T ∈ {Shenzhen, TBX11K} and each source S ≠ T, the paired
  difference in out-of-domain AUC (FUSION − DenseNet-121) on the same target images. Tested with
  a paired DeLong test, confirmed with a paired stratified bootstrap (2,000 resamples), with Holm
  correction across all primary (S, T) pairs.
- **Secondary (transfer gap):** G = AUC_in(T) − AUC_cross(S → T), both on the same T images
  (in-domain out-of-fold scores vs cross-domain scores), with a paired-bootstrap CI. Also
  reported:
  - the relative gap G / (AUC_in(T) − 0.5);
  - the difference in G between the two model families (paired bootstrap over T);
  - G with S subsampled (stratified) to the in-domain training size of T.
- Sensitivity and specificity at the source-chosen threshold, with Wilson 95% CIs.
- Montgomery as a target is reported descriptively only (n = 138); it is not part of the
  Holm family.

## Shortcut control
- A dataset-identification probe within each label class (normal vs normal, abnormal vs
  abnormal) for each pair of datasets. All images are on the common 512 × 512 grid.
- Fixed-capacity classifier: standardisation + L2 logistic regression with C = 1
  (class-balanced, never tuned). Out-of-fold balanced accuracy and AUC over 5-fold CV × 5
  repeats.
- Representations: each hand-crafted family, FUSION, and frozen ImageNet DenseNet-121
  features (no training). Run on lung-masked and unmasked inputs.

## Rules
- Grids and the deep recipe above are fixed at the freeze.
- No outer-test-fold or target-domain result is used to change features, preprocessing,
  grids, or the recipe.
- Every reported number comes from a saved run directory (config, commit, environment,
  predictions, metrics). Runs with a modified recipe (smoke tests) are excluded by the
  aggregation script. When several runs share an identity, one must be selected explicitly.
