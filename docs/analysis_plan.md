# Analysis plan

Status: draft. It is frozen before the main experiments run; any later change is reported as
a deviation.

## Task and data
- Binary classification of frontal chest radiographs: TB-related abnormality vs normal.
- Datasets: Shenzhen and Montgomery (U.S. National Library of Medicine), and TBX11K (train and
  val splits; its test labels are not public). Images are the unit of analysis.
- Before any experiment, a perceptual-hash check for duplicate images within and across
  datasets.

## Models (fixed in advance)
- **Hand-crafted (primary):** shape context + LBP + GLCM + Gabor + HOG. Each feature block is
  standardised and scaled by 1/sqrt(block dimension), then classified with an RBF-SVM.
- **Deep (primary):** DenseNet-121 (ImageNet initialisation, full fine-tuning, fixed recipe,
  3 seeds).
- Secondary: individual feature families, other CNN/transformer backbones, and a
  CXR-pretrained backbone.

## Inputs
- **Primary: lung-masked inputs**, from one automatic segmenter applied identically to every
  dataset. It is validated by Dice against the Montgomery manual masks.
- Unmasked inputs are used only as a shortcut ablation.
- Aspect ratio preserved (pad to square, then resize to 512 × 512).

## Protocols
1. **In-domain reference** for each dataset: stratified 5-fold outer CV × 5 repeats, with
   inner 5-fold CV for hyperparameters. Reported as the mean ± SD of AUC over repeats.
2. **Cross-domain** S → T: train on all of S (hyperparameters by inner CV on S), score every
   image in T once. The decision threshold is chosen on S only (Youden's J on S out-of-fold
   scores, using a fold split different from the one used for the hyperparameter search) and
   applied unchanged to T.

## Endpoints
- **Primary:** for each target T in {Shenzhen, TBX11K}, the paired difference in out-of-domain
  AUC between the hand-crafted model and DenseNet-121 on the same target images. Tested with a
  paired DeLong test and confirmed with a paired bootstrap; Holm correction across the pairs.
- **Secondary (transfer gap):** in-domain AUC on T minus S → T AUC, with both terms measured on
  T. Also reported with the S training set subsampled to match the in-domain training size, and
  as a relative gap.
- Sensitivity and specificity at the source-chosen threshold, with Wilson 95% CIs.
- Montgomery as a target is reported descriptively only (n = 138).

## Shortcut control
- A dataset-identification probe within each label class, with images resampled to a common
  resolution, a fixed-capacity linear classifier, and balanced accuracy reported. Run on masked
  and unmasked inputs.

## Rules
- Hyperparameter grids are fixed before the freeze.
- No outer-test-fold or target-domain result is used to change features, preprocessing, or
  grids.
- Every reported number comes from a saved run directory (config, commit, environment,
  predictions, metrics).
