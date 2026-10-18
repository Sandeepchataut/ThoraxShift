# ThoraxShift: summary of the experiment results

This document gives the results of the full run that completed on 2026-10-11.
The text uses ASD-STE100 Simplified Technical English.

## 1. Purpose of the comparison

The study compares three types of image features for tuberculosis (TB) screening on chest X-rays:

- Hand-crafted features.
- Features from convolutional neural networks (CNNs).
- Features from newer architectures: transformers and self-supervised models.

Each model learns and is tested on the same dataset (in-domain).
This summary gives the in-domain results first.
Section 9 gives the cross-dataset results as secondary data.

### 1.1 Models

| Group (code name) | Models |
|---|---|
| Hand-crafted | Shape context (SC), LBP, GLCM, Gabor, HOG, and FUSION (all five together) |
| ImageNet CNN (`cnn_in1k`) | ResNet-18, ResNet-50, DenseNet-121, EfficientNet-B0, ConvNeXt-T |
| ImageNet transformer (`vit_in1k`) | Swin-T, ViT-S, ViT-B |
| Self-supervised, natural images (`ssl_natural`) | DINOv2 ViT-S/14, DINOv2 ViT-B/14 |
| Chest X-ray pretrained (`cxr`) | DenseNet-121 (TorchXRayVision), RAD-DINO ViT-B/14 |

All deep backbones are frozen. Only a classifier head learns from the data.
There are four heads: RBF-SVM, logistic regression, MLP, and attention pooling.
The hand-crafted models use an RBF-SVM. This is the same classifier and the same search grid as the deep RBF-SVM head.

### 1.2 Datasets

| Dataset | Images | TB | Normal | Note |
|---|---|---|---|---|
| Shenzhen | 662 | 336 | 326 | NLM set |
| Montgomery | 138 | 58 | 80 | NLM set. Small set, so the confidence intervals are wide. |
| TBX11K | 4600 | 800 | 3800 | Train and validation splits only. TB = active and latent TB. Healthy = normal. Sick but not TB = removed. |

### 1.3 Method of evaluation

- The inputs are lung-masked unless this document gives a different mask.
- In-domain evaluation uses stratified 5-fold cross-validation. Shenzhen and Montgomery have 5 repeats. TBX11K has 2 repeats.
- The metric is the area under the ROC curve (AUC).
- Tables with "mean ± SD" give the mean and standard deviation over the repeats.
- Paired tests use repeat 0. They use the out-of-fold scores of the same images.
- The paired test is a DeLong test and a paired stratified bootstrap with 2000 resamples. The tests use Holm correction.
- In this document, "significant" means two conditions: the Holm-corrected DeLong p is less than 0.05, and the bootstrap 95% CI does not include 0.

## 2. Status of the run

All jobs completed. No job stopped because of an error.

| Job | Result |
|---|---|
| `prep.sh` (job 1300821) | Data download, image preparation, lung segmentation, audit, and weight download completed. |
| `handcrafted.sh` (job 1300920) | 72 of 72 hand-crafted jobs completed. The final aggregation failed because the deep runs did not exist at that time. This is the expected result. |
| `deep.sh` (job 1300921) | 811 of 811 jobs completed in 20.8 h. The final aggregation and the figures completed. |

- The final aggregation uses 858 runs: `outputs/runs/20261011T134724Z_aggregate_benchmark/`.
- The figures are in `outputs/figures/`.
- Lung segmentation on Montgomery has a mean Dice of 0.867 against the manual masks. The minimum Dice is 0.794. No dataset has an empty mask.
- The two experiment jobs calculated the hand-crafted feature caches at the same time. All runs that record the cache hash show one hash for each dataset. Thus the cache data is consistent.
- The aggregation records `git_dirty: true` at commit `3440d14`. Thus the code had changes that were not committed when the run started.

## 3. Main result: hand-crafted FUSION against frozen deep features

In-domain, no frozen deep backbone is significantly better than hand-crafted FUSION.

There are 144 paired comparisons: 12 backbones, 4 heads, and 3 datasets.

| Head | Deep model significantly better | FUSION significantly better | No significant difference |
|---|---|---|---|
| RBF-SVM | 0 | 18 | 18 |
| Logistic regression | 0 | 16 | 20 |
| MLP | 0 | 21 | 15 |
| Attention pooling | 0 | 17 | 19 |
| **Total** | **0** | **72** | **72** |

The Holm family is 36 comparisons for each head.

### 3.1 FUSION against each backbone, RBF-SVM head (same classifier)

The values are AUC on repeat 0. "Diff" is deep minus FUSION. A negative diff shows that FUSION is better.

| Backbone | Shenzhen (FUSION 0.932) | Montgomery (FUSION 0.827) | TBX11K (FUSION 0.998) |
|---|---|---|---|
| ResNet-18 | 0.900 (−0.032)* | 0.759 (−0.068) | 0.978 (−0.020)* |
| ResNet-50 | 0.909 (−0.023) | 0.804 (−0.023) | 0.986 (−0.012)* |
| DenseNet-121 | 0.908 (−0.025) | 0.837 (+0.010) | 0.992 (−0.006)* |
| EfficientNet-B0 | 0.902 (−0.030)* | 0.788 (−0.038) | 0.990 (−0.008)* |
| ConvNeXt-T | 0.905 (−0.027) | 0.623 (−0.203)* | 0.992 (−0.006)* |
| Swin-T | 0.889 (−0.043)* | 0.731 (−0.095) | 0.989 (−0.009)* |
| ViT-S | 0.884 (−0.048)* | 0.860 (+0.033) | 0.985 (−0.013)* |
| ViT-B | 0.898 (−0.034)* | 0.647 (−0.180)* | 0.992 (−0.006)* |
| DINOv2 ViT-S/14 | 0.916 (−0.016) | 0.857 (+0.030) | 0.995 (−0.003) |
| DINOv2 ViT-B/14 | 0.897 (−0.035)* | 0.725 (−0.102) | 0.996 (−0.002) |
| DenseNet-121 (CXR) | 0.897 (−0.035)* | 0.839 (+0.012) | 0.994 (−0.004)* |
| RAD-DINO ViT-B/14 | 0.953 (+0.020) | 0.919 (+0.092) | 0.998 (−0.000) |

\* = significant.

- RAD-DINO is the only backbone with a higher AUC than FUSION on Shenzhen and Montgomery. These differences are not significant after Holm correction.
- On Shenzhen and TBX11K, FUSION is significantly better than most ImageNet CNNs and ImageNet transformers.
- On Montgomery, most differences are not significant because the dataset is small.

### 3.2 Best model on each dataset (mean over repeats)

| Dataset | Best model | AUC | Rank of FUSION |
|---|---|---|---|
| Shenzhen | RAD-DINO / attention pooling | 0.964 ± 0.004 | 13 of 66 |
| Montgomery | RAD-DINO / attention pooling | 0.920 ± 0.012 | see note |
| TBX11K | RAD-DINO / logistic regression | 0.998 ± 0.000 | at the ceiling |

Note: on Montgomery, FUSION has 0.831 ± 0.005. On TBX11K, FUSION has 0.998 ± 0.000. Many models are at 0.99 or more on TBX11K, so the ranks on TBX11K have no meaning.

## 4. CNNs against newer architectures

### 4.1 Mean in-domain AUC for each group

The values are the mean and the maximum over all backbones and heads in the group.

| Group | Shenzhen mean / max | Montgomery mean / max | TBX11K mean / max |
|---|---|---|---|
| ImageNet CNN | 0.899 / 0.915 | 0.776 / 0.833 | 0.984 / 0.993 |
| ImageNet transformer | 0.885 / 0.901 | 0.743 / 0.863 | 0.986 / 0.991 |
| Self-supervised, natural | 0.914 / 0.947 | 0.792 / 0.849 | 0.994 / 0.997 |
| Chest X-ray pretrained | 0.922 / 0.964 | 0.844 / 0.920 | 0.991 / 0.998 |
| Hand-crafted FUSION | 0.929 | 0.831 | 0.998 |

- On Shenzhen and Montgomery, the ImageNet transformers have the lowest mean AUC. They are not better than the ImageNet CNNs.
- The self-supervised and chest X-ray groups have a higher mean AUC than the ImageNet groups.
- The high group mean of the chest X-ray group comes from RAD-DINO. The chest X-ray DenseNet-121 is not better than the ImageNet DenseNet-121 (see 4.2).
- The mean of each deep group is lower than FUSION on Shenzhen and TBX11K.

### 4.2 Pre-specified paired contrasts (in-domain)

There are 8 pre-specified contrasts (model a against model b). Each contrast is tested on 3 datasets with 2 heads. The heads are RBF-SVM and attention pooling. The Holm family is 24 comparisons for each head.

| Contrast | Question | Significant results | Conclusion |
|---|---|---|---|
| ResNet-50 vs ViT-S | CNN against plain transformer, about 25M parameters | 1 of 6: ViT-S is better on TBX11K, attention pooling (diff −0.013) | No consistent difference |
| ConvNeXt-T vs Swin-T | Modern CNN against hierarchical transformer, about 28M parameters | 0 of 6 | No difference |
| ViT-B vs DINOv2-B | Supervised against self-supervised pretraining | 1 of 6: DINOv2-B is better on Shenzhen, attention pooling (diff −0.046) | Small advantage for self-supervised pretraining |
| DINOv2-B vs RAD-DINO | Natural-image against chest X-ray self-supervised pretraining | 3 of 6: RAD-DINO is better on Shenzhen, SVM (−0.055); Montgomery, SVM (−0.194); Montgomery, attention pooling (−0.140) | Chest X-ray self-supervised pretraining helps |
| DenseNet-121 vs DenseNet-121 (CXR) | ImageNet against supervised chest X-ray pretraining | 1 of 6: the ImageNet model is better on TBX11K, attention pooling (diff +0.014) | Supervised chest X-ray pretraining does not help |
| ResNet-18 vs ResNet-50 | Model size | 0 of 6 | Size has no effect |
| ViT-S vs ViT-B | Model size | 1 of 6: ViT-S is better on Montgomery, SVM (diff +0.213) | The larger model is not better |
| DINOv2-S vs DINOv2-B | Model size | 1 of 6: DINOv2-S is better on Montgomery, SVM (diff +0.132) | The larger model is not better |

Thus:

1. The architecture type (CNN or transformer) has no consistent effect on in-domain AUC.
2. A larger model does not give a higher AUC. On Montgomery, two smaller models are significantly better than the larger models in the same family.
3. The type of pretraining has a larger effect than the architecture. Self-supervised pretraining on chest X-rays (RAD-DINO) gives the best deep features.

## 5. Classifier heads

The values are the mean in-domain AUC over the 3 datasets and all backbones in the group.

| Group | Attention pooling | Logistic regression | MLP | RBF-SVM |
|---|---|---|---|---|
| ImageNet CNN | 0.881 | 0.889 | 0.887 | 0.887 |
| ImageNet transformer | 0.866 | 0.879 | 0.867 | 0.874 |
| Self-supervised, natural | 0.902 | 0.905 | 0.897 | 0.898 |
| Chest X-ray pretrained | 0.917 | 0.922 | 0.925 | 0.912 |

- The differences between the heads are small. They are less than 0.015 in each group.
- Attention pooling gives the best single results with RAD-DINO: 0.964 on Shenzhen and 0.920 on Montgomery.
- The choice of head is less important than the choice of backbone.

## 6. Individual hand-crafted features

| Feature | Shenzhen | Montgomery | TBX11K |
|---|---|---|---|
| Gabor | 0.915 | 0.801 | 0.991 |
| LBP | 0.889 | 0.760 | 0.989 |
| HOG | 0.878 | 0.792 | 0.980 |
| GLCM | 0.880 | 0.750 | 0.981 |
| Shape context | 0.748 | 0.638 | 0.932 |
| **FUSION (all five)** | **0.929** | **0.831** | **0.998** |

- FUSION is better than each single feature on all three datasets.
- Gabor is the best single feature. Shape context is the weakest feature.
- On Shenzhen, Gabor alone (0.915) is better than the mean of each deep group, except the chest X-ray group.

## 7. Hybrid models (backbone + FUSION)

A hybrid model adds the five hand-crafted feature blocks to the deep features. It uses an RBF-SVM.

| Dataset | Hybrid better than its backbone alone | Mean gain over the backbone | Mean difference to FUSION alone |
|---|---|---|---|
| Shenzhen | 11 of 12 | +0.025 | +0.001 |
| Montgomery | 10 of 12 | +0.059 | +0.000 |
| TBX11K | 12 of 12 | +0.008 | +0.001 |

- In-domain, the hybrid model is better than the backbone alone.
- The hybrid model is not better than FUSION alone.
- Cause: the code gives each feature block the same weight in the RBF kernel. Thus the deep block is 1 of 6 blocks. The five hand-crafted blocks control the kernel distance.
- The hybrid results thus show the hand-crafted features. They do not show a combination with equal contributions.

## 8. Lung mask against full image (in-domain)

The values are AUC with the lung mask minus AUC without a mask (RBF-SVM, mean over repeats).

- Shenzhen: the difference is between −0.016 and +0.013 for all models.
- TBX11K: the difference is between −0.017 and −0.001 for all models.
- Montgomery: the difference is between −0.167 and +0.060. This dataset is small, so these values are not stable.

The lung mask has a very small effect on in-domain AUC.

## 9. Cross-dataset results (secondary)

These results are not the purpose of the study. They are here because the pipeline calculated them.

- The pre-registered primary endpoint has 48 comparisons. These are 4 transfer pairs and 12 backbones, FUSION against deep, RBF-SVM. Of these, 28 are significant: 23 show that the deep model is better, and 5 show that FUSION is better.
- FUSION transfers badly. The mean cross-dataset AUC is 0.628. On Montgomery → TBX11K, the AUC is 0.422. On Shenzhen → Montgomery, the AUC is 0.482. Both values are below chance.
- LBP and GLCM cause most of this problem. For example, on Montgomery → TBX11K, LBP has 0.130 and GLCM has 0.220. HOG alone transfers well, with a mean of 0.778.
- RAD-DINO with attention pooling has the best cross-dataset result: a mean of 0.935 over the six pairs.
- A decision threshold set on the source dataset does not transfer. FUSION classifies all images as TB on 3 of the 6 pairs. 52% of the deep models have a sensitivity or a specificity below 0.5 on the target dataset.

## 10. Cautions

Read these points before you use the results.

1. **TBX11K is at the ceiling.** FUSION has 0.998. Gabor features alone have 0.991. Many deep models have 0.99 or more. TBX11K thus cannot show differences between the models.
2. **A probe can identify the dataset of each image.** A dataset-identification probe has a balanced accuracy of 0.95 or more for most features. It has this accuracy also with lung masks, and also within one class (TB only or normal only). All 216 probe results are significant (permutation p < 0.005). The datasets thus have strong differences that are not related to TB. These differences can also exist between the TB and healthy images in one dataset. The very high TBX11K AUC can be an example of this.
3. **Montgomery is small (n = 138).** The confidence intervals are wide. Some of the large differences on Montgomery (for example ViT-S against ViT-B, +0.213) are not reliable.
4. **The input resolution is not the same for all models.** The DINOv2 and RAD-DINO models use 518 px. The other models use 224 px. Some of the advantage of these models can come from the resolution.
5. **No fine-tuned reference exists.** The fine-tuning stage did not run. `tables/finetune_reference.csv` is empty. The results thus compare frozen features only.
6. **The folder has no model checkpoint files.** A search for `.pt`, `.pth` and `.ckpt` files found none.
7. **The in-domain paired tests are not in the aggregation output.** The aggregation script tests only cross-dataset comparisons. The tests in sections 3 and 4.2 were calculated for this summary. The method is the same as the aggregation: `tbshift.eval.metrics.auc_difference` on the run directories listed in `inputs.json` of the aggregation run. The repeat-0 folds are the same for the hand-crafted runs and the deep runs.
8. **TBX11K labels.** The "sick but not TB" class is not in the data. A different label rule can change the TBX11K results.

## 11. Conclusions

1. For in-domain TB classification, hand-crafted FUSION features are equal to or better than frozen deep features. No frozen backbone is significantly better than FUSION in 144 paired comparisons.
2. Newer architectures are not better than CNNs. On Shenzhen and Montgomery, the ImageNet transformers (Swin-T, ViT-S, ViT-B) have the lowest mean AUC of all groups. ConvNeXt-T and Swin-T have no difference.
3. A larger model is not better. ResNet-50 is not better than ResNet-18. ViT-B and DINOv2-B are not better than their small versions.
4. The pretraining data is the most important factor for deep features. RAD-DINO (self-supervised on chest X-rays) is the best backbone in 11 of 12 combinations of dataset and head. The exception is Montgomery with the RBF-SVM head (0.834 ± 0.107). But RAD-DINO is not significantly better than FUSION in-domain.
5. Supervised pretraining on chest X-rays (TorchXRayVision DenseNet-121) gives no advantage over ImageNet pretraining.
6. The head type has a small effect. The backbone and its pretraining have a larger effect.

## 12. Source files

| Content | Location |
|---|---|
| Aggregation summary | `outputs/runs/20261011T134724Z_aggregate_benchmark/summary.md` |
| All tables (CSV and LaTeX) | `outputs/runs/20261011T134724Z_aggregate_benchmark/tables/` |
| Shortcut probe | `outputs/runs/20261011T122249Z_shortcut_probe_shenzhen-montgomery-tbx11k/` |
| Segmentation Dice | `outputs/runs/20261010T150048Z_segmentation_dice_montgomery/` |
| Figures | `outputs/figures/` |
| Job logs | `prep_1300821.out`, `hc_1300920.out`, `dl_1300921.out`, `outputs/logs/` |
