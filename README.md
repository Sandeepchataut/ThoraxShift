# tb-cxr-domain-shift

## Summary

Automated tuberculosis (TB) screening from chest X-rays can perform well on the dataset it was
built on, yet lose accuracy when it is used at a new site with different scanners, image
resolutions and patient populations. This project measures how much that loss happens for two
kinds of models:

- **Hand-crafted models:** thoracic edge-map shape descriptors (log-polar shape context) and
  texture descriptors (LBP, GLCM, Gabor, HOG), extracted inside automatically segmented lung
  regions and classified with an RBF support vector machine.
- **Deep models:** convolutional and transformer networks trained on the same images.

Each model is trained on one public chest X-ray collection and tested on the others, without
any tuning on the test collection. The comparison uses ROC AUC with 95% confidence intervals,
paired DeLong tests, and sensitivity and specificity at a decision threshold fixed on the
training collection. A dataset-identification probe checks whether a model is recognising the
source dataset rather than the disease. The full protocol is in
[docs/analysis_plan.md](docs/analysis_plan.md).

## Research flow

```mermaid
flowchart TD
    A[Public chest X-ray datasets<br/>Shenzhen · Montgomery · TBX11K] --> B[Acquisition audit<br/>and duplicate check]
    B --> C[Preprocessing<br/>normalise · pad to square · resize 512]
    C --> D[Automatic lung segmentation<br/>same model for every dataset]
    D --> E1[Hand-crafted features<br/>shape context · LBP · GLCM · Gabor · HOG]
    D --> E2[Deep models<br/>CNN · transformer]
    E1 --> F1[RBF-SVM]
    E2 --> F2[Fine-tuned classifier]
    F1 --> G{Evaluation}
    F2 --> G
    G --> H1[In-domain<br/>nested cross-validation]
    G --> H2[Cross-domain<br/>train on one dataset, test on another]
    G --> H3[Shortcut control<br/>dataset-identification probe]
    H1 --> I[Statistics<br/>AUC with 95% CI · paired DeLong · bootstrap · Holm]
    H2 --> I
    H3 --> I
    I --> J[Robustness comparison<br/>hand-crafted vs deep]
```
