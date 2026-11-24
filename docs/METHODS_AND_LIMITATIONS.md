# Methods and Limitations

## Study Scope and Labels

This workflow analyzes mouse lung H&E whole-slide images (WSIs) from a heat and ozone co-exposure study. Patch labels are exported via QuPath from collaborator-supplied histopathological annotations: `Lesion` patches have centers inside included ROIs, whereas `Normal`, `Normal-Far`, and `Normal-Red` patches are sampled outside ROIs and merged into the `Normal` class. Full annotation filtering and negative sampling parameters are documented in [Patch Generation](PATCH_GENERATION.md). In classification outputs, `Lesion` corresponds to model class index `0` and binary metric label `1`.

## Data Identity and Partitioning

The input audit reviewed 43,335 original PNG exports (excluding 15,538 records referencing unavailable derived files) and resolved exact decoded duplicates deterministically. Source-aware partitioning yields 23,309 training, 7,909 validation, and 8,704 test patches drawn from 19, 5, and 4 distinct source slides, respectively.

Animal-level evaluation covers 21 animals across 7 experimental groups (3 animals per group). Three source-aware outer folds ensure that every held-out animal and its source aliases are strictly excluded from the corresponding training and inner-validation splits. Content hashes and canonical animal IDs are checked programmatically to prevent leakage. All compared configurations use identical animal outcomes and fixed window coordinates.

## Model Configurations

| Identifier | Configuration |
|---|---|
| `v2` | Random-initialized custom ViT |
| `v3` | ImageNet-pretrained `torchvision` ViT-B/16 |
| `v4` | Pretrained ViT-B/16 with full PNG inputs and consistent physical field of view |
| `v5` | Frozen Phikon-v2 features with fixed L2-regularized logistic regression |
| `v6` | `v4` with class- and source-balanced training sampling |

A ResNet-50 classifier is included as a convolutional baseline. Because initialization, preprocessing, and sampling vary across configurations, cross-configuration differences reflect the combined pipeline design rather than backbone architecture alone.

## `v6` Training Protocol

The ViT-B/16 backbone is initialized from `torchvision` ImageNet1K V1 weights with a two-class linear classification head, and all parameters are fine-tuned end to end. Inputs are full 224 × 224 PNG patches augmented with random horizontal flips, `ColorJitter(0.1, 0.1, 0.1, 0.05)`, and ImageNet normalization.

Optimization uses AdamW (backbone LR `1e-5`, head LR `1e-4`, weight decay `0.05`) with a 20-epoch cosine schedule, batch size 32, early stopping patience 6, and checkpoint selection by validation Lesion AUC under CUDA FP16. Classification training uses seed 42; each of the three animal-scoring outer folds trains three independent models with seeds 42, 43, and 44.

During training, both classes receive equal sampling probability (`0.5` each). Conditional on class, source slide sampling probability is proportional to the square root of its patch count. Each epoch draws `N` patches with replacement (`N` = training split size) using unweighted cross-entropy loss.

## Inference and Animal-Level Scoring

Patch classification uses original 224 × 224 PNGs without test-time augmentation, with the operating threshold chosen on the validation split via Youden's J and applied unchanged to the test split.

Whole-slide analysis evaluates 132,032 fixed window locations. For `v4`–`v6`, level-0 windows of 408/409 pixels match a 224-pixel input at a target resolution of 0.5 µm/pixel (`v2`/`v3` use 512-pixel windows), resized via OpenCV `INTER_AREA`. Dense inference averages softmax probabilities across four flip orientations (original, horizontal, vertical, and combined). For `v2`/`v3`/`v4`/`v6`, each animal's score is the mean lesion probability over its tissue windows averaged equally across the three seeds, without post-hoc outcome fitting or temperature scaling. For `v5`, each outer fold uses a single deterministic balanced logistic regression classifier (`C = 1`) on L2-normalized CLS embeddings.

## Statistical Analysis

Associations between animal-level scores and recorded outcomes (`Area` and `IL6`) are evaluated using Pearson and Spearman correlations, two-sided p-values, and 10,000 paired-animal bootstrap resamples (seed `20260920`). Holm correction is applied across the four primary tests within each model configuration. Sensitivity checks include an 18-animal subset excluding the `30-30-120-N` group, 112 leave-one-animal/group diagnostics, and descriptive within-group-centered correlations.

Patch classification AUC intervals are estimated using 5,000 patch-level and 10,000 source-cluster bootstrap resamples (seed `20260919`, conditioned on the fitted models).

## Limitations and Interpretation

- **Cohort size and exploratory selection**: The patch test set comprises 4 held-out source slides and the association cohort comprises 21 animals. Because all configurations were evaluated on the same test sources and `v6` was chosen for presentation following evaluation, cross-model comparisons should be interpreted as exploratory rather than independent external validation. Holm adjustment accounts for the four tests within a configuration rather than model selection across configurations.
- **Between-group vs. within-group signal**: Pooled animal-level correlations primarily capture exposure-group differences. For `v6`, descriptive within-group-centered correlations are modest (`0.055` for Area and `0.173` for IL-6), offering limited power to resolve subtle within-group variation at `n = 3` per group.
- **Spatial interpretability**: Spatial evaluation relies on 24 non-exhaustive ROI references across 21 animals (65 retained objects; 3 animals have two reference versions evaluated separately, and 1 animal has no ROI reference). Unannotated tissue does not carry a confirmed `Normal` label. While `v6` window coverage spans 95.0%–100%, conditional annotated-center AUC spans 0.461–0.985. Probability heatmaps reflect window-level classification scores rather than calibrated pixel segmentation or direct lesion-area measurements.
- **External data**: Phikon-v2 provides external pretrained representations for `v5`. No other external mouse lung dataset was used in training, validation, or inference for any reported configuration. Lesion area and IL-6 measurements are analyzed on the study's recorded scales.
