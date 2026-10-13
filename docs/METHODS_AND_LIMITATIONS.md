# Methods

## Study scope and labels

The analysis concerns mouse lung H&E histology from a heat and ozone co-exposure study. Patch labels are assigned by the QuPath export procedure using supplied annotations: Lesion centers lie inside included ROIs; Normal, Normal-Far and Normal-Red are sampled outside them and combined as Normal. Annotation selection, filtering and negative sampling are specified in [Patch generation](PATCH_GENERATION.md). Lesion has model class index 0 and binary metric label 1.

## Data identity and partitioning

The input audit covered 43,335 original PNG exports. A further 15,538 records referring to unavailable derived images were excluded. Exact decoded duplicates were resolved deterministically. Patch classification uses 23,309 training, 7,909 validation and 8,704 test images, from 19, 5 and 4 distinct source slides. Slide counts are not interpreted as animal counts.

Animal-level analysis uses 21 animals in seven groups of three. Three source-aware outer folds separate each held-out animal and its source aliases from the corresponding training and inner-validation data. Content hashes and canonical animal identities are used to verify separation. All model configurations use the same animal outcomes and fixed window identities.

## Model configurations

| Identifier | Configuration |
|---|---|
| v2 | Random-initialized custom ViT |
| v3 | ImageNet-initialized torchvision ViT-B/16 |
| v4 | Pretrained ViT with complete PNG inputs and consistent physical field of view |
| v5 | Frozen Phikon-v2 features with fixed logistic-regression classifiers |
| v6 | v4 with class- and source-balanced training sampling |

A ResNet-50 model provides an additional classification comparison. Initialization, training, preprocessing, precision and threshold procedures differ across configurations; these comparisons do not isolate architecture effects.

## v6 training

The ViT-B/16 backbone is initialized with torchvision ImageNet1K V1 weights, and the classifier is replaced by a two-class linear layer. All parameters are trained. Inputs are complete 224 × 224 PNG images with random horizontal flips, ColorJitter(0.1, 0.1, 0.1, 0.05), and ImageNet normalization.

AdamW uses learning rates of 1e-5 for the backbone and 1e-4 for the head, weight decay 0.05, a 20-epoch cosine schedule and batch size 32. Training runs for at most 20 epochs, with patience 6 and checkpoint selection by strict improvement in validation Lesion AUC. Computation uses CUDA FP16. The classification model uses seed 42; each of the three outer folds uses seeds 42, 43 and 44.

Within each training role, each class receives total sampling probability 0.5. Conditional on class, source probability is proportional to the square root of its patch count. Each epoch samples N patches with replacement, where N is the training-role size; cross entropy is unweighted. This changes sampling frequency rather than biological sample size, and small source-class subsets can be sampled repeatedly.

## Prediction and animal scores

Classification inference uses complete PNGs without test-time augmentation. The decision threshold is selected on validation data by the first maximum of Youden's J and then applied to test data.

Whole-slide analysis retains 132,032 fixed window identities. The physical field of view is represented by 408/409-pixel level-0 windows for v4–v6, corresponding to 224 pixels at a target resolution of 0.5 µm/pixel. The v2/v3 windows are 512 pixels. Integer placement of 409-pixel windows produces a documented −0.5 source-pixel center offset. Dense crops are resized to 224 pixels using OpenCV INTER_AREA.

Dense prediction averages softmax probabilities from the original, horizontal, vertical and combined flips. For v2/v3/v4/v6, each animal's score is the mean probability across its windows, followed by an equal average across the three seeds. No outcome fitting, ROI filtering or temperature calibration is applied to this score. The v5 model uses one deterministic classifier per outer fold, with per-image L2-normalized CLS features and fixed C = 1 balanced logistic regression; it is not a three-seed ensemble.

## Statistical analysis

Area and IL6 are analyzed on their recorded scales using Pearson and Spearman correlations, two-sided p-values, and 10,000 paired-animal bootstrap draws (seed 20260920). Holm adjustment is applied to the four primary tests within each model configuration. A fixed 18-animal sensitivity analysis excludes the `30-30-120-N` group; the primary analysis retains all 21 animals. Additional diagnostics comprise 112 leave-one-animal/group analyses and descriptive within-group-centered correlations.

Classification AUC intervals use 5,000 patch-level and 10,000 source-cluster bootstrap draws (seed 20260919). Cluster resampling is based on the four test sources. All intervals condition on the fitted models and do not include training uncertainty.

## Interpretation and limitations

The model configurations were evaluated on the same test sources, and v6 was selected for presentation after their results were examined. These are exploratory comparisons rather than independent external validation. Holm adjustment addresses the four tests within each configuration, not selection across configurations.

The classification test set contains four source slides and the association cohort contains 21 animals. Patches, seeds, image flips and annotation versions are not independent biological replicates. For v6, within-group-centered correlations are 0.054569 for Area and 0.173181 for IL6, providing little evidence of a strong within-group relationship despite the pooled associations.

Spatial evaluation uses 24 nonexhaustive references across 21 animals, containing 65 retained objects. Three animals have two reference versions, analyzed separately; one animal has no ROI reference. Unannotated tissue has no assigned Normal label. v6 window coverage ranges from approximately 95.02% to 100%, while conditional annotated-center AUC ranges from 0.4609 to 0.9847. Coverage quantifies evaluated extent, not recognition accuracy. Probability fields are window-based classification outputs, not pixel segmentation or calibrated lesion-area estimates.

Dense-slide provenance, tissue-selection scale and thumbnail resampling differ across reconstruction stages. A 3.84% coordinate-union difference is descriptive and does not establish the cause of changes in correlation. Area is recorded as mm², but its calibration has not been independently verified; IL6 units and tissue pairing require confirmation from experimental records.

Phikon-v2 provides external pretrained features for v5. The separate public mouse infection image archive that was assessed during data exploration was not used for training, validation or inference in any reported configuration.
