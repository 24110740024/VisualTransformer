# Methods and interpretation

## Scope

Computational pathology for mouse lung H&E in a study of heat/ozone co-exposure and spatial lipid remodeling. This repository covers the pathology analysis, not all lipid analyses or the entire manuscript revision. Existing QuPath Lesion/Normal categories are the fixed references. Normal-Far and Normal-Red map to Normal.

## Identity and partitioning

43,335 original PNG files and decoded pixels were audited. 15,538 unavailable historical derived records were excluded, and exact decoded duplicates were handled deterministically. A6 uses 23,309 train / 7,909 validation / 8,704 test patches from 19 / 5 / 4 distinct source WSIs. Source counts are not automatically animal counts. A5 uses 21 animals, seven groups of three, with source-aware three-fold cross-fitting. Each animal and source aliases are excluded from the corresponding training and inner-validation roles. A source has a stable content identity; renaming a file does not make it independent.

## v6 training

ImageNet1K V1 ViT-B/16; a new two-class linear head and all parameters trainable. Full 224x224 PNGs, no random/center crop; random horizontal flip and ColorJitter(0.1,0.1,0.1,0.05), then ImageNet normalization. AdamW: backbone1e-5, head1e-4, weight decay0.05; cosine scheduler20; batch32; at most20 epochs; patience6; strict validation Lesion-AUC improvement. CUDA FP16. Sampling is fitted only to the current training role: each class has total probability0.5; within a class, source probability is proportional to the square root of its patch count. Draw N patches with replacement each epoch, use unweighted cross entropy. Small cells can still repeat; this is not equal-source sampling.

## Scoring and geometry

132,032 outcome-independent original center identities are retained. v2/v3 use 512-pixel windows; v4/v5/v6 use source-calibrated 408/409 level0 windows corresponding to 224px at the target0.5 micrometers/pixel. The 409-pixel window has a recorded -0.5 source-pixel center offset from integer placement. Dense crops use OpenCV INTER_AREA to224; A6 PNGs are not recropped. A5 scores are all-window mean Lesion probabilities per animal/seed, then equal means of seeds42/43/44 for v2/v3/v4/v6. v5 uses one deterministic outer-fold head per animal, not three seeds. No ROI or outcome-driven threshold/filter/calibration enters these means.

## Five variants and interpretation

v2: random custom ViT. v3: torchvision ImageNet ViT. v4: full-PNG/scale-consistent ViT. v5: frozen Phikon-v2 CLS1024, per-image L2 normalization and fixed C1 balanced logistic heads, trained only on each role's train. v6: v4 plus the above sampling. ResNet50 is the matching-input classification comparison; initialization, training, preprocessing, precision and threshold recipes differ, so the comparison does not isolate architecture effects. v5 is a feature-model/linear-head comparison, not end-to-end fine-tuning.

v6 is the suggested main presentation **after observing the results**. It was not the sole prospectively specified primary model. The same test sources were seen during additional development. All five variants and ResNet are reported; numerical ordering is not proof of superiority or external validation. Holm correction is within the four primary tests of each variant, not correction for selecting a method across variants.

## Biological and spatial limits

Area and IL6 correlations use raw outcomes, Pearson and Spearman, two-sided p-values, and paired-animal percentile bootstrap (10,000 draws, seed20260920). Four primary tests are additionally Holm-adjusted. v6 within-group-centered Pearson r is only0.054569 (Area) and0.173181 (IL6): do not infer a strong within-group relationship. Bootstrap conditions on fixed fitted scores.

There are only four classification test sources and21 association animals. Patches, seeds, flips and repeated ROI versions are not biological replicates. Three animals have two stored ROI versions; retain them separately, never union or choose by performance. One animal has no ROI; absence of annotation is not a Normal label. The24 references contain65 retained objects; saved references are nonexhaustive. v6 window coverage95.02–100% is not accuracy; conditional annotated-center AUC ranges approximately0.4609–0.9847. Weak cases and high-probability regions outside stored contours are retained. Do not report whole-slide Dice/IoU, a pixel segmentation claim or a calibrated true lesion area from these fields.

Old dense-source/version chains, 512-vs256 tissue decisions, BOX thumbnails and case-level tile-count changes remain reconstruction limitations. A3.84% coordinate-union difference is a diagnostic, not proof of the reason for a historical correlation change.

The Area source field suggests mm² but its calibration is not independently confirmed; IL6 units and tissue pairing require author records. The authors handle manuscript/SI edits, including retiring the old S3 LASSO/CV panel and synchronizing TableS2. No manuscript is automatically edited by this release.

## External resources

Phikon-v2 is externally pretrained and its own license terms apply; see [license note](PHIKON_LICENSE_NOTE.md). Its weights/features are not redistributed here and are not relicensed as Apache2.0.

The downloaded public mouse infection H&E dataset (Zenodo17486453) was technically inspected but **never used in any v2–v6 training, validation or inference**. It has no directly compatible patch-wise Lesion/Normal ground truth for this experiment. It does not add animals to n21. External data might support future representation learning with a separately specified study, but do not claim that happened here.
