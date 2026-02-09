# Pathological Image Preprocessing and Patch Generation Pipeline
This study uses mouse lung tissue H&E-stained whole-slide images (WSIs) as input. The overall workflow is divided into two types of patches for distinct purposes:
- **Training Patches**: Exported from QuPath annotations (224×224), used for model training and validation.
- **Inference Patches**: Full-slide coverage patches (512×512, stride=256), used for dense inference and visualization of entire WSIs.

## 1. QuPath Annotation & Training Patch Export (224×224)
### Core Workflow
1. Annotate lesion regions on WSIs in QuPath (v0.4+).
2. Run a custom Groovy export script (e.g., `qupath_tiles_v2.groovy`) to crop tissue regions into **224×224 pixel patches** at ~0.5 μm/px resolution (~20× magnification).
3. Generate four training categories based on expert annotations and hard negative sampling strategies:

| Category       | Definition                                                                 |
|----------------|----------------------------------------------------------------------------|
| Lesion         | Regions with alveolar destruction, inflammatory infiltration, or hemorrhage |
| Normal         | Morphologically normal alveolar areas adjacent to lesions                  |
| Normal-Far     | Normal lung tissue distant from lesions (spatial negative control)          |
| Normal-Red     | Hard negatives: morphologically normal but densely eosinophilic regions    |

### Quality Control
The export script enforces strict patch-level QC:
- Filters out blank areas, coverslip edges, scanning artifacts, and low-tissue-content patches via tissue coverage thresholds.
- Balances negative sample ratios and integrates hard negatives to improve model robustness to complex backgrounds.
- Customized labeling and sampling thresholds for mouse lung pathology, aligned with common Transformer-based WSI pipelines (annotation-guided patch sampling + hard negative mining).

## 2. Patch-Level ViT Lesion Classification Model (Lesion vs. Normal)
A Vision Transformer (ViT) binary classifier is built to distinguish Lesion from Non-Lesion (Normal / Normal-Far / Normal-Red) patches.

### Model Architecture
- Backbone: ImageNet-pretrained ViT (input resolution: 224×224).
- Classification head: Replaced with a 2-class linear output layer.

### Staged Fine-Tuning Strategy
| Stage                  | Operation                                                                 |
|------------------------|---------------------------------------------------------------------------|
| Stage 1 (`exp_qupath224_finetune`) | Freeze patch embedding and lower Transformer blocks; update high-level features + classifier |
| Stage 2 (`stage2`)     | Unfreeze full backbone; train with reduced learning rate to adapt to mouse lung histology |
| Stage 3 (`stage3b_clean`) | Short fine-tuning on cleaned training data to refine decision boundaries  |

### Training Details
- Loss: Cross-entropy loss with class weighting and hard sampling to mitigate class imbalance.
- Augmentation: Online random flips + mild color jitter to improve generalization across staining/scanning batches.
- Metrics: ROC-AUC, accuracy, sensitivity, specificity.
- Output: Best model weights `vit_best.pth`.

## 3. WSI-Level Inference & Case-Level Feature Aggregation (512×512, stride=256)
Inference is performed on full tissue regions without QuPath annotations:

### Key Steps
1. **Patch Generation**: Use a standalone pathological segmentation/slicing script to generate tissue coordinates (H5 files). Densely sample overlapping patches (512×512) with stride=256.
2. **TTA Inference**: Run `predict_dual_tta_safe.py` to apply test-time augmentation (rotations/flips) and average predictions at logit/probability level for stability.
3. **Post-Processing**:
   - HSV background filtering: Remove near-white/low-saturation patches to avoid skewed area statistics.
   - Temperature scaling (e.g., T=2.0): Calibrate logits for conservative, reliable probability outputs.
4. **Feature Aggregation**:
   - Patch-level outputs: `all_patches.csv`, grid intermediate files (`_rawgrid/*.npy`).
   - Case-level summary: `case_scores.csv` (high-probability lesion area, patch density, probability percentiles, etc.).

### Important Notes
- `case_scores_filtered.csv`: Cleaned/ID-normalized version of `case_scores.csv` (**does NOT contain IL-6 or area values**). It is merged internally with external metadata (`1IL6Area_clean.csv`) for downstream analysis.
- If only `all_patches.csv` is available, run `make_case_prob_from_patches.py` to reconstruct case-level features, then clean with `filter_case_prob.py`.

## 4. Injury Score Calibration & IL-6/Area Correlation Analysis
Use `calib_gridsearch_v4.py` to optimize a continuous pathological injury score:

### Core Logic
1. Grid search over thresholds, nonlinear transformations, and normalization schemes.
2. Align `case_scores_filtered.csv` with `1IL6Area_clean.csv` (containing IL-6, area, etc.) by case ID.
3. Select parameters that maximize Pearson correlation between the injury score and `log(1+IL-6)` (or transformed area).
4. Output a 1D continuous injury score for correlation visualization, LASSO/Boruta feature selection, and multimodal modeling.

> Key: Injury score computation relies **only on numerical outputs** (case/patch tables, rawgrid files). Heatmaps are for visualization only and not used in scoring.

## 5. Heatmap Visualization & False Positive Suppression
Run `uni_overlay.py` to map patch probabilities back to WSI coordinates and generate publication-quality heatmaps/overlays:

### Visualization Strategy
1. Grid construction: Build regular grids using the **actual stride (256 pixels)** to fill probability values.
2. False positive suppression:
   - Zero out grids below a fixed threshold (e.g., 0.64).
   - Remove small connected components at grid level (e.g., `min_comp_tiles`) to eliminate isolated false positives.
3. Image rendering:
   - Interpolate to continuous maps with a unified [0,1] color scale and color bar.
   - Add high-probability contours; enforce `--force_stride 256` to avoid grid artifacts.
4. Outcome: Reduces false alarms on eosinophilic normal regions while maintaining interpretability and reproducibility.
