# Pathological Image Preprocessing and Patch Generation Pipeline

This study uses mouse lung tissue H&E-stained whole-slide images (WSIs) as input. The overall workflow produces two types of patches for distinct purposes:

- **Training patches (224×224)**: exported from QuPath annotations, used for ViT training/validation.
- **Inference patches (512×512, stride=256)**: full-slide dense sampling for case-level inference and WSI visualization.

---

## Repository modules (where to find the code)

- `qupath/`  
  QuPath Groovy scripts only (tile export from annotations).

- `vit_wsi/`  
  ViT patch classifier + inference + case-level aggregation + evaluation + visualization.

- `pathology_seg/`  
  Standalone WSI tissue detection / coordinate generation / patch slicing utilities (H5, thumbnails, level check), used to support dense inference (512×512, stride=256).

---

## 1. QuPath Annotation & Training Patch Export (224×224)

### Core workflow
1. Annotate lesion regions on WSIs in QuPath (v0.4+).
2. Run the Groovy export script **`qupath/qupath_tiler_v2.groovy`** to crop tissue regions into **224×224** patches at ~0.5 μm/px (~20×).
3. Export four patch categories (expert annotation + hard negative mining):

| Category       | Definition |
|----------------|------------|
| Lesion         | Regions with alveolar destruction, inflammatory infiltration, or hemorrhage |
| Normal         | Morphologically normal alveolar areas adjacent to lesions |
| Normal-Far     | Normal lung tissue distant from lesions (spatial negative control) |
| Normal-Red     | Hard negatives: morphologically normal but densely eosinophilic regions |

### Quality control
The export script enforces patch-level QC:
- Filters blank areas, coverslip edges, scanning artifacts, and low-tissue-content patches via tissue coverage thresholds.
- Balances negative sample ratios and integrates hard negatives to improve robustness.

---

## 2. Patch-level ViT Lesion Classification (Lesion vs Non-Lesion)

A Vision Transformer (ViT) binary classifier distinguishes Lesion from Non-Lesion
(Normal / Normal-Far / Normal-Red).

### Model architecture
- Backbone: ImageNet-pretrained ViT (input resolution 224×224).
- Head: 2-class linear classifier.

### Staged fine-tuning strategy
| Stage | Operation |
|------|-----------|
| Stage 1 (`exp_qupath224_finetune`) | Freeze patch embedding + lower blocks; train high-level blocks + head |
| Stage 2 (`stage2`) | Unfreeze full backbone; lower LR to adapt to mouse lung histology |
| Stage 3 (`stage3b_clean`) | Short fine-tune on cleaned data to refine boundaries |

### Training details (code)
Training script: **`vit_wsi/pipeline/train_finetune.py`**  
Core modules: **`vit_wsi/core/{dataset.py, model.py, utils.py}`**

- Loss: cross-entropy (+ class weighting / sampling).
- Augmentation: flips + mild color jitter.
- Metrics: ROC-AUC, accuracy, sensitivity, specificity.
- Output: best weights `vit_best.pth`.

---

## 3. WSI-level Inference & Case-level Feature Aggregation (512×512, stride=256)

Inference is performed on full tissue regions (without QuPath annotations).

### Key steps
1. **Tissue coordinates / patch grid generation (H5)**  
   Use scripts under **`pathology_seg/`** to detect tissue regions and generate dense sampling coordinates (512×512, stride=256).  
   (Entry points: `pathology_seg/1-create_patches_fp.py`, `pathology_seg/2-get_png.py`, `pathology_seg/3-level_check.py`.)

2. **TTA inference**  
   Run **`vit_wsi/pipeline/predict_dual_tta_safe.py`** to apply test-time augmentation and average predictions for stability.

3. **Post-processing**
   - HSV background filtering: remove near-white / low-saturation patches.
   - Temperature scaling (e.g., T=2.0): conservative probability calibration.

4. **Feature aggregation**
   - Patch-level outputs: `all_patches.csv`, grid intermediates (`_rawgrid/*.npy`).
   - Case-level summary: `case_scores.csv` (area-like metrics, percentiles, density, etc.).
   - If only `all_patches.csv` is available:
     - build case table: **`vit_wsi/pipeline/make_case_prob_from_patches.py`**
     - clean/normalize: **`vit_wsi/pipeline/filter_case_prob.py`**
   - `case_scores_filtered.csv` does **NOT** contain IL-6/area values; it is merged with external metadata (e.g., `1IL6Area_clean.csv`) downstream.

---

## 4. Injury Score Calibration & IL-6/Area Correlation Analysis

Use **`vit_wsi/evaluation/calib_gridsearch_v4.py`** to optimize a continuous pathological injury score.

### Core logic
1. Grid search over thresholds, nonlinear transforms, normalization schemes.
2. Align `case_scores_filtered.csv` with `1IL6Area_clean.csv` by case ID.
3. Select parameters maximizing Pearson correlation between injury score and `log(1+IL-6)` (or transformed area).
4. Output a 1D continuous injury score for correlation visualization and downstream feature selection.

> Injury score computation relies only on numerical case/patch tables and rawgrid files. Heatmaps are visualization only.

---

## 5. Heatmap Visualization & False Positive Suppression

Run **`vit_wsi/visualization/uni_overlay.py`** to map patch probabilities back to WSI coordinates and generate publication-quality overlays.

### Visualization strategy
1. Grid construction: use the actual stride (256) to fill probability grids.
2. False positive suppression:
   - apply fixed thresholding (e.g., 0.64),
   - remove small connected components (`min_comp_tiles`).
3. Rendering:
   - interpolate probabilities on a unified [0,1] scale,
   - optional contours; enforce `--force_stride 256` to avoid grid artifacts.
