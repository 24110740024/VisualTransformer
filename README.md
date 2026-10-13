# ViT mouse-lung H&E pathology: audited September 2026 rebuild

Code, predictions and results for the computational pathology component of a mouse heat/ozone co-exposure study. The September26 rebuild and bounded optimization are complete. This October13 release packages those fixed results; it does **not** retrain the models or edit the manuscript.

**Current entrypoint: `rebuild_v6/`.** v6 is the recommended main presentation, with all other completed recipes reported as comparisons. The choice of presentation was made after observing results, not a prospectively specified unique primary model. These are not new independent external-validation results.

[中文导航](docs/README_zh.md) · [Reproduce/check results](docs/REPRODUCING.md) · [Methods and limits](docs/METHODS_AND_LIMITATIONS.md) · [Changes from the old pipeline](docs/CHANGELOG_20261013.md)

## Workflow

```text
Original QuPath PNGs + source/alias identities
  → deterministic pixel deduplication and fixed source-aware partitions
  → A6: train / validation-selected checkpoint + threshold / held-out-source test
  → A5: three outer folds × three seeds, each held-out animal excluded from fitting
  → four-flip probabilities on fixed calibrated WSI windows
  → all-window animal means → equal seed mean
  → fixed raw Area / IL6 associations + uncertainty and robustness
```

- Audited original PNG files: **43,335**; unavailable historical derived records excluded: **15,538**.
- A6 patches: **23,309 train / 7,909 validation / 8,704 test**, from **19 / 5 / 4 source WSIs**.
- A5: **21 animals**, seven groups of three; **132,032 fixed window identities**. v6 has nine outer-fold/seed models, **396,096 predictions**, 63 seed-animal scores and 21 equal-seed means.
- Lesion is model index **0**, metric-positive label **1**. No outcome-fitted injury-score calibration, LASSO selection, temperature scaling, ROI-based filtering or test-set threshold selection.

## Results

| Model recipe | A6 test AUC | 95% source-cluster interval |
|---|---:|---:|
| Random ViT v2 | 0.869556 | [0.814904, 0.931134] |
| Pretrained ViT v3 | 0.868257 | [0.825655, 0.923752] |
| ResNet-50 baseline | 0.855901 | [0.779518, 0.930580] |
| Scale-consistent ViT v4 | 0.887736 | [0.829997, 0.943078] |
| Frozen Phikon-v2 + linear v5 | 0.878963 | [0.772109, 0.946644] |
| Balanced-sampling ViT v6 | 0.891999 | [0.848038, 0.946250] |

The table uses each recipe's own validation threshold for threshold-dependent metrics. Different initialization/training/preprocessing/precision recipes make this a descriptive comparison, not an isolated architecture superiority test. Four test sources give limited cluster-level precision.

| Version | Area Pearson r | IL6 Pearson r |
|---|---:|---:|
| v2 | 0.525968 | 0.599750 |
| v3 | 0.643734 | 0.647543 |
| v4 | 0.628813 | 0.649805 |
| v5 | 0.652916 | 0.660155 |
| v6 | 0.699976 | 0.720957 |

v6 Area: Pearson p=0.000411566, Holm4 p=0.001234700. v6 IL6: Pearson p=0.000226535, Holm4 p=0.000906139. Both Spearman Holm4 p≈0.00594325. Full precision, all nominal/adjusted p-values, intervals and the fixed secondary18 analysis are in [the complete table](results/tables/all_associations_v2_v3_v4_v5_v6.csv).

The v6 within-group-centered correlations are only **0.054569 / 0.173181** (Area/IL6), descriptive without p-values. Do not interpret the across-animal correlations as strong within-group effects. Holm4 controls each recipe's four primary tests, not selection across recipes.

![Six recipe ROC comparison](results/figures/a6_all_six_models.png)

![v6 animal associations](results/figures/a5_primary21_v6.png)

## Contents and quick check

| Directory | Contents |
|---|---|
| `rebuild_v6/` | Portable v6 training, A6 evaluation, dense inference, aggregation, statistics and exact rectangle-field mathematics |
| `scripts/` | CPU result verifier and authorized-data preparation |
| `data/` | SHA-bound portable manifests and all132032 calibrated coordinates; **no raw images** |
| `results/` | All-variant summary/animal tables, v6 validation/test and all nine A5 prediction tables, training histories, plots and check receipts |
| `tests/` | CPU contract, identity-leakage and numerical-equivalence tests |
| `archive/20260926/` | Actual original v6/reference source bytes; requires the original sealed dependency environment to execute |
| `VisualTransformer/` | Earlier workflow retained for history; **not the current v6 entrypoint** |

```bash
python -m pip install -r requirements-analysis.txt
python scripts/verify_results.py
```

See [reproduction instructions](docs/REPRODUCING.md) for full bootstrap, data preparation and training/inference commands. The portable adapter has a new checkpoint identity (format106). Original format6 weights are not silently accepted. The published numbers come from the separately audited September26 run, not a fresh execution of the portable adapter.

## Availability and interpretation

Large model checkpoints, original PNGs, whole-slide images, offline libraries and full-resolution heatmap arrays remain in the authors' sealed handoff; they are not committed to Git. This repository is not a complete cloud-machine backup. No public download of raw study data or weights is currently asserted. Contact the repository maintainers for access arrangements.

ROI annotations are nonexhaustive references. Three dual-ROI cases remain separate and one unannotated case is not labelled Normal. Probability fields are **not pixel segmentation or calibrated lesion-area measurements**. See [all-case heatmap overviews](results/figures/overview_1.png), [overview2](results/figures/overview_2.png), [overview3](results/figures/overview_3.png) and [spatial diagnostics](results/figures/spatial_diagnostics.png). Coverage is not accuracy; all weak cases are retained. Area calibration, IL6 units and tissue pairing still require author records.

The old8,943-patch/0.852075-AUC result and correlation0.622/0.630 belong to a previous historical reconstruction. They do not describe this release. The older repository's correlation grid-search/LASSO-related injury-score approach and display suppression are superseded; see [change log](docs/CHANGELOG_20261013.md).

## Licenses and attribution

Original repository code retains [Apache2.0](LICENSE). Third-party resources and v5 Phikon-derived results are **not** relicensed under that code license; see [third-party notices](THIRD_PARTY_NOTICES.md) and [results-license scope](results/LICENSE_SCOPE.md). Phikon weights/features are not distributed. Public mouse infection images inspected separately were **not used in v2–v6** and are not redistributed here. No affiliation or endorsement by upstream model providers is implied.
