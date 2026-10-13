# Research results

## Numerical tables

| File | Contents |
|---|---|
| `tables/a6_all_six_models.csv` | Classification metrics and source-cluster AUC intervals for six configurations |
| `tables/all_associations_v2_v3_v4_v5_v6.csv` | All 40 reported correlations, p-values and available intervals |
| `tables/animal_scores_all_versions.csv` | Five model scores and recorded outcomes for the 21 animals |
| `tables/case_scores_per_seed.csv` | 63 v6 animal-by-seed scores |
| `tables/case_scores_equal_seed_mean.csv` | 21 v6 scores averaged equally over seeds |
| `tables/holm_primary4.csv` | Four primary v6 tests with Holm adjustment |
| `tables/influence_all112.csv` | Fixed leave-one-animal/group diagnostics |
| `tables/within_group_descriptive.csv` | Within-group-centered associations |
| `tables/spatial_per_reference.csv` | Coverage and annotated-center spatial summaries |
| `tables/spatial_per_included_object.csv` | Object-level spatial summaries |

The primary association analysis uses all 21 animals. The fixed 18-animal sensitivity analysis is reported separately. Window, seed and annotation-version counts are not animal sample sizes.

## Predictions

`predictions/a6_v6_val.csv` and `a6_v6_test.csv` contain 7,909 validation and 8,704 test predictions. The nine `a5_seed*_fold*.csv.gz` files contain all 396,096 v6 dense-window predictions. Source hashes, fold, seed and window identities are included for verification.

In classification tables, `label_index = 0` corresponds to Lesion, while `y_lesion = 1` is the positive label used for binary metrics. `prob_lesion` is the lesion-class probability. The decision threshold is recorded in `tables/validation_threshold_v6.json` and was selected using validation data.

`training/` contains the ten v6 training histories and their checkpoint-selection summary. `figures/` contains ROC curves, animal associations, group summaries, heatmap overviews for all 21 animals, and spatial diagnostics. Available SVG files provide vector versions of the corresponding plots.

## Interpretation and verification

The comparisons are exploratory and use shared test sources. The source-cluster AUC interval has four source clusters. Spatial reference coverage is a measure of evaluated extent, not accuracy or segmentation overlap. Full methodological limits are described in [Methods](../docs/METHODS_AND_LIMITATIONS.md).

`python scripts/verify_results.py --full-bootstrap` recomputes the included prediction-derived statistics. Phikon-derived v5 entries and comparison displays retain the license scope described in [LICENSE_SCOPE.md](LICENSE_SCOPE.md).
