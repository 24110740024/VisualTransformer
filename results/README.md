# Research Results

## Summary Tables

| File | Contents |
|---|---|
| `tables/a6_all_six_models.csv` | Patch classification metrics and source-cluster AUC CIs for all six configurations |
| `tables/all_associations_v2_v3_v4_v5_v6.csv` | Full-precision correlations, p-values, and bootstrap intervals across models |
| `tables/animal_scores_all_versions.csv` | Animal-level scores across all five pathology models alongside recorded outcomes (`n = 21`) |
| `tables/case_scores_per_seed.csv` | Per-seed `v6` animal scores (21 animals × 3 seeds) |
| `tables/case_scores_equal_seed_mean.csv` | Equal-weight three-seed mean `v6` scores (`n = 21`) |
| `tables/holm_primary4.csv` | Four primary `v6` association tests with Holm correction |
| `tables/influence_all112.csv` | Leave-one-animal and leave-one-group sensitivity diagnostics |
| `tables/within_group_descriptive.csv` | Descriptive within-group-centered associations |
| `tables/spatial_per_reference.csv` | Window coverage and annotated-center spatial summaries per reference |
| `tables/spatial_per_included_object.csv` | Object-level spatial summary statistics |

Primary association analyses include all 21 animals; the 18-animal sensitivity analysis is archived alongside them.

## Predictions, Training Logs, and Figures

- `predictions/a6_v6_val.csv` and `predictions/a6_v6_test.csv` store the 7,909 validation and 8,704 test patch predictions (`label_index = 0` denotes Lesion; `y_lesion = 1` is the binary positive target; `prob_lesion` is the predicted lesion probability). The validation-selected Youden's J threshold is stored in `tables/validation_threshold_v6.json`.
- `predictions/a5_seed*_fold*.csv.gz` store all 396,096 `v6` dense-window predictions with source hashes, fold/seed labels, and window coordinates.
- `training/` contains the ten `v6` training logs and checkpoint selection summary (`selection_summary.json`).
- `figures/` contains ROC curves, animal association scatter plots, group summaries, whole-slide heatmap overviews for all 21 animals, and spatial diagnostics (with SVG vector versions where available).

## Verification

Run `python scripts/verify_results.py --full-bootstrap` to recompute all summary statistics and bootstrap intervals from the included predictions. See [Methods & Limitations](../docs/METHODS_AND_LIMITATIONS.md) for study scope and statistical interpretation, and [LICENSE_SCOPE.md](LICENSE_SCOPE.md) for license terms covering Phikon-derived (`v5`) comparison results.
