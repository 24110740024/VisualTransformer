# October13 release changes

Base repository revision: `0a66e758ba51175fb51becba4f9d14d1ca71c5a7`.

The old root README is preserved in [legacy_readme_before_20261013.md](legacy_readme_before_20261013.md). Existing code under `VisualTransformer/` is retained; a new current-entrypoint notice distinguishes it from the rebuilt workflow.

| Earlier description | Current audited rebuild |
|---|---|
| Staged fine-tuning | v6 full-parameter ImageNet ViT-B/16 training, two LR groups, fixed sampler |
| Fixed512 dense window | v6 calibrated408/409 footprint on all original132032 center identities |
| Injury-score parameter search against transformed IL6 | Direct raw all-window probability mean, then equal seed mean; outcomes only in final association |
| Temperature scaling / display false-positive suppression | No calibration/filtering in the score; fixed0–1 continuous display, exact footprint geometry |
| Old clean4 8943 / AUC0.852075 | Current original-export8704 test set; v6 AUC0.891999; all comparison versions reported |
| Single headline without full development context | All five pathology variants + ResNet; prior test visibility and posthoc presentation disclosed |

## Portable layer versus original run

The `archive/20260926` source files are retained byte-for-byte and bound by SHA. The `rebuild_v6` layer adapts path-dependent guards, asset lookup, manifest binding, checkpoint format and CLI orchestration. It does not overwrite frozen inputs or results. The training body from RNG seeding through optimization/validation/checkpoint selection is AST-identical to the original v6 body. Sampling, transforms, model/optimizer construction, metric/bootstrap and dense preprocessing/TTA functions are also compared by AST.

CPU verification reproduces existing predictions' metrics and statistics; it is not a new training run or proof of numerical identity on different hardware. The new adapter has not regenerated all model weights. Original archived format6 and new portable format106 are intentionally distinct.

Public tables remove machine paths and private historical filenames; probability strings are retained, and identities remain source/pixel hashes with row indices. All nine v6 prediction outputs are included, without choosing animals, seeds or windows. Data preparation must use the unchanged published manifest bytes. Scientific results are dated September26; software packaging/checks are dated October13.
