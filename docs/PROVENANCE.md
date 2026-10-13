# Implementation and result provenance

The reported numerical results were generated on 26 September 2026. Prediction tables retain the original probability values and row identities; exported metadata replace machine-specific paths with relative asset paths. SHA256 records bind the distributed tables, manifests and source identities.

## Distributed implementation

`rebuild_v6/` supplies command-line entrypoints for training, classification evaluation, whole-slide inference, animal scoring and statistical analysis. Filesystem configuration, asset lookup and checkpoint identity were adapted for distribution. The training loop and selected numerical functions are compared with the source used for the reported v6 run using abstract syntax tree equivalence tests.

The distributed checkpoint format is 106; the original run used format 6. The formats require their matching code and configuration. CPU tests and statistical verification have been completed, but the distributed implementation has not been evaluated by repeating the complete GPU training and inference pipeline. Reproduction on a different software or hardware stack may therefore yield different fitted models.

## Verification records

- `RELEASE_FILES_SHA256.json`: file-level SHA256 inventory.
- `data/MANIFEST_INDEX.json`: original and distributed manifest hashes and row counts.
- `data/MANIFEST_ARCHIVE.json`: integrity information for the multipart manifest archive.
- `results/predictions/PROVENANCE.json`: source and export hashes for dense predictions.
- `results/RELEASE_NUMERIC_CHECK.json`: numerical verification summary.
- `tests/reference/SOURCE_SHA256.json`: original-source and extracted-fixture hashes used by equivalence tests.

The reference files are test inputs, not additional executable workflows. Full training records, original model checkpoints and image assets are retained by the authors. Scientific data, predicted probabilities and fitted model outputs are unchanged by repository documentation or packaging updates.
