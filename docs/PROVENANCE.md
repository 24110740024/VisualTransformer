# Implementation and Artifact Provenance

The reported numerical results were generated on September 26, 2026. Released prediction tables preserve the original probability outputs and canonical row identifiers, with machine-specific paths normalized to relative paths. SHA-256 inventories bind the released manifests, predictions, and summary tables.

## Software Package

`rebuild_v6/` provides modular command-line entrypoints for training, patch classification evaluation, whole-slide dense inference, animal-level scoring, and statistical verification. Automated unit tests under `tests/` verify source separation, balanced sampling logic, and numerical consistency between the packaged pipeline and the original experiment scripts.

All statistical metrics, confidence intervals, and tables in `results/` have been verified on CPU via `scripts/verify_results.py`. Re-running full GPU training from scratch on a different CUDA or PyTorch stack may exhibit minor floating-point variation typical of nondeterministic GPU kernels.

## Patch Export Provenance

The exact QuPath export script used in this study (`preprocessing/qupath/qupath_tiler_v2.groovy`) is included unmodified alongside its export metadata (`preprocessing/qupath/EXPORT_PROVENANCE.json`). See [Patch Generation](PATCH_GENERATION.md) for the distinction between ROI annotations, sampled negative classes, exported PNG patches, and dense whole-slide windows.

## Integrity Manifests

- `RELEASE_FILES_SHA256.json`: file-level SHA-256 checksums across the repository.
- `preprocessing/qupath/EXPORT_PROVENANCE.json`: QuPath script checksum, patch counts, and export summary.
- `data/MANIFEST_INDEX.json` & `data/MANIFEST_ARCHIVE.json`: split manifest hashes and multipart archive metadata.
- `results/predictions/PROVENANCE.json`: provenance metadata for dense whole-slide predictions.
- `results/RELEASE_NUMERIC_CHECK.json`: summary of automated numerical verification.
- `tests/reference/SOURCE_SHA256.json`: checksums for test reference fixtures.

Original whole-slide images and trained checkpoints are retained by the study authors. Documentation updates do not modify any underlying scientific data, prediction tables, or reported metrics.
