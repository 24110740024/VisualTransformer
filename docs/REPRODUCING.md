# Reproduction and data access

The public release has two levels. Existing predictions and statistics can be checked on CPU with no raw images or model weights. Training/inference additionally require the authors' original image assets and a CUDA environment. Downloading this repository alone does not provide those assets. Requests for raw images and trained checkpoints should be directed to the repository maintainers; access is not provided by this repository.

## 1. Verify the published results

Use Python 3.11 or newer (the archived environment used Python 3.12.14). Install `requirements-analysis.txt` in a fresh environment:

```bash
python -m pip install -r requirements-analysis.txt
python scripts/verify_results.py
python scripts/verify_results.py --full-bootstrap
```

The verification script checks the 7,909 validation / 8,704 test rows, positive-class convention, validation-only Youden threshold, A6 metrics, all 396,096 v6 A5 probabilities, 63 seed-animal means, 21 final means, 40 associations across five variants, within-variant Holm adjustments and all 112 v6 deletion diagnostics. `--full-bootstrap` additionally recomputes the v6 5,000 patch / 10,000 source-cluster AUC bootstrap and four 10,000-draw paired-animal intervals. Bootstrap outputs condition on fitted models; they do not include training uncertainty.

## 2. Prepare authorized image data

The original PNG exports and 21 dense source slides are required for image-based reproduction. The original QuPath exporter, annotation rules and PNG-generation parameters are described in [Patch generation](PATCH_GENERATION.md). The fixed manifests below define the reported dataset; freshly generated patches do not automatically reproduce those identities. The release supplies exact row/source/pixel identities and the accepted 408/409-pixel coordinates in `data/manifests.zip.part001` and subsequent numbered parts. The scripts automatically reassemble the original ZIP in memory and verify every part and the full ZIP against `data/MANIFEST_ARCHIVE.json`; the split is only a transport change. The distributed manifests use relative asset paths while retaining the original row and content identities. `MANIFEST_INDEX.json` binds the original and portable representations by SHA256. The supplied manifest hashes define the reported data partitions.

```bash
python scripts/prepare_data.py --out local_data --png-root /path/to/original_pngs --wsi-root /path/to/original_wsis
```

This copies matching assets into `local_data/images` and `local_data/wsi`, named by identity hash, and creates `local_data/manifests`. Both source roots are read-only inputs. Expect tens of GB of free space. Omitting both roots extracts metadata only; it is not a successful image-data preparation. Missing/hash-mismatched assets cause failure, with partial output retained for inspection. Use a new output directory for another attempt.

The decoded PNG hash includes the exact prefix `224,224,RGB:` followed by RGB bytes. Image-file SHA and decoded-pixel SHA are different identities. Canonical source/animal role assignment was performed before this release; aliases are not treated as new samples.

## 3. Train the portable v6 recipe

Install `requirements.txt` with a CUDA-compatible PyTorch/torchvision pair. `environment/recorded_environment_20260926.txt` records the original environment; broad installation constraints are convenience bounds, not a guarantee of bitwise equivalence. The distributed implementation has been tested on CPU but has not regenerated the reported models through a complete GPU training run.

Obtain the official [torchvision ImageNet1K V1 ViT-B/16 initialization](https://download.pytorch.org/models/vit_b_16-c867db91.pth). Set `VIT_IMAGENET_WEIGHTS` to its local path (or place it in PyTorch's checkpoint cache). The code verifies full SHA256:

`c867db91d3e12c6cbadabb610d73c24a546bf82d8c03a9fea34f43a712ddb0e9`

First run the CPU input check. The `--device cuda` option names the intended production recipe; `--preflight_only` does not create a model or use the GPU:

```bash
python rebuild_v6/train.py --train_manifest local_data/manifests/a6_train.csv --val_manifest local_data/manifests/a6_val.csv --out_dir outputs/a6_seed42 --seed 42 --device cuda --weighted_sampler --preflight_only
```

Then remove `--preflight_only` to train A6. For A5, use each `a5_fold1`, `a5_fold2`, `a5_fold3` train/val pair, separately for seeds 42, 43, 44, giving nine models. There is one A6 model (seed42) and nine A5 models. The specified analysis combines all three seeds in each fold. The model checkpoint is selected only by its own validation Lesion AUC. For interrupted portable runs, pass `--resume outputs/<run>/last.pth` with otherwise identical arguments/code/runtime. Resume requires a checkpoint produced by the same implementation and configuration.

The distributed implementation uses checkpoint format 106; the reported v6 run used format 6. The formats are not interchangeable. Original checkpoints require the matching original runtime and configuration. Only load checkpoints from trusted sources: training checkpoints include Python/NumPy RNG objects.

## 4. A6 evaluation

```bash
python rebuild_v6/evaluate.py --train local_data/manifests/a6_train.csv --val local_data/manifests/a6_val.csv --test local_data/manifests/a6_test.csv --checkpoint outputs/a6_seed42/best.pth --out outputs/a6_evaluation
```

Lesion is model index0, but metric-positive label1. A6 uses complete 224-pixel PNGs, no TTA, CUDA FP16, and softmax in the logits' original precision. The Youden threshold is locked on validation before test inference. If the optimum threshold is nonfinite, threshold-dependent metrics are reported as unavailable. The source-cluster bootstrap has four test WSI clusters.

## 5. A5 inference and aggregation

Example for seed42/fold1; repeat for all nine prescribed pairs:

```bash
python rebuild_v6/predict_dense.py --coordinates local_data/manifests/coordinates_all21.csv --train local_data/manifests/a5_fold1_train.csv --val local_data/manifests/a5_fold1_val.csv --checkpoint outputs/a5_seed42_fold1/best.pth --seed 42 --fold 1 --out outputs/a5_seed42_fold1_predictions
```

The reader accepts the actual level0 tiled JPEG TIFF representation used in the study; it fails on unsupported TIFF layouts instead of silently decoding them differently. Every held-out source is checked against train and validation. Four flips each produce a class probability, and these probabilities are averaged in FP16. No temperature scaling, outcome fitting, ROI-based selection or post-manifest filtering is performed.

```bash
python rebuild_v6/aggregate.py --coordinates local_data/manifests/coordinates_all21.csv --runs outputs/a5_seed42_fold1_predictions outputs/a5_seed42_fold2_predictions outputs/a5_seed42_fold3_predictions outputs/a5_seed43_fold1_predictions outputs/a5_seed43_fold2_predictions outputs/a5_seed43_fold3_predictions outputs/a5_seed44_fold1_predictions outputs/a5_seed44_fold2_predictions outputs/a5_seed44_fold3_predictions --out outputs/a5_aggregate
python rebuild_v6/analyze.py --scores outputs/a5_aggregate/animal_scores.csv --outcomes results/tables/animal_scores_all_versions.csv --out outputs/a5_statistics
```

The outcome table is used only after prediction and aggregation. The fixed 18-animal sensitivity subset excludes the fixed `30-30-120-N` group; the primary analysis includes all 21 animals. All 112 deletion diagnostics remain in the report.

## 6. Spatial outputs and implementation provenance

`rebuild_v6/rectangle_field.py` provides the original exact half-open rectangle mathematics for 408/409 footprints. Three all-case overview PNGs and the spatial summary are included. Full-resolution case arrays, detailed panels, independent ROI geometry and the complete audited spatial workflow are retained by the authors and are not included in this repository. Heatmaps are descriptive probability fields, not pixel segmentation. A display-grid image must not be used to calculate exact ROI measurements.

Implementation provenance and the distinction between the distributed software and the reported model runs are documented in [PROVENANCE.md](PROVENANCE.md). The reference sources used by the numerical-equivalence tests are in `tests/reference/`.

Run CPU software tests after installing the full dependencies:

```bash
python -m unittest discover -s tests -v
```
