# Data and metadata

This directory contains metadata for reproducing the reported partitions and inference locations. Raw PNG images, whole-slide images and model weights are not included.

| File | Contents |
|---|---|
| `MANIFEST_INDEX.json` | Row counts and hashes for the 12 role manifests and coordinate table |
| `SOURCE_ANIMAL_IDENTITIES.json` | Source content hashes mapped to canonical animal identifiers |
| `MANIFEST_ARCHIVE.json` | Size and SHA256 of each archive part and the reconstructed ZIP |
| `manifests.zip.part001`–`part005` | Fixed role manifests, asset identities and 132,032 window coordinates |

`scripts/prepare_data.py` reassembles and verifies the archive automatically. The parts must remain together and unmodified. Metadata alone can be extracted with:

```bash
python scripts/prepare_data.py --out local_data
```

Image-based analyses require the additional asset mapping options described in [Reproducibility](../docs/REPRODUCING.md).

## Manifest roles

`a6_train`, `a6_val` and `a6_test` describe patch classification. Each `a5_fold1`, `a5_fold2` and `a5_fold3` group contains training, validation and held-out annotated roles for cross-fitted animal scoring. Training and validation roles exclude the corresponding held-out animals and source aliases.

## Key fields

| Field | Definition |
|---|---|
| `label` | Model class index: 0 = Lesion; 1 = Normal |
| `class` | Recorded class name, Lesion or Normal |
| `source_id` / `source_sha256` | Content-based source slide identity |
| `canonical_animal_id` / `case_id` | Animal identifier used for role separation or scoring |
| `pixel_sha256` | SHA256 of `224,224,RGB:` followed by decoded RGB bytes |
| `source_png_sha256` | SHA256 of the original PNG file bytes |
| `split` / `outer_fold` | Data role and outer-fold assignment |
| `path` / `source_path` | Relative path expected after asset preparation |
| `x`, `y`, `crop_size`, `level` | Level-0 window origin and footprint in source pixels |

Relative paths are resolved against the extracted manifest directory. File names and repeated windows do not define independent biological samples. Model prediction and outcome tables are documented in [Results](../results/README.md).
