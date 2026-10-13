# QuPath patch generation

The classification dataset consists of PNG patches exported from mouse lung H&E whole-slide images using existing QuPath annotations supplied by research collaborators. Patch export and downstream computational analyses used those annotations; the export procedure does not create new expert annotations or constitute a separate patch-by-patch pathological review.

The [export script](../preprocessing/qupath/qupath_tiler_v2.groovy) is supplied without modification. [Export provenance](../preprocessing/qupath/EXPORT_PROVENANCE.json) records its SHA256, export counts and links to the fixed training manifests. Saved projects record QuPath 0.6.0. The script header specifies QuPath 0.4+, but the exact application runtime used for each export was not independently recorded.

## Source images and annotations

The saved collection contains 32 project-image entries corresponding to 29 distinct source-slide hashes. Repeated project entries for the same source are not independent slides or animals. There are 104 recovered annotation objects: 99 with named pathology classes, four `Region*` objects and one unclassified object.

The script removes asterisks from class names and excludes names beginning with `Region` or `Ignore`. All remaining annotations are included in the lesion-candidate set, including the unclassified object. Thus, the selection is not restricted to an annotation class literally named `Lesion`. The retained pathology classes are listed in the provenance record.

An image without an included annotation is skipped. If no lesion patch is successfully exported, negative sampling also stops. An entirely unannotated slide is therefore not automatically converted into Normal training patches.

## Resolution and patch positions

| Parameter | Export setting |
|---|---|
| PNG dimensions | 224 × 224 pixels |
| Target resolution | 0.5 µm/pixel |
| Downsample | Target resolution divided by the image server's averaged source pixel size |
| Level-0 crop width and height | `round(224 × downsample)` |
| Basic lesion/normal grid step | One level-0 crop width |
| Additional negative candidate grid step | `round(0.75 × crop width)` |
| Image-frame exclusion | Patch centers within 1,500 µm of a frame edge |
| Basic Normal search region | Bounding box of included annotations, expanded by 6,000 level-0 pixels and clipped to the image |

For Lesion patches, the grid is anchored to each ROI's bounding-box origin, rounded down to level-0 integer coordinates. The patch center must lie within that ROI. This rule does not require the entire patch to be contained within the annotation. Additional negative candidates can overlap because their grid step is smaller than the crop width.

The physical field of view depends on source calibration. The script substitutes 0.25 µm/pixel when the averaged calibration is NaN; that fallback is an assumption, not a measured calibration. Source calibration should be checked before applying the script to another collection. The target resolution does not by itself establish an optical magnification.

## Export categories

| Directory | Selection and target count |
|---|---|
| `Lesion` | Centers inside an included annotation; accepted patches define the positive count P |
| `Normal` | Centers outside every included annotation, scanned in raster order in the expanded search region; target P |
| `Normal-Far` | Eligible external centers satisfying a distance criterion; target `round(0.5 × P)` |
| `Normal-Red` | Eligible external centers selected using redness or texture, with fallback sampling; target `round(0.5 × P)` |

These are target counts, not guaranteed yields. The combined negative target can approach 2P. The three Normal export categories are combined into the Normal model class; they are not three additional model outputs.

Far and Red sampling initially require a distance of 2,000 µm. Distance is calculated to annotation bounding rectangles, with an explicit inside-ROI check; it is not the exact distance to an irregular lesion boundary. Red candidates additionally exceed either the 65th percentile of mean CIELAB a* or the 50th percentile of grayscale standard deviation in the eligible candidate pool.

Up to three selection rounds relax the distance by a factor of 0.7 to a floor of 1,200 µm. The numeric a* and standard-deviation thresholds are each reduced by 0.05, bounded below by their candidate-pool 50th and 45th percentiles. This is a subtraction from feature values, not a five-percentage-point decrement in percentile rank.

Far fallback retains the 1,200 µm distance floor. Red fallback samples the union of high-redness and high-texture candidates without a minimum-distance condition. These candidates still have centers outside the included annotations and have passed the image filters. Consequently, not every exported Normal-Red patch satisfies the initial distance criterion.

Each sampling call shuffles candidates with Java random seed 42. Coordinates are not globally excluded after selection, so categories or repeated selection rounds can contain the same location. Subsequent decoded-pixel deduplication is separate from patch export.

## Background and image filters

For every export, the returned image must be 224 × 224 and pass the background check. On a two-pixel sampling grid, background rejection uses any of:

- More than 60% near-white pixels, where all RGB channels are at least 245.
- Mean grayscale greater than 220 with standard deviation below 12.
- More than 85% low-optical-density pixels, where summed RGB optical density is below 0.25 and each channel uses `-ln((channel + 1) / 256)`.

Normal-category exports additionally require estimated tissue fraction of at least 0.15 and reject patches whose top or bottom 10% band has mean grayscale below 140. The latter dark-band filter is not applied in the Lesion export branch. These are image-selection heuristics, not a pixel-level tissue or lesion segmentation.

## Output and connection to training

Run the script on the intended image in an open QuPath project. Output is written under `<project>/tiles_v2/<image name>/`, with the four category directories above. File names contain the level-0 x/y origin and an export timestamp. The `WRITE_BASE_POS` and `WRITE_BASE_NEG` variables are unused by the export branches; changing them does not disable those exports. Re-execution can create additional timestamped files, so a new export should be kept separate from an existing study collection.

The accepted collection contains:

| Export category | PNG files |
|---|---:|
| Lesion | 17,074 |
| Normal | 13,729 |
| Normal-Far | 3,988 |
| Normal-Red | 8,544 |
| Total | 43,335 |

Original PNG bytes and decoded RGB identities were checked before training. Decoded-pixel identity is SHA256 of `224,224,RGB:` followed by RGB bytes. The collection contains 40,097 unique decoded pixel identities and no conflicting binary labels within an identity. Deterministic deduplication and source/animal-aware partitioning produce the fixed manifests distributed in [data](../data/README.md): 23,309 training, 7,909 validation and 8,704 test patches for classification, plus the three outer-fold role sets for animal scoring. Source-alias exclusions are also applied, so the classification totals need not equal the full collection's unique-pixel count.

All 43,335 PNG centers were checked against their corresponding saved project geometries and export categories: 43,332 ordinary matches and three boundary cases consistent with the recorded QuPath containment semantics. This establishes coordinate/category agreement. It does not establish exhaustive annotation of the slide or independent expert adjudication of every sampled negative patch. Unannotated tissue outside this export procedure has no automatically assigned normal ground truth.

## Classification patches and whole-slide inference

The PNG exports are the annotated classification dataset. The 132,032 fixed whole-slide inference windows used for animal scoring are a separate input set. For v6, these use 408/409-pixel source windows resized to 224 pixels, with four-flip probability averaging. The export script does not generate that dense inference manifest. A slide without a saved ROI can undergo whole-slide inference without contributing automatically labeled Normal training patches.

The script, saved project metadata and accepted exports document the input-generation process. QuPath export was not re-executed for this repository addition. Reproducing the exact dataset requires the corresponding source images, saved project annotations and fixed identity manifests, rather than newly drawing ROIs or treating fresh exports as the same dataset.
