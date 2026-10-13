# Third-party notices

## torchvision ViT and ImageNet initialization

The v3, v4 and v6 configurations use the ViT-B/16 implementation from [torchvision](https://github.com/pytorch/vision) and official `ViT_B_16_Weights.IMAGENET1K_V1` initialization. ImageNet initialization files and trained study checkpoints are not redistributed in this repository. Upstream software and model terms remain applicable.

## Owkin Phikon-v2

The v5 comparison uses [Owkin/phikon-v2](https://huggingface.co/owkin/phikon-v2), revision `2ae989a9c40cffaa27f0a6cb29cc94d1d6f9a5fd`. Copyright and attribution remain with Owkin Inc. The [original license](third_party/phikon_v2/LICENSE.pdf) and its [text extraction](third_party/phikon_v2/LICENSE.extracted.txt) are included.

The license covers non-commercial research use by non-profit entities and includes conditions for redistribution of results. Phikon-derived v5 scores, statistics and comparison displays are distributed subject to the same applicable terms, not the repository's Apache 2.0 code license. The affected material is identified in [results/LICENSE_SCOPE.md](results/LICENSE_SCOPE.md). No Phikon weights or feature caches are distributed. The original license disclaimers apply; no upstream affiliation or endorsement is implied.

## Study images

The heatmap overviews contain study H&E image derivatives. Original study PNGs and whole-slide images are not included. The repository does not distribute the separate public mouse infection image archive, which was not used in any reported model configuration.
