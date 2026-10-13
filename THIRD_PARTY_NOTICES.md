# Third-party notices

The existing Apache2.0 repository license applies to the repository code as appropriate; it does not override upstream restrictions on pretrained models, third-party data or derived results.

## torchvision / ImageNet ViT

The v3/v4/v6 backbone implementation is torchvision ViT-B/16, using official `ViT_B_16_Weights.IMAGENET1K_V1` initialization. See [torchvision](https://github.com/pytorch/vision) and its upstream license. We do not redistribute the ImageNet initialization or study model checkpoints in this Git repository. A new two-class head is trained and all backbone parameters are fine-tuned. No provider endorsement is claimed.

## Owkin Phikon-v2 (comparison v5)

Source: [Owkin/phikon-v2](https://huggingface.co/owkin/phikon-v2), revision `2ae989a9c40cffaa27f0a6cb29cc94d1d6f9a5fd`; Owkin Inc. Original complete [license](third_party/phikon_v2/LICENSE.pdf) and [text extraction](third_party/phikon_v2/LICENSE.extracted.txt) are included. Refer to the PDF as the authoritative text. This is a restricted non-commercial research license for non-profit entities, including same-conditions requirements for sharing results. It is not an unrestricted commercial open-source license. Materials are provided without the warranties disclaimed in that license, and without endorsement by Owkin.

The backbone was used frozen; CLS features were L2-normalized per image and fixed local logistic-regression heads fitted to this study's own training roles. No Phikon weights or feature caches are distributed here. The v5 scores/statistics and comparisons derived from them are identified as Phikon-derived results and distributed subject to the **same Owkin license terms**, not Apache2.0. See [results scope](results/LICENSE_SCOPE.md). Retain these notices and license when reusing or redistributing those results; uses outside the license require the rights holder's authorization.

## Study images and external mouse archive

Study PNG/WSI assets remain with the authors and are not included. Displayed H&E heatmap overview panels derive from the study's images. The separately inspected Zenodo17486453 mouse infection archive was not used in any v2–v6 computation and is not distributed by this repository. It must not be represented as additional training or independent validation data for these results.

## Historical code

Earlier modules under `VisualTransformer/` retain their upstream headers and attribution. This release does not reassign their authorship or extend permissions for third-party dependencies.
