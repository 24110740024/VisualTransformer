# Software tests

The tests check class conventions, source/animal separation, training-only sampling probabilities, image preprocessing, test-time augmentation, manifest integrity, statistical kernels and equivalence with reference training code.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The suite runs on CPU. It does not train a study model or establish model generalization. Numerical verification of the supplied research predictions is provided separately by `scripts/verify_results.py`.

The seven files in `reference/` contain the selected function definitions and training-loop body used for syntax-tree comparisons with the reported v6 implementation. They exclude environment configuration and orchestration code and are parsed without execution. Executable syntax is preserved; documentation wording is standardized. Original-source and extracted-fixture SHA256 values are recorded in `reference/SOURCE_SHA256.json`.
