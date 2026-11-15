# Software tests

The tests check class conventions, source/animal separation, training-only sampling probabilities, image preprocessing, test-time augmentation, manifest integrity, statistical kernels and equivalence with reference training code.

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The suite runs on CPU. `scripts/verify_results.py` separately checks the numerical results derived from the supplied research predictions.

The seven files in `reference/` contain selected function definitions and the training-loop body from the reported v6 implementation. The tests parse these files and compare their syntax trees with the distributed code. Function comparisons exclude docstrings and constant error-message text; calculations, data strings, calls and validation conditions are compared. The training-loop body is compared in full. Original-source and extracted-fixture SHA256 values are recorded in `reference/SOURCE_SHA256.json`.
