"""Bind deterministic train-only probabilities to the portable manifest bytes."""
import hashlib
from sampling import POLICY, probabilities

def load_training_probabilities(manifest,train_rows):
    p=probabilities(train_rows)
    return p,dict(policy=POLICY,train_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                  probability_float64_sha256=hashlib.sha256(p.astype('<f8').tobytes()).hexdigest(),
                  rows=len(train_rows),samples_per_epoch=len(train_rows),replacement=True,loss_class_weights=None)
