"""Evaluate a new A6 model with manifest-keyed, single-pass predictions.

Recovered protocol: Lesion is model index 0 but metric positive label 1;
validation-only Youden J; no TTA; full 224-pixel scale-consistent evaluation transforms; CUDA FP16;
5,000 patch-IID and 10,000 source-WSI cluster bootstrap draws, seed 20260919.
Never imports a historical threshold or trains/chooses a model using test data.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scale_transforms import build_transforms
from candidate_model import create_model, validate_checkpoint_identity, V2
from candidate_guard import gpu_guard

CLASSES = ["Lesion", "Normal"]
BOOTSTRAP_SEED = 20260919
PATCH_DRAWS = 5000
CLUSTER_DRAWS = 10000


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, value):
    temporary = Path(str(path) + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def read_manifest(path, expected_split):
    path = Path(path).resolve(strict=True)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"path", "label", "source_id", "historical_file", "split"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: missing {required - set(reader.fieldnames or [])}")
        rows = list(reader)
    keys = set()
    for row in rows:
        if row["split"] != expected_split or row["label"] not in ("0", "1"):
            raise ValueError(f"{path}: split or label mismatch")
        if not row["source_id"] or not row["historical_file"]:
            raise ValueError(f"{path}: blank source or record key")
        if row["historical_file"] in keys:
            raise ValueError(f"{path}: duplicate historical_file record key")
        keys.add(row["historical_file"])
        image = Path(row["path"])
        if not image.is_absolute():
            image = path.parent / image
        row["path"] = str(image.resolve(strict=True))
        row["label"] = int(row["label"])
        if row.get("class", CLASSES[row["label"]]) != CLASSES[row["label"]]:
            raise ValueError(f"{path}: class/label inconsistent")
    if not rows or {r["label"] for r in rows} != {0, 1}:
        raise ValueError(f"{path}: both labels required")
    return rows


def audit_manifests(paths):
    data = {split: read_manifest(path, split) for split, path in paths.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        for key in ("source_id", "path", "historical_file", "pixel_sha256"):
            if key == "pixel_sha256" and not all(r.get(key) for r in data[a] + data[b]):
                raise ValueError("Complete pixel_sha256 required for independent evaluation")
            values_a = {r[key] for r in data[a]}
            values_b = {r[key] for r in data[b]}
            if values_a & values_b:
                raise ValueError(f"{a}/{b} {key} overlap: {sorted(values_a & values_b)[:3]}")
    audit = {split: {"path": str(Path(paths[split]).resolve()),
                     "sha256": sha256(paths[split]), "rows": len(rows),
                     "sources": len({r["source_id"] for r in rows}),
                     "class_counts": dict(Counter(CLASSES[r["label"]] for r in rows))}
             for split, rows in data.items()}
    frozen = json.loads((V2 / "FROZEN.json").read_text(encoding="utf-8"))
    for split, rows in data.items():
        expected = frozen["manifests"]["a6_" + split]
        if (audit[split]["sha256"] != expected["sha256"]
                or len(rows) != expected["rows"]
                or audit[split]["sources"] != expected["source_count"]
                or audit[split]["class_counts"] != expected["classes"]):
            raise ValueError("Evaluation input differs from frozen original-export v2: " + split)
        if len({r["pixel_sha256"] for r in rows}) != len(rows):
            raise ValueError("Original-export v2 must contain unique pixels within each split")
    if audit["test"]["sources"] != 4:
        raise ValueError("Original-export v2 retains four source-independent test WSIs")
    return data, audit


class PredictionDataset(Dataset):
    def __init__(self, rows, img_size):
        self.rows = rows
        self.transform = build_transforms(train=False, img_size=img_size)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        with Image.open(self.rows[index]["path"]) as image:
            return self.transform(image.convert("RGB")), index


def predict(model, rows, device, batch_size, workers, img_size):
    loader = DataLoader(PredictionDataset(rows, img_size), batch_size=batch_size,
                        shuffle=False, num_workers=workers, pin_memory=device.type == "cuda")
    probabilities = np.full(len(rows), np.nan, dtype=np.float64)
    with torch.inference_mode():
        for images, indices in loader:
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                logits = model(images.to(device, non_blocking=True))
            # Match historical export: softmax on original logits dtype (FP16 on CUDA).
            p = logits.softmax(1)[:, 0].cpu().numpy().astype(np.float64)
            probabilities[indices.numpy()] = p
    if not np.isfinite(probabilities).all() or not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError("Missing or invalid model probabilities")
    return probabilities


def select_validation_threshold(labels, probabilities):
    """Historical clean-4 argmax policy; report rather than guess if it is nonfinite."""
    fpr, tpr, thresholds = roc_curve(labels, probabilities, pos_label=1)
    index = int(np.argmax(tpr - fpr))
    threshold = float(thresholds[index])
    return {"method": "validation-only Youden J; first np.argmax in sklearn roc_curve order",
            "threshold": threshold if np.isfinite(threshold) else None,
            "status": "locked" if np.isfinite(threshold) else "nonfinite_optimum_no_threshold_metrics",
            "youden_j": float((tpr - fpr)[index]),
            "validation_auc_lesion": float(roc_auc_score(labels, probabilities))}


def metrics(labels, probabilities, threshold):
    result = {"n": len(labels), "lesion_n": int(labels.sum()),
              "normal_n": int(len(labels) - labels.sum()),
              "auc_lesion": float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else None}
    if threshold is None:
        return result
    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = map(int, confusion_matrix(labels, predicted, labels=[0, 1]).ravel())
    result.update(TN=tn, FP=fp, FN=fn, TP=tp,
                  accuracy=(tp + tn) / len(labels),
                  precision=tp / (tp + fp) if tp + fp else 0.0,
                  recall=tp / (tp + fn) if tp + fn else 0.0,
                  specificity=tn / (tn + fp) if tn + fp else None,
                  f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return result


def bootstrap_auc(labels, probabilities, source_ids, source_order,
                  patch_draws=PATCH_DRAWS, cluster_draws=CLUSTER_DRAWS):
    """Uniform patch sampling, then independent uniform source-cluster sampling."""
    rng_patch = np.random.default_rng(BOOTSTRAP_SEED)
    patch_values = []
    for _ in range(patch_draws):
        indices = rng_patch.integers(0, len(labels), size=len(labels))
        if len(np.unique(labels[indices])) == 2:
            patch_values.append(float(roc_auc_score(labels[indices], probabilities[indices])))
    rng_cluster = np.random.default_rng(BOOTSTRAP_SEED)
    groups = {source: np.flatnonzero(source_ids == source) for source in source_order}
    cluster_values = []
    for _ in range(cluster_draws):
        chosen = rng_cluster.choice(source_order, size=len(source_order), replace=True)
        indices = np.concatenate([groups[source] for source in chosen])
        if len(np.unique(labels[indices])) == 2:
            cluster_values.append(float(roc_auc_score(labels[indices], probabilities[indices])))
    def result(values, requested):
        return {"requested_draws": requested, "valid_draws": len(values),
                "percentile_95_ci": np.percentile(values, [2.5, 97.5]).tolist() if values else None}
    return {"seed": BOOTSTRAP_SEED,
            "patch_iid_descriptive": result(patch_values, patch_draws),
            "source_wsi_cluster": result(cluster_values, cluster_draws),
            "cluster_count": len(source_order),
            "interpretation": "Patch AUC uncertainty; 4 source clusters. Per-WSI patch AUC is not animal-level classification AUC."}


def write_predictions(path, rows, probabilities, threshold=None):
    fields = list(rows[0]) + ["manifest_row_index", "label_index", "y_lesion", "prob_lesion"]
    if threshold is not None:
        fields.append("pred_lesion_at_validation_threshold")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, (row, probability) in enumerate(zip(rows, probabilities, strict=True)):
            output = dict(row, manifest_row_index=index, label_index=row["label"],
                          y_lesion=int(row["label"] == 0), prob_lesion=float(probability))
            if threshold is not None:
                output["pred_lesion_at_validation_threshold"] = int(probability >= threshold)
            writer.writerow(output)


def self_test():
    # Checks prevent the earlier historical label-index/positive-label reversal.
    class_indices = np.array([0, 0, 1, 1])
    y = (class_indices == 0).astype(int)
    p = np.array([0.9, 0.8, 0.2, 0.1])
    selected = select_validation_threshold(y, p)
    assert selected["threshold"] == 0.8 and selected["validation_auc_lesion"] == 1.0
    result = metrics(y, p, selected["threshold"])
    assert result["TP"] == 2 and result["TN"] == 2 and result["FP"] == result["FN"] == 0
    failed = select_validation_threshold(y, 1 - p)
    assert failed["threshold"] is None and failed["validation_auc_lesion"] == 0.0
    sources = np.array(["a", "b", "a", "b"])
    bootstrap = bootstrap_auc(y, p, sources, ["a", "b"], patch_draws=25, cluster_draws=25)
    assert bootstrap["source_wsi_cluster"]["percentile_95_ci"] == [1.0, 1.0]
    assert bootstrap["patch_iid_descriptive"]["percentile_95_ci"] == [1.0, 1.0]
    print("Synthetic metric, positive-class, validation threshold and cluster-bootstrap checks passed; no model evaluated.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--train-manifest", type=Path)
    parser.add_argument("--val-manifest", type=Path)
    parser.add_argument("--test-manifest", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not all((args.train_manifest, args.val_manifest, args.test_manifest)):
        parser.error("--train-manifest, --val-manifest and --test-manifest are required")
    paths = {"train": args.train_manifest, "val": args.val_manifest, "test": args.test_manifest}
    data, audit = audit_manifests(paths)
    if args.preflight_only:
        print(json.dumps({"manifest_audit": audit, "cross_split_source_path_pixel_overlap": 0}, ensure_ascii=False, indent=2))
        return
    if not args.checkpoint or not args.out_dir:
        parser.error("--checkpoint and --out-dir are required for evaluation")
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("Invalid batch size or worker count")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise ValueError("Evaluation output directory must be new or empty")
    checkpoint_hash = sha256(args.checkpoint)
    # This trainer's local checkpoint includes Python/NumPy RNG state.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = validate_checkpoint_identity(checkpoint)
    if config["seed"] != 42:
        raise ValueError("A6 candidate requires seed42")
    expected = config["manifests"]["manifests"]
    for split in ("train", "val"):
        if expected[split]["sha256"] != audit[split]["sha256"]:
            raise ValueError(f"{split} manifest changed since training")
    gpu_guard()
    name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = torch.device("cpu" if name == "auto" else name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")
    architecture = dict(config["architecture"])
    model = create_model(pretrained=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    protocol = {"checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": checkpoint_hash,
                "checkpoint_config_sha256": checkpoint["config_sha256"], "epoch": checkpoint["epoch"],
                "candidate": "TRAIN_BALANCE_CANDIDATE_V6", "initialization": config["initialization"],
                "positive_class": "Lesion", "model_class_index": 0, "metric_positive_label": 1,
                "tta": False, "transform": repr(build_transforms(False, architecture["img_size"])),
                "device": str(device), "cuda_autocast_dtype": "float16" if device.type == "cuda" else None,
                "softmax": "original logits dtype", "batch_size": args.batch_size,
                "manifests": audit, "cross_split_source_path_pixel_overlap": 0,
                "python": platform.python_version(), "torch": str(torch.__version__),
                "evaluator_sha256": sha256(__file__),
                "historical_export_turn": "e0816dbc-ff58-46fc-92bf-5fc3546e8431",
                "historical_clean4_statistics_turn": "45c653b9-bfd8-4c97-a571-a0feb71a4186"}
    save_json(args.out_dir / "evaluation_protocol.json", protocol)
    val_prob = predict(model, data["val"], device, args.batch_size, args.num_workers, architecture["img_size"])
    val_y = np.array([int(row["label"] == 0) for row in data["val"]])
    locked = select_validation_threshold(val_y, val_prob)
    locked.update(checkpoint_sha256=checkpoint_hash, validation_manifest_sha256=audit["val"]["sha256"])
    write_predictions(args.out_dir / "val_predictions.csv", data["val"], val_prob)
    save_json(args.out_dir / "validation_threshold.json", locked)
    print(json.dumps(locked), flush=True)
    # Test prediction begins only after the new validation threshold is written.
    test_prob = predict(model, data["test"], device, args.batch_size, args.num_workers, architecture["img_size"])
    test_y = np.array([int(row["label"] == 0) for row in data["test"]])
    write_predictions(args.out_dir / "test_predictions.csv", data["test"], test_prob, locked["threshold"])
    source_ids = np.array([row["source_id"] for row in data["test"]])
    names = {row["source_id"]: row.get("legacy_wsi", row["source_id"]) for row in data["test"]}
    source_order = sorted(names, key=lambda source: names[source])
    per_source = []
    for source in source_order:
        mask = source_ids == source
        per_source.append(dict(source_id=source, legacy_wsi=names[source],
                               **metrics(test_y[mask], test_prob[mask], locked["threshold"])))
    with (args.out_dir / "test_per_source_metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_source[0]))
        writer.writeheader()
        writer.writerows(per_source)
    aucs = [row["auc_lesion"] for row in per_source if row["auc_lesion"] is not None]
    summary = {"protocol": protocol, "validation_threshold": locked,
               "test": metrics(test_y, test_prob, locked["threshold"]),
               "bootstrap": bootstrap_auc(test_y, test_prob, source_ids, source_order),
               "per_source_patch_auc_macro_mean": float(np.mean(aucs)) if aucs else None,
               "per_source_patch_auc_macro_sd": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else None,
               "per_source_patch_auc_range": [min(aucs), max(aucs)] if aucs else None}
    save_json(args.out_dir / "evaluation_summary.json", summary)
    print(json.dumps({"output": str(args.out_dir), "test": summary["test"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
