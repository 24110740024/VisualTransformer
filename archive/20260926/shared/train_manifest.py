"""Train a newly initialized ViT using audited train/validation manifests only.

CSV fields: path, label (0=Lesion, 1=Normal), source_id, split.
Relative image paths are resolved against the CSV directory. When supplied,
pixel_sha256 or pixel_hash must be populated for every row of both manifests.
Repeated paths represent separate logical records only when every row has a
unique historical_file identifier; their multiplicity and label conflicts are reported.
Resume accepts this trainer's last.pth only; it is not pretrained initialization.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, WeightedRandomSampler
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataset import build_transforms
from model import ViT

FORMAT_VERSION = 1
CLASSES = ["Lesion", "Normal"]
HISTORY_FIELDS = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc",
                  "val_auc_lesion", "lr_used", "seconds", "improved"]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path, expected_split):
    path = Path(path).resolve(strict=True)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"path", "label", "source_id", "split"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: required columns path,label,source_id,split")
        hash_fields = [key for key in ("pixel_sha256", "pixel_hash")
                       if key in reader.fieldnames]
        rows = []
        seen_paths = set()
        duplicated_paths = set()
        for line, row in enumerate(reader, 2):
            if row["label"] not in ("0", "1"):
                raise ValueError(f"{path}:{line}: label must be 0 or 1")
            if row["split"] != expected_split:
                raise ValueError(f"{path}:{line}: expected split={expected_split}; test is not accepted")
            if not row["source_id"].strip() or not row["path"].strip():
                raise ValueError(f"{path}:{line}: empty source_id or image path")
            image_path = Path(row["path"])
            if not image_path.is_absolute():
                image_path = path.parent / image_path
            image_path = image_path.resolve(strict=True)
            if not image_path.is_file():
                raise ValueError(f"{path}:{line}: image is not a file: {image_path}")
            path_key = os.path.normcase(str(image_path))
            if path_key in seen_paths:
                duplicated_paths.add(path_key)
            seen_paths.add(path_key)
            values = [row[key].strip().lower() for key in hash_fields]
            if hash_fields and (not all(values) or len(set(values)) != 1):
                raise ValueError(f"{path}:{line}: missing or inconsistent pixel hash")
            row.update(path=str(image_path), label=int(row["label"]),
                       source_id=row["source_id"].strip(),
                       _pixel_hash=values[0] if values else None)
            rows.append(row)
    if duplicated_paths:
        identifiers = [row.get("historical_file", "").strip() for row in rows]
        if not all(identifiers) or len(set(identifiers)) != len(identifiers):
            raise ValueError(f"{path}: repeated paths require unique historical_file identifiers on every row")
    if not rows or {row["label"] for row in rows} != {0, 1}:
        raise ValueError(f"{path}: both Lesion and Normal are required")
    return rows


def preflight(train_manifest, val_manifest):
    train = read_manifest(train_manifest, "train")
    val = read_manifest(val_manifest, "val")
    for key in ("source_id", "path"):
        values = lambda rows: {os.path.normcase(row[key]) if key == "path" else row[key] for row in rows}
        overlap = values(train) & values(val)
        if overlap:
            raise ValueError(f"train/val {key} overlap: {sorted(overlap)[:8]}")
    hash_presence = [row["_pixel_hash"] is not None for row in train + val]
    if any(hash_presence) and not all(hash_presence):
        raise ValueError("Pixel hashes must be present in both manifests and every row")
    if all(hash_presence):
        overlap = {row["_pixel_hash"] for row in train} & {row["_pixel_hash"] for row in val}
        if overlap:
            raise ValueError(f"train/val pixel hash overlap: {sorted(overlap)[:8]}")
    summary = {
        "classes": CLASSES, "positive_class_index": 0,
        "source_overlap": 0, "pixel_hash_check": "provided_hashes" if all(hash_presence) else "unavailable",
        "manifests": {},
    }
    for split, path, rows in (("train", train_manifest, train), ("val", val_manifest, val)):
        path_labels = {}
        for row in rows:
            path_labels.setdefault(os.path.normcase(row["path"]), set()).add(row["label"])
        summary["manifests"][split] = {
            "path": str(Path(path).resolve()), "sha256": sha256(path), "rows": len(rows),
            "source_ids": sorted({row["source_id"] for row in rows}),
            "class_counts": {CLASSES[label]: sum(row["label"] == label for row in rows)
                             for label in (0, 1)},
            "repeated_path_rows": len(rows) - len(path_labels),
            "paths_used_with_both_labels": sum(len(labels) > 1 for labels in path_labels.values()),
        }
    return train, val, summary


class ManifestDataset(Dataset):
    def __init__(self, rows, train):
        self.rows = rows
        self.transform = build_transforms(train=train, img_size=224)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, row["label"]


def seed_worker(_worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def rng_state(generators):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "generators": {key: gen.get_state() for key, gen in generators.items()}}


def restore_rng(state, generators):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if not torch.cuda.is_available():
            raise ValueError("CUDA checkpoint requires CUDA for same-run resume")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    for key, gen in generators.items():
        gen.set_state(state["generators"][key].cpu())


def atomic_torch_save(value, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_json(value, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def write_history(history, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(history)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("train_manifest", "val_manifest", "out_dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--weighted_sampler", action="store_true")
    parser.add_argument("--class_weight", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--resume", type=Path, help="Only this run's last.pth, never an old/pretrained checkpoint")
    parser.add_argument("--preflight_only", action="store_true", help="Validate manifests without constructing a model")
    args = parser.parse_args(argv)
    if min(args.epochs, args.bs, args.patience) <= 0 or args.num_workers < 0:
        parser.error("epochs, bs, patience must be positive and num_workers nonnegative")
    if not np.isfinite(args.lr) or args.lr <= 0 or not np.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("lr must be positive and weight_decay nonnegative, both finite")
    if not 0 <= args.seed < 2**32:
        parser.error("seed must be in [0, 2**32)")
    return args


def main(argv=None):
    args = parse_args(argv)
    train_rows, val_rows, audit = preflight(args.train_manifest, args.val_manifest)
    if args.preflight_only:
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        return
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    device = torch.device(device_name)
    out_dir = args.out_dir.resolve()
    configuration = {key: str(value.resolve()) if isinstance(value, Path) else value
                     for key, value in vars(args).items() if key not in ("resume", "preflight_only")}
    configuration.update(device_resolved=device_name, initialization="random", model="ViT",
                         selection="validation Lesion-positive AUC, strict improvement",
                         architecture={"img_size": 224, "patch_size": 16, "embed_dim": 768,
                                       "depth": 12, "heads": 12, "mlp_dim": 3072,
                                       "dropout": 0.1, "pos_dropout": 0.0},
                         manifests=audit,
                         code_sha256={str(path.relative_to(ROOT)): sha256(path)
                                      for path in (Path(__file__), ROOT / "model.py", ROOT / "dataset.py")},
                         runtime={"python": platform.python_version(), "torch": torch.__version__,
                                  "numpy": np.__version__, "cuda": torch.version.cuda})
    config_hash = hashlib.sha256(json.dumps(configuration, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    checkpoint = None
    if args.resume:
        if args.resume.resolve() != out_dir / "last.pth":
            raise ValueError("--resume must point to this out_dir/last.pth")
        # Explicit local checkpoint contains Python/NumPy RNG tuples in addition to tensors.
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint.get("format_version") != FORMAT_VERSION or checkpoint.get("config_sha256") != config_hash:
            raise ValueError("Resume configuration, manifest bytes, code, or runtime mismatch")
    elif out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError("Fresh training requires an empty output directory; use --resume to continue")
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_json({"config_sha256": config_hash, "configuration": configuration}, out_dir / "config.json")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    generators = {key: torch.Generator().manual_seed(args.seed + offset)
                  for offset, key in enumerate(("sampling", "train_workers", "val_workers"))}
    train_set, val_set = ManifestDataset(train_rows, True), ManifestDataset(val_rows, False)
    counts = np.bincount([row["label"] for row in train_rows], minlength=2)
    if args.weighted_sampler:
        weights = [1.0 / counts[row["label"]] for row in train_rows]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True, generator=generators["sampling"])
    else:
        sampler = RandomSampler(train_set, generator=generators["sampling"])
    loader_options = dict(batch_size=args.bs, num_workers=args.num_workers,
                          pin_memory=device.type == "cuda", worker_init_fn=seed_worker, persistent_workers=False)
    train_loader = DataLoader(train_set, sampler=sampler, generator=generators["train_workers"], **loader_options)
    val_loader = DataLoader(val_set, shuffle=False, generator=generators["val_workers"], **loader_options)
    model = ViT(num_classes=2).to(device)
    loss_weight = torch.tensor(counts.sum() / counts, dtype=torch.float32, device=device) if args.class_weight else None
    criterion = nn.CrossEntropyLoss(weight=loss_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    start_epoch, best_auc, best_epoch, no_improve, history = 1, -1.0, -1, 0, []
    if checkpoint:
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"] + 1
        best_auc, best_epoch = checkpoint["best_auc"], checkpoint["best_epoch"]
        no_improve, history = checkpoint["no_improve"], checkpoint["history"]
        restore_rng(checkpoint["rng"], generators)
        write_history(history, out_dir / "history.csv")
        if no_improve >= args.patience:
            print("This run already stopped for patience; no additional training performed.")
            return

    print(json.dumps({"event": "training_started", "device": str(device),
                      "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
                      "seed": args.seed, "train_records": len(train_set), "val_records": len(val_set),
                      "batches_per_epoch": len(train_loader), "start_epoch": start_epoch,
                      "config_sha256": config_hash}), flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        started, lr_used = time.perf_counter(), optimizer.param_groups[0]["lr"]
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for batch_index, (image, label) in enumerate(train_loader, 1):
            image, label = image.to(device, non_blocking=True), label.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(image)
                loss = criterion(logits, label)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite training loss at epoch {epoch}; last completed epoch remains resumable")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * label.size(0)
            train_correct += (logits.argmax(1) == label).sum().item()
            train_total += label.size(0)
            if batch_index == 1 or batch_index % 100 == 0 or batch_index == len(train_loader):
                progress = {"event": "batch_progress", "phase": "train", "epoch": epoch,
                            "batch": batch_index, "batches": len(train_loader),
                            "elapsed_seconds": time.perf_counter()-started,
                            "train_loss_so_far": train_loss/train_total}
                atomic_json(progress, out_dir / "live.json")
                print(json.dumps(progress), flush=True)
        scheduler.step()
        model.eval()
        val_loss, val_correct, val_total, probabilities, targets = 0.0, 0, 0, [], []
        with torch.no_grad():
            for image, label in val_loader:
                image, label = image.to(device, non_blocking=True), label.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    logits = model(image)
                    loss = criterion(logits, label)
                val_loss += loss.item() * label.size(0)
                val_correct += (logits.argmax(1) == label).sum().item()
                val_total += label.size(0)
                probabilities.extend(logits.softmax(1)[:, 0].cpu().tolist())
                targets.extend((label == 0).long().cpu().tolist())
        val_auc = float(roc_auc_score(targets, probabilities))
        if not np.isfinite(val_auc) or not np.isfinite(val_loss):
            raise RuntimeError("Nonfinite validation metrics; last completed epoch remains resumable")
        improved = val_auc > best_auc
        if improved:
            best_auc, best_epoch, no_improve = val_auc, epoch, 0
        else:
            no_improve += 1
        history.append(dict(epoch=epoch, train_loss=train_loss/train_total, train_acc=train_correct/train_total,
                            val_loss=val_loss/val_total, val_acc=val_correct/val_total, val_auc_lesion=val_auc,
                            lr_used=lr_used, seconds=time.perf_counter()-started, improved=improved))
        state = {"format_version": FORMAT_VERSION, "config_sha256": config_hash,
                 "configuration": configuration, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch,
                 "best_auc": best_auc, "best_epoch": best_epoch, "no_improve": no_improve,
                 "history": history, "rng": rng_state(generators), "classes": CLASSES,
                 "positive_class_index": 0, "initialization": "random"}
        if improved:
            atomic_torch_save(state, out_dir / "best.pth")
        atomic_torch_save(state, out_dir / "last.pth")
        write_history(history, out_dir / "history.csv")
        atomic_json({"best_epoch": best_epoch, "best_val_auc_lesion": best_auc,
                     "completed_epoch": epoch, "no_improve": no_improve,
                     "config_sha256": config_hash}, out_dir / "status.json")
        print(json.dumps(history[-1]), flush=True)
        if no_improve >= args.patience:
            print(f"Early stopping after epoch {epoch}; best validation AUC epoch {best_epoch}", flush=True)
            break


if __name__ == "__main__":
    main()
