"""Predict only a new A5 checkpoint's seven held-out animals on frozen tiles.

No WSI materialization, outcome access, tile filtering, calibration, or score
selection. TIFF JPEG tiles are decoded on demand with bounded per-worker caches.
"""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

import cv2
import numpy as np
from PIL import Image
import PIL
import tifffile
import torch
from torch.utils.data import Dataset, DataLoader

CANDIDATE = Path(__file__).resolve().parent
HERE = CANDIDATE.parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from candidate_model import create_model, validate_checkpoint_identity
from candidate_guard import gpu_guard
from build_dense_manifest import (
    CONFIG as DENSE_CONFIG, FIELDS as COORD_FIELDS, EXPECTED_CASES,
    canonical_json, decode_jpeg_tile, normalized_path, sha256_file,
)
from train_manifest import preflight as training_preflight

PROTOCOL = "A5_DENSE_INFERENCE_BALANCE_V6"
SEEDS = (42, 43, 44)
FOLDS = (1, 2, 3)
KEY_FIELDS = ("case_id", "source_sha256", "x", "y", "crop_size", "level")
INTEGER_FIELDS = ("outer_fold", "x", "y", "crop_size", "level", "source_width",
                  "source_height", "pad_right", "pad_bottom")


def json_read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def atomic_json(path, value):
    temporary = Path(str(path) + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def coordinate_key(row):
    return (str(row["case_id"]), str(row["source_sha256"]), int(row["x"]),
            int(row["y"]), int(row["crop_size"]), int(row["level"]))


def _load_original_coordinates(directory, fold=None):
    """Validate the full 21-case freeze before selecting a fold, without images."""
    directory = Path(directory).resolve(strict=True)
    manifest_path = directory / "coordinates_all21.csv"
    complete_path = directory / "COMPLETE.json"
    frozen_path = directory / "FROZEN.json"
    complete, frozen = json_read(complete_path), json_read(frozen_path)
    configuration = json_read(directory / "config.json")
    expected_config_hash = hashlib.sha256(canonical_json(DENSE_CONFIG)).hexdigest()
    if configuration != {"configuration": DENSE_CONFIG, "config_sha256": expected_config_hash}:
        raise ValueError("Dense configuration differs from this prespecified new-run protocol")
    if complete.get("status") != "complete_coordinate_generation_awaiting_visual_qc":
        raise ValueError("Coordinate generation was not completed")
    if (frozen.get("status") != "accepted_for_inference"
            or frozen.get("outcomes_used") is not False
            or frozen.get("model_outputs_used") is not False):
        raise ValueError("Explicit outcome-independent FROZEN.json QC acceptance is required")
    manifest_hash = sha256_file(manifest_path)
    complete_hash = sha256_file(complete_path)
    if (frozen.get("manifest_sha256") != manifest_hash
            or complete.get("manifest_sha256") != manifest_hash
            or frozen.get("complete_sha256") != complete_hash):
        raise ValueError("Frozen manifest or COMPLETE.json hash mismatch")
    if complete.get("config_sha256") != expected_config_hash or frozen.get("config_sha256") != expected_config_hash:
        raise ValueError("Frozen dense configuration hash mismatch")
    if complete.get("generator_sha256") != sha256_file(HERE / "build_dense_manifest.py"):
        raise ValueError("Dense generator changed since coordinate generation")
    if complete.get("case_count") != 21 or complete.get("historical_tile_membership_claimed") is not False:
        raise ValueError("Expected a new-run 21-animal dense set, not historical tile-membership claims")
    cases = {entry["case_id"]: entry for entry in complete["cases"]}
    if len(complete["cases"]) != 21 or set(cases) != EXPECTED_CASES:
        raise ValueError("COMPLETE.json must contain exactly the prespecified 21 animals")
    if len({entry["source_sha256"] for entry in cases.values()}) != 21:
        raise ValueError("Expected one unique source WSI hash per animal")
    counts, keys, selected = Counter(), set(), []
    with manifest_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != COORD_FIELDS:
            raise ValueError("Unexpected dense coordinate schema")
        for row in reader:
            for field in INTEGER_FIELDS:
                row[field] = int(row[field])
            case = row["case_id"]
            if case not in cases:
                raise ValueError(f"Unexpected coordinate case: {case}")
            meta = cases[case]
            if (row["source_sha256"] != meta["source_sha256"]
                    or normalized_path(row["source_path"]) != normalized_path(meta["source_path"])
                    or row["config_sha256"] != expected_config_hash
                    or row["outer_fold"] != int(case[-1])
                    or row["outer_fold"] != int(meta["outer_fold"])):
                raise ValueError(f"Coordinate source/fold/configuration mismatch for {case}")
            w, h, x, y, size = (row[k] for k in ("source_width", "source_height", "x", "y", "crop_size"))
            if (w != int(meta["source_width"]) or h != int(meta["source_height"])
                    or not (0 <= x < w and 0 <= y < h) or size != 512 or row["level"] != 0
                    or row["pad_right"] != max(0, x + size - w)
                    or row["pad_bottom"] != max(0, y + size - h)):
                raise ValueError(f"Invalid coordinate, dimensions, level or padding for {case}")
            key = coordinate_key(row)
            if key in keys:
                raise ValueError(f"Duplicate frozen tile key: {key}")
            keys.add(key)
            counts[case] += 1
            if fold is None or row["outer_fold"] == fold:
                selected.append(row)
    if sum(counts.values()) != complete["total_coordinates"] or set(counts) != EXPECTED_CASES:
        raise ValueError("Coordinate count/case coverage disagrees with COMPLETE.json")
    if any(counts[case] != cases[case]["coordinate_count"] for case in EXPECTED_CASES):
        raise ValueError("Per-case frozen coordinate counts disagree")
    if fold is not None and {r["case_id"] for r in selected} != {c for c in EXPECTED_CASES if c.endswith(f"-{fold}")}:
        raise ValueError("Fold does not contain exactly its seven held-out animals")
    identities = {"directory": str(directory), "manifest_sha256": manifest_hash,
                  "complete_sha256": complete_hash, "frozen_sha256": sha256_file(frozen_path),
                  "config_sha256": expected_config_hash,
                  "total_coordinates": sum(counts.values()), "case_counts": dict(sorted(counts.items()))}
    return selected, cases, identities


def load_frozen_coordinates(directory, fold=None):
    from scale_geometry import load_scaled
    original, cases, identities = _load_original_coordinates(HERE / 'dense_v1')
    return load_scaled(directory, original, cases, identities, fold)


class TiledJpegReader:
    """Top-left, 8-bit contiguous JPEG TIFF reader; never allocates full WSI."""
    def __init__(self, path, cache_tiles=32):
        if cache_tiles < 1:
            raise ValueError("cache_tiles must be positive")
        self.path = Path(path)
        self.cache_tiles = cache_tiles
        self.cache = OrderedDict()
        self.stream = None
        with tifffile.TiffFile(self.path) as tif:
            page = tif.pages[0]
            if (not page.is_tiled or int(page.compression) != 7 or page.samplesperpixel != 3
                    or page.bitspersample != 8 or int(page.planarconfig) != 1):
                raise ValueError("Requires tiled JPEG, 8-bit, contiguous RGB level0 TIFF")
            orientation = page.tags.get("Orientation")
            if orientation is not None and int(orientation.value) != 1:
                raise ValueError("Unsupported TIFF orientation")
            self.width, self.height = int(page.imagewidth), int(page.imagelength)
            self.tw, self.th = int(page.tilewidth), int(page.tilelength)
            self.cols, self.rows = math.ceil(self.width / self.tw), math.ceil(self.height / self.th)
            self.offsets, self.counts = tuple(page.dataoffsets), tuple(page.databytecounts)
            self.tables = page.jpegtables
        if len(self.offsets) != self.cols * self.rows or len(self.counts) != self.cols * self.rows:
            raise ValueError("Unexpected TIFF JPEG tile layout")
        self.stream = self.path.open("rb")

    def _tile(self, index):
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        count = self.counts[index]
        if count <= 0:
            raise ValueError(f"Empty JPEG tile {index}: {self.path}")
        self.stream.seek(self.offsets[index])
        compressed = self.stream.read(count)
        if len(compressed) != count:
            raise ValueError(f"Truncated JPEG tile {index}: {self.path}")
        image = decode_jpeg_tile(compressed, (self.tw, self.th), self.tables)
        pixels = np.asarray(image).copy()
        image.close()
        self.cache[index] = pixels
        if len(self.cache) > self.cache_tiles:
            self.cache.popitem(last=False)
        return pixels

    def crop(self, x, y, size=512):
        if not (0 <= x < self.width and 0 <= y < self.height and size > 0):
            raise ValueError("Invalid crop origin or size")
        result = np.full((size, size, 3), 255, dtype=np.uint8)
        right, bottom = min(x + size, self.width), min(y + size, self.height)
        for ty in range(y // self.th, (bottom - 1) // self.th + 1):
            for tx in range(x // self.tw, (right - 1) // self.tw + 1):
                tile = self._tile(ty * self.cols + tx)
                left, top = max(x, tx * self.tw), max(y, ty * self.th)
                end_x, end_y = min(right, (tx + 1) * self.tw), min(bottom, (ty + 1) * self.th)
                result[top - y:end_y - y, left - x:end_x - x] = tile[
                    top - ty * self.th:end_y - ty * self.th,
                    left - tx * self.tw:end_x - tx * self.tw]
        return result

    def close(self):
        if self.stream is not None:
            self.stream.close()
            self.stream = None
        self.cache.clear()

    def __del__(self):
        self.close()


def preprocess_crop(rgb):
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.shape[0] != rgb.shape[1] or rgb.shape[0] not in (408,409):
        raise ValueError("Dense input must be calibrated square RGB uint8 crop of 408/409 pixels")
    pixels = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    pixels = (pixels - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return torch.from_numpy(np.ascontiguousarray(pixels.transpose(2, 0, 1)))


class DenseCropDataset(Dataset):
    def __init__(self, rows, max_open_sources=2, cache_tiles=32):
        self.rows = rows
        self.max_open_sources = max_open_sources
        self.cache_tiles = cache_tiles
        self.readers = OrderedDict()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = row["source_path"]
        if path not in self.readers:
            if len(self.readers) >= self.max_open_sources:
                _, expired = self.readers.popitem(last=False)
                expired.close()
            self.readers[path] = TiledJpegReader(path, self.cache_tiles)
        self.readers.move_to_end(path)
        reader = self.readers[path]
        if (reader.width, reader.height) != (row["source_width"], row["source_height"]):
            raise ValueError(f"WSI dimensions changed: {path}")
        return preprocess_crop(reader.crop(row["x"], row["y"], row["crop_size"])), index

    def __getstate__(self):
        return {**self.__dict__, "readers": OrderedDict()}

    def close(self):
        for reader in self.readers.values():
            reader.close()
        self.readers.clear()

    def __del__(self):
        self.close()


def worker_init(_worker):
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


def flips4_probabilities(model, images, amp=False):
    """Average four softmax probabilities, never average logits or features."""
    probabilities = []
    for axes in (None, (-1,), (-2,), (-2, -1)):
        transformed = images if axes is None else torch.flip(images, dims=axes)
        with torch.autocast(device_type=images.device.type, dtype=torch.float16,
                            enabled=amp and images.device.type == "cuda"):
            logits = model(transformed)
            if logits.ndim != 2 or logits.shape[1] != 2:
                raise ValueError("Expected two-class logits")
            probabilities.append(torch.softmax(logits / 1.0, dim=1))
    return torch.stack(probabilities).mean(0)[:, 0]


def validate_checkpoint(checkpoint, seed, fold, train_manifest, val_manifest, rows):
    config = validate_checkpoint_identity(checkpoint)
    if config.get("seed") != seed:
        raise ValueError("Candidate seed mismatch")
    train, val, audit = training_preflight(train_manifest, val_manifest)
    for split, records in (("train", train), ("val", val)):
        if any(int(r.get("outer_fold", -1)) != fold for r in records):
            raise ValueError(f"{split} manifest belongs to a different outer fold")
        expected = config["manifests"]["manifests"][split]
        if expected["sha256"] != audit["manifests"][split]["sha256"]:
            raise ValueError(f"{split} manifest hash mismatch with checkpoint")
        held_sources = {r["source_sha256"] for r in rows}
        if held_sources & {r["source_id"] for r in records}:
            raise ValueError(f"Held-out source appears in {split}")
    if config["architecture"]["img_size"] != 224:
        raise ValueError("Dense inference expects the recovered 224-pixel model")
    return config, audit


def verify_sources(rows, cases):
    states = {}
    for case in sorted({r["case_id"] for r in rows}):
        meta = cases[case]
        path = Path(meta["source_path"]).resolve(strict=True)
        before = path.stat()
        if before.st_size != int(meta["expected_bytes"]) or sha256_file(path) != meta["source_sha256"]:
            raise ValueError(f"Frozen source file changed: {case}")
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"Source changed during hashing: {case}")
        states[case] = {"path": str(path), "sha256": meta["source_sha256"],
                        "bytes": after.st_size, "mtime_ns": after.st_mtime_ns}
        print(f"Verified source {case}", flush=True)
    return states


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--fold", type=int, choices=FOLDS, required=True)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("Invalid batch size or workers")
    if not args.preflight_only and args.out_dir is None:
        parser.error("--out-dir required for inference")
    if args.out_dir and args.out_dir.exists():
        raise FileExistsError("Refusing to overwrite an inference output directory")
    cv2.setNumThreads(1)
    rows, cases, frozen = load_frozen_coordinates(args.dense_dir, args.fold)
    checkpoint_hash = sha256_file(args.checkpoint)
    # Local trainer checkpoint includes Python/NumPy RNG state, not just tensors.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    training_config, train_audit = validate_checkpoint(checkpoint, args.seed, args.fold,
                                                       args.train_manifest, args.val_manifest, rows)
    source_states = verify_sources(rows, cases)
    if args.preflight_only:
        print(json.dumps({"seed": args.seed, "fold": args.fold, "tiles": len(rows),
                          "heldout_cases": sorted(source_states), "source_overlap": 0,
                          "frozen": frozen, "checkpoint_sha256": checkpoint_hash}, indent=2))
        return
    gpu_guard()
    name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = torch.device("cpu" if name == "auto" else name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")
    configuration = {"protocol": PROTOCOL, "seed": args.seed, "fold": args.fold,
                     "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": checkpoint_hash,
                     "checkpoint_config_sha256": checkpoint["config_sha256"],
                     "best_epoch": checkpoint["best_epoch"], "training_manifests": train_audit,
                     "source_overlap": 0, "frozen": frozen, "sources": source_states,
                     "batch_size": args.batch_size, "num_workers": args.num_workers,
                     "device": str(device), "autocast_float16": device.type == "cuda",
                     "crop_size": "source_calibrated_408_or_409", "scale_binding_sha256": sha256_file(HERE / "scale_candidate_v4/dense_scaled/FROZEN.json"), "resize": "cv2.INTER_AREA RGB 224x224",
                     "normalization": "ImageNet", "tta": "flips4_mean_softmax", "temperature": 1.0,
                     "positive_class": "Lesion", "positive_class_index": 0,
                     "padding_rgb": [255, 255, 255], "post_manifest_filter": False,
                     "reader_max_open_sources_per_worker": 2, "reader_jpeg_cache_tiles_per_source": 32,
                     "python": platform.python_version(), "torch": str(torch.__version__),
                     "numpy": np.__version__, "opencv": cv2.__version__, "Pillow": PIL.__version__,
                     "tifffile": tifffile.__version__,
                     "code_sha256": {"predict_candidate.py": sha256_file(__file__),
                                     "build_dense_manifest.py": sha256_file(HERE / "build_dense_manifest.py"),
                                     "candidate_model.py": sha256_file(CANDIDATE / "candidate_model.py")}}
    config_hash = hashlib.sha256(canonical_json(configuration)).hexdigest()
    args.out_dir.mkdir(parents=True)
    dataset = None
    started = time.monotonic()
    try:
        atomic_json(args.out_dir / "run_config.json", {"config_sha256": config_hash, "configuration": configuration})
        model = create_model(pretrained=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.to(device).eval()
        del checkpoint
        dataset = DenseCropDataset(rows)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=device.type == "cuda",
                            worker_init_fn=worker_init, persistent_workers=False)
        fields = COORD_FIELDS + ["seed", "prob_lesion", "checkpoint_sha256", "inference_config_sha256"]
        partial = args.out_dir / "tile_predictions.csv.partial"
        completed = 0
        with partial.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            with torch.inference_mode():
                for images, indices in loader:
                    indices = indices.numpy().tolist()
                    if indices != list(range(completed, completed + len(indices))):
                        raise ValueError("Missing, duplicate or reordered inference records")
                    probabilities = flips4_probabilities(model, images.to(device, non_blocking=True), amp=device.type == "cuda")
                    probabilities = probabilities.cpu().numpy().astype(np.float64)
                    if not np.isfinite(probabilities).all() or not ((0 <= probabilities) & (probabilities <= 1)).all():
                        raise ValueError("Invalid model probability; no tiles may be silently dropped")
                    for index, probability in zip(indices, probabilities, strict=True):
                        writer.writerow({**rows[index], "seed": args.seed, "prob_lesion": float(probability),
                                         "checkpoint_sha256": checkpoint_hash, "inference_config_sha256": config_hash})
                    completed += len(indices)
                    if completed % (args.batch_size * 20) == 0 or completed == len(rows):
                        stream.flush()
                        print(f"seed{args.seed} fold{args.fold}: {completed}/{len(rows)} tiles", flush=True)
            stream.flush()
            os.fsync(stream.fileno())
        if completed != len(rows):
            raise ValueError("Inference did not cover every frozen held-out tile")
        dataset.close()
        for case, state in source_states.items():
            stat = Path(state["path"]).stat()
            if (stat.st_size, stat.st_mtime_ns) != (state["bytes"], state["mtime_ns"]):
                raise ValueError(f"WSI changed during inference: {case}")
        if sha256_file(args.checkpoint) != checkpoint_hash:
            raise ValueError("Checkpoint changed during inference")
        final_path = args.out_dir / "tile_predictions.csv"
        partial.replace(final_path)
        atomic_json(args.out_dir / "COMPLETE.json", {
            "status": "complete", "protocol": PROTOCOL, "seed": args.seed, "fold": args.fold,
            "rows": completed, "case_counts": dict(Counter(r["case_id"] for r in rows)),
            "config_sha256": config_hash, "checkpoint_sha256": checkpoint_hash,
            "manifest_sha256": frozen["manifest_sha256"], "frozen_sha256": frozen["frozen_sha256"],
            "predictions_sha256": sha256_file(final_path), "seconds": time.monotonic() - started})
        print(f"COMPLETE {final_path}", flush=True)
    except Exception as exc:
        atomic_json(args.out_dir / "FAILED.json", {"status": "failed", "error": repr(exc),
                                                   "seed": args.seed, "fold": args.fold,
                                                   "config_sha256": config_hash})
        raise
    finally:
        if dataset is not None:
            dataset.close()


if __name__ == "__main__":
    main()
