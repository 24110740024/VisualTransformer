"""Aggregate nine frozen A5 outputs without reading biological outcome data."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import itertools
import math
from pathlib import Path

from predict_candidate import (SEEDS, FOLDS, PROTOCOL, EXPECTED_CASES, coordinate_key,
                              load_frozen_coordinates, json_read, atomic_json,
                              sha256_file, canonical_json)


def validate_prediction_rows(rows, expected_rows, seed, fold, checkpoint_hash, config_hash):
    """Require exact frozen keys, including source identity and crop dimensions."""
    expected = {coordinate_key(row) for row in expected_rows}
    if len(expected) != len(expected_rows):
        raise ValueError("Expected coordinate keys are not unique")
    seen, probabilities = set(), defaultdict(list)
    for row in rows:
        key = coordinate_key(row)
        if key not in expected or key in seen:
            raise ValueError(f"Unexpected or duplicate prediction key: {key}")
        if (int(row["seed"]) != seed or int(row["outer_fold"]) != fold
                or row["checkpoint_sha256"] != checkpoint_hash
                or row["inference_config_sha256"] != config_hash):
            raise ValueError("Prediction seed/fold/checkpoint/configuration mismatch")
        value = float(row["prob_lesion"])
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Invalid prediction probability")
        probabilities[row["case_id"]].append(value)
        seen.add(key)
    if seen != expected:
        raise ValueError(f"Missing {len(expected - seen)} frozen tile predictions")
    return {case: {"n_tiles": len(values), "mean_probability": math.fsum(values) / len(values)}
            for case, values in probabilities.items()}


def equal_weight_seed_scores(seed_case_metrics, cases, case_counts):
    expected = set(itertools.product(SEEDS, EXPECTED_CASES))
    if set(seed_case_metrics) != expected:
        raise ValueError("Require exactly 21 cases for each of seeds42/43/44")
    rows = []
    for case in sorted(EXPECTED_CASES):
        for seed in SEEDS:
            if seed_case_metrics[(seed, case)]["n_tiles"] != case_counts[case]:
                raise ValueError("Per-seed case tile counts differ from the frozen manifest")
        means = [seed_case_metrics[(seed, case)]["mean_probability"] for seed in SEEDS]
        rows.append({"case_id": case, "outer_fold": int(case[-1]),
                     "source_sha256": cases[case]["source_sha256"],
                     "n_tiles": case_counts[case], "seed_count": len(SEEDS),
                     **{f"seed{seed}_mean_probability": mean for seed, mean in zip(SEEDS, means, strict=True)},
                     "oof_prob_mean_raw": math.fsum(means) / len(SEEDS)})
    return rows


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-dir", type=Path, required=True)
    parser.add_argument("--runs", nargs=9, type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError("Refusing to overwrite aggregate results")
    coordinates, cases, frozen = load_frozen_coordinates(args.dense_dir)
    by_fold = {fold: [r for r in coordinates if r["outer_fold"] == fold] for fold in FOLDS}
    seen_runs, seed_case_metrics, provenance = set(), {}, []
    for directory in args.runs:
        directory = directory.resolve(strict=True)
        complete = json_read(directory / "COMPLETE.json")
        run = json_read(directory / "run_config.json")
        config = run["configuration"]
        config_hash = hashlib.sha256(canonical_json(config)).hexdigest()
        if (complete.get("status") != "complete" or complete.get("protocol") != PROTOCOL
                or config.get("protocol") != PROTOCOL
                or run["config_sha256"] != config_hash or complete["config_sha256"] != config_hash):
            raise ValueError(f"Unfinished or inconsistent inference artifact: {directory}")
        seed, fold = complete["seed"], complete["fold"]
        if (seed, fold) not in itertools.product(SEEDS, FOLDS) or (seed, fold) in seen_runs:
            raise ValueError("Require exactly one run for each seed/fold pair")
        if config["seed"] != seed or config["fold"] != fold:
            raise ValueError("Run configuration seed/fold mismatch")
        if (config.get("tta") != "flips4_mean_softmax" or config.get("temperature") != 1.0
                or config.get("post_manifest_filter") is not False or config.get("positive_class_index") != 0):
            raise ValueError("Unexpected dense inference method")
        for key in ("manifest_sha256", "frozen_sha256"):
            if complete[key] != frozen[key] or config["frozen"][key] != frozen[key]:
                raise ValueError("The nine runs must share the same frozen coordinates and QC acceptance")
        checkpoint_hash = complete["checkpoint_sha256"]
        if config["checkpoint_sha256"] != checkpoint_hash:
            raise ValueError("Checkpoint identity mismatch")
        predictions = directory / "tile_predictions.csv"
        if sha256_file(predictions) != complete["predictions_sha256"]:
            raise ValueError("Prediction file hash changed")
        with predictions.open(encoding="utf-8-sig", newline="") as stream:
            metrics = validate_prediction_rows(csv.DictReader(stream), by_fold[fold], seed, fold,
                                                checkpoint_hash, config_hash)
        expected_counts = {case: frozen["case_counts"][case] for case in EXPECTED_CASES if case.endswith(f"-{fold}")}
        if complete["rows"] != sum(expected_counts.values()) or complete["case_counts"] != expected_counts:
            raise ValueError("Inference COMPLETE counts differ from frozen coordinates")
        for case, value in metrics.items():
            seed_case_metrics[(seed, case)] = value
        seen_runs.add((seed, fold))
        provenance.append({"seed": seed, "fold": fold, "directory": str(directory),
                           "checkpoint_sha256": checkpoint_hash, "inference_config_sha256": config_hash,
                           "predictions_sha256": complete["predictions_sha256"],
                           "complete_sha256": sha256_file(directory / "COMPLETE.json")})
    if seen_runs != set(itertools.product(SEEDS, FOLDS)):
        raise ValueError("Missing seed/fold runs")
    final = equal_weight_seed_scores(seed_case_metrics, cases, frozen["case_counts"])
    per_seed = [{"seed": seed, "case_id": case, "outer_fold": int(case[-1]),
                 "source_sha256": cases[case]["source_sha256"], **values}
                for (seed, case), values in sorted(seed_case_metrics.items())]
    args.out_dir.mkdir(parents=True)
    write_csv(args.out_dir / "case_scores_per_seed.csv", per_seed)
    write_csv(args.out_dir / "case_scores_equal_seed_mean.csv", final)
    atomic_json(args.out_dir / "COMPLETE.json", {
        "status": "complete", "protocol": "A5_DENSE_EQUAL_SEED_AGGREGATION_BALANCE_V6",
        "case_count": 21, "seeds": list(SEEDS), "folds": list(FOLDS), "frozen": frozen,
        "aggregation": "Unweighted arithmetic mean of all frozen tile probabilities per case and seed; then equal-weight mean across seeds42/43/44",
        "outcomes_read": False, "calibration": False, "winsorization": False, "topk": False,
        "input_runs": sorted(provenance, key=lambda r: (r["seed"], r["fold"])),
        "outputs_sha256": {name: sha256_file(args.out_dir / name) for name in
                            ("case_scores_per_seed.csv", "case_scores_equal_seed_mean.csv")},
        "aggregator_sha256": sha256_file(__file__)})
    print(f"COMPLETE: 63 seed-case rows, 21 equal-seed scores -> {args.out_dir}")


if __name__ == "__main__":
    main()
