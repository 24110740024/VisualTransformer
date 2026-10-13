"""Run the recovered final A5 statistics on a verified, complete new aggregate.

Fixed primary21 raw-outcome associations; optional, explicitly secondary n18.
No model/seed/endpoint selection, transformations, group adjustment or plots.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import itertools
import math
from pathlib import Path
import platform
import warnings

import numpy as np
import scipy
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prepare_outcomes import (EXPECTED_CASES, EXCLUDED_N18, STATISTICS_SPEC,
                              checked_file, load_frozen_outcomes, read_csv, read_json,
                              sha256_file, unique_keyed, write_csv, write_json)

SEEDS, FOLDS = (42, 43, 44), (1, 2, 3)
AGGREGATE_PROTOCOL = "A5_DENSE_EQUAL_SEED_AGGREGATION_BALANCE_V6"


def close_float(a, b):
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-15)


def probability(value):
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError("Nonfinite/out-of-range aggregate probability")
    return number


def validate_score_tables(final_rows, per_seed_rows, outcomes, case_counts):
    """Require exact21 case/source keys and independently recompute seed means."""
    expected = unique_keyed(outcomes, "case_id", EXPECTED_CASES)
    final = unique_keyed(final_rows, "case_id", EXPECTED_CASES)
    if set(case_counts) != EXPECTED_CASES or any(type(n) is not int or n <= 0 for n in case_counts.values()):
        raise ValueError("Frozen dense case counts invalid")
    per_seed = {}
    for row in per_seed_rows:
        key = (int(row["seed"]), row["case_id"])
        if key in per_seed or key not in set(itertools.product(SEEDS, EXPECTED_CASES)):
            raise ValueError("Duplicate or unexpected per-seed animal")
        case = key[1]
        if (row["source_sha256"] != expected[case]["source_sha256"]
                or int(row["outer_fold"]) != int(case[-1])
                or int(row["n_tiles"]) != case_counts[case]):
            raise ValueError(f"Per-seed source/fold/count mismatch: {case}")
        per_seed[key] = probability(row["mean_probability"])
    if set(per_seed) != set(itertools.product(SEEDS, EXPECTED_CASES)):
        raise ValueError("Incomplete per-seed table: require all63 rows")
    joined = []
    for case in sorted(EXPECTED_CASES):
        row, outcome = final[case], expected[case]
        if (row["source_sha256"] != outcome["source_sha256"] or int(row["outer_fold"]) != int(case[-1])
                or int(row["seed_count"]) != 3 or int(row["n_tiles"]) != case_counts[case]):
            raise ValueError(f"Final source/fold/count mismatch: {case}")
        means = [per_seed[(seed, case)] for seed in SEEDS]
        if any(not close_float(probability(row[f"seed{seed}_mean_probability"]), mean)
               for seed, mean in zip(SEEDS, means, strict=True)):
            raise ValueError("Final per-seed means differ from63-row table")
        score = probability(row["oof_prob_mean_raw"])
        if not close_float(score, math.fsum(means) / 3):
            raise ValueError("Final score is not the equal-weight three-seed mean")
        joined.append({**outcome, "n_tiles": case_counts[case], "pathology_score": score})
    return joined


def validate_aggregate(directory, outcomes):
    directory = Path(directory).resolve(strict=True)
    marker = read_json(directory / "COMPLETE.json")
    if (marker.get("status") != "complete" or marker.get("protocol") != AGGREGATE_PROTOCOL
            or marker.get("case_count") != 21 or marker.get("seeds") != list(SEEDS)
            or marker.get("folds") != list(FOLDS)
            or any(marker.get(k) is not False for k in ("outcomes_read", "calibration", "winsorization", "topk"))):
        raise ValueError("Require completed, unmodified outcome-independent A5 aggregate")
    for name in ("case_scores_per_seed.csv", "case_scores_equal_seed_mean.csv"):
        checked_file(directory / name, marker["outputs_sha256"][name])
    checked_file(Path(__file__).with_name("aggregate_candidate.py"), marker["aggregator_sha256"])
    frozen = marker["frozen"]
    dense = Path(frozen["directory"])
    for filename, hashkey in (("coordinates_all21.csv", "manifest_sha256"),
                              ("COMPLETE.json", "complete_sha256"), ("FROZEN.json", "frozen_sha256")):
        checked_file(dense / filename, frozen[hashkey])
    dense_done, dense_qc = read_json(dense / "COMPLETE.json"), read_json(dense / "FROZEN.json")
    if (dense_qc.get("status") != "accepted_for_inference"
            or dense_qc.get("outcomes_used") is not False or dense_qc.get("model_outputs_used") is not False
            or dense_qc.get("manifest_sha256") != frozen["manifest_sha256"]
            or dense_qc.get("complete_sha256") != frozen["complete_sha256"]):
        raise ValueError("Dense coordinates were not independently frozen and accepted")
    cases = unique_keyed(dense_done["cases"], "case_id", EXPECTED_CASES)
    for outcome in outcomes:
        case = outcome["case_id"]
        if (cases[case]["source_sha256"] != outcome["source_sha256"]
                or cases[case]["coordinate_count"] != frozen["case_counts"][case]):
            raise ValueError(f"Dense source identity/count differs from locked outcomes: {case}")
    if sum(frozen["case_counts"].values()) != frozen["total_coordinates"]:
        raise ValueError("Frozen total coordinate count mismatch")
    runs = marker["input_runs"]
    if len(runs) != 9 or {(r["seed"], r["fold"]) for r in runs} != set(itertools.product(SEEDS, FOLDS)):
        raise ValueError("Aggregate must include exactly all nine prespecified runs")
    for record in runs:
        run_dir = Path(record["directory"])
        checked_file(run_dir / "COMPLETE.json", record["complete_sha256"])
        run_done = read_json(run_dir / "COMPLETE.json")
        if (run_done.get("status") != "complete" or run_done.get("protocol") != "A5_DENSE_INFERENCE_BALANCE_V6"
                or (run_done["seed"], run_done["fold"]) != (record["seed"], record["fold"])):
            raise ValueError("Input inference incomplete or seed/fold mismatch")
        for key in ("checkpoint_sha256", "predictions_sha256"):
            if run_done[key] != record[key]:
                raise ValueError("Input inference identity mismatch")
        if run_done["config_sha256"] != record["inference_config_sha256"]:
            raise ValueError("Input inference configuration mismatch")
        for key in ("manifest_sha256", "frozen_sha256"):
            if run_done[key] != frozen[key]:
                raise ValueError("Input inference used another dense manifest")
        checked_file(run_dir / "tile_predictions.csv", record["predictions_sha256"])
    joined = validate_score_tables(read_csv(directory / "case_scores_equal_seed_mean.csv"),
                                   read_csv(directory / "case_scores_per_seed.csv"), outcomes, frozen["case_counts"])
    return joined, marker


def paired_bootstrap_ci(x, y, statistic, draws=10000, seed=20260920, minimum_valid=None):
    """Exact recovered sampling sequence: rng.integers(0,n,n) in each loop.

    Reset the generator per call. Nonfinite constant-input samples are skipped,
    matching the old estimator; their count is reported and inadequate runs fail.
    Parameters are exposed for meaningful synthetic tests, not CLI tuning.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 1 or x.shape != y.shape or len(x) < 3 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Bootstrap needs paired finite one-dimensional samples")
    if draws < 1:
        raise ValueError("Bootstrap draws must be positive")
    required = math.ceil(0.95 * draws) if minimum_valid is None else minimum_valid
    rng = np.random.default_rng(seed)
    coefficients = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", stats.ConstantInputWarning)
        warnings.simplefilter("ignore", stats.NearConstantInputWarning)
        for _ in range(draws):
            index = rng.integers(0, len(x), len(x))
            coefficient = float(statistic(x[index], y[index]))
            if math.isfinite(coefficient):
                coefficients.append(coefficient)
    if len(coefficients) < required:
        raise ValueError(f"Insufficient finite bootstrap coefficients: {len(coefficients)}/{draws}")
    low, high = np.percentile(coefficients, [2.5, 97.5])
    return {"low": float(low), "high": float(high), "draws": draws, "finite_draws": len(coefficients),
            "discarded_nonfinite_draws": draws-len(coefficients), "seed": seed}


def association_results(rows, label, bootstrap):
    x = np.asarray([r["pathology_score"] for r in rows], dtype=float)
    results = []
    for outcome in ("Area", "IL6"):
        y = np.asarray([r[outcome] for r in rows], dtype=float)
        for name, function in (("Pearson", stats.pearsonr), ("Spearman", stats.spearmanr)):
            point, p = function(x, y)
            if not math.isfinite(float(point)) or not math.isfinite(float(p)):
                raise ValueError(f"Undefined {name} {outcome} association")
            result = {"analysis": label, "n_animals": len(rows), "outcome": outcome, "statistic": name,
                      "coefficient": float(point), "p_two_sided_nominal": float(p), "multiple_testing_adjusted": False}
            if bootstrap:
                result["paired_animal_percentile95_CI"] = paired_bootstrap_ci(x, y, lambda a, b: function(a, b)[0])
            results.append(result)
    return results


def analyze(aggregate_dir, outcomes_dir, out_dir, include_n18=False):
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError("Refusing to overwrite association results")
    outcomes, outcome_marker = load_frozen_outcomes(outcomes_dir)
    joined, aggregate_marker = validate_aggregate(aggregate_dir, outcomes)
    results = association_results(joined, "primary21", bootstrap=True)
    if include_n18:
        secondary = [r for r in joined if r["case_id"] not in EXCLUDED_N18]
        if len(secondary) != 18:
            raise ValueError("Fixed n18 sensitivity cohort mismatch")
        results += association_results(secondary, "secondary18_fixed_alias_history_exclusions", bootstrap=False)
    out_dir.mkdir(parents=True)
    write_csv(out_dir / "joined21.csv", joined)
    write_json(out_dir / "association_results.json", results)
    write_json(out_dir / "COMPLETE.json", {
        "status": "complete", "protocol": STATISTICS_SPEC["protocol"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "primary_n": 21, "optional_secondary_n18_requested": include_n18,
        "spec": STATISTICS_SPEC, "units": outcome_marker["units"],
        "aggregate_directory": str(Path(aggregate_dir).resolve()),
        "aggregate_complete_sha256": sha256_file(Path(aggregate_dir) / "COMPLETE.json"),
        "outcomes_directory": str(Path(outcomes_dir).resolve()),
        "outcomes_frozen_sha256": sha256_file(Path(outcomes_dir) / "OUTCOMES_FROZEN.json"),
        "aggregate_outputs_sha256": aggregate_marker["outputs_sha256"],
        "outcome_outputs_sha256": outcome_marker["outputs_sha256"],
        "outputs_sha256": {name: sha256_file(out_dir / name) for name in ("joined21.csv", "association_results.json")},
        "analysis_script_sha256": sha256_file(__file__),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "limitations": ["Unadjusted association may reflect between-exposure-group gradients.",
                        "Four nominal primary p-values; no multiplicity adjustment.",
                        "Bootstrap uncertainty conditions on fitted scores; no model/training resampling.",
                        "Area and IL6 physical units remain unverified."]})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-dir", type=Path, required=True)
    parser.add_argument("--outcomes-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--include-n18", action="store_true", help="Also report prespecified secondary18; never replace primary21")
    args = parser.parse_args()
    analyze(args.aggregate_dir, args.outcomes_dir, args.out_dir, args.include_n18)
    print(f"COMPLETE: fixed primary21 biological associations -> {args.out_dir}")


if __name__ == "__main__":
    main()
