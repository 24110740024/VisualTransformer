"""V6 CPU verification: same independent arithmetic as the v2 audit.

Only the two producer script hash paths differ. Does not import training/GPU code.
"""
from collections import defaultdict
from pathlib import Path
import csv, hashlib, itertools, json, math
import numpy as np
from scipy import stats
HERE = Path(__file__).resolve().parent.parent / "balance_candidate_v6"
STAT_DIR = None  # Explicitly bound by completed-state guards before use.
SEEDS = (42, 43, 44)
FOLDS = (1, 2, 3)

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def near(a, b, context, tolerance=1e-12):
    require(math.isclose(float(a), float(b), rel_tol=0, abs_tol=tolerance), context)


def verify_inputs():
    """Recheck hashes and recalculate the full tile -> seed -> animal chain."""
    inputs = {}

    def checked(path, expected=None):
        path = Path(path)
        digest = sha(path)
        if expected is not None:
            require(digest == expected, f"Hash mismatch: {path}")
        inputs[str(path)] = digest

    marker = read_json(STAT_DIR / "COMPLETE.json")
    checked(STAT_DIR / "COMPLETE.json")
    require(marker["status"] == "complete" and marker["primary_n"] == 21, "Incomplete primary analysis")
    for name, digest in marker["outputs_sha256"].items():
        checked(STAT_DIR / name, digest)
    checked(HERE / "analyze_candidate.py", marker["analysis_script_sha256"])
    aggregate = Path(marker["aggregate_directory"])
    outcomes_dir = Path(marker["outcomes_directory"])
    checked(aggregate / "COMPLETE.json", marker["aggregate_complete_sha256"])
    checked(outcomes_dir / "OUTCOMES_FROZEN.json", marker["outcomes_frozen_sha256"])
    for directory, key in ((aggregate, "aggregate_outputs_sha256"), (outcomes_dir, "outcome_outputs_sha256")):
        for name, digest in marker[key].items():
            checked(directory / name, digest)
    agg_marker = read_json(aggregate / "COMPLETE.json")
    checked(HERE / "aggregate_candidate.py", agg_marker["aggregator_sha256"])
    require(agg_marker["status"] == "complete", "Incomplete aggregate")
    require(all(agg_marker[k] is False for k in ("outcomes_read", "calibration", "winsorization", "topk")), "Unexpected aggregation operation")
    frozen = agg_marker["frozen"]
    dense_dir = Path(frozen["directory"])
    for name, key in (("coordinates_all21.csv", "manifest_sha256"), ("COMPLETE.json", "complete_sha256"), ("FROZEN.json", "frozen_sha256")):
        checked(dense_dir / name, frozen[key])
    dense_marker = read_json(dense_dir / "FROZEN.json")
    require(dense_marker["status"] == "accepted_for_inference", "Coordinates not accepted")
    require(dense_marker["outcomes_used"] is False and dense_marker["model_outputs_used"] is False, "Outcome-dependent coordinate freeze")
    coordinates = read_csv(dense_dir / "coordinates_all21.csv")
    require(len(coordinates) == frozen["total_coordinates"] == 132032, "Coordinate count mismatch")
    coordinate_fields = tuple(coordinates[0])
    expected_coords = {fold: set() for fold in FOLDS}
    for row in coordinates:
        expected_coords[int(row["outer_fold"])].add(tuple(row[k] for k in coordinate_fields))
    require(sum(map(len, expected_coords.values())) == len(coordinates), "Duplicate frozen coordinates")
    runs = agg_marker["input_runs"]
    require(len(runs) == 9 and {(r["seed"], r["fold"]) for r in runs} == set(itertools.product(SEEDS, FOLDS)), "Incomplete nine-model set")
    tile_means = {}
    total_predictions = 0
    for run in runs:
        directory = Path(run["directory"])
        checked(directory / "COMPLETE.json", run["complete_sha256"])
        checked(directory / "tile_predictions.csv", run["predictions_sha256"])
        done = read_json(directory / "COMPLETE.json")
        require(done["status"] == "complete" and done["seed"] == run["seed"] and done["fold"] == run["fold"], "Incorrect inference run")
        require(done["manifest_sha256"] == frozen["manifest_sha256"] and done["frozen_sha256"] == frozen["frozen_sha256"], "Inference freeze mismatch")
        rows = read_csv(directory / "tile_predictions.csv")
        seen = set()
        probabilities = defaultdict(list)
        for row in rows:
            key = tuple(row[k] for k in coordinate_fields)
            require(key not in seen, "Duplicate predicted coordinate")
            seen.add(key)
            require(int(row["seed"]) == run["seed"], "Seed identity mismatch")
            require(row["checkpoint_sha256"] == run["checkpoint_sha256"], "Checkpoint identity mismatch")
            require(row["inference_config_sha256"] == run["inference_config_sha256"], "Inference config mismatch")
            probability = float(row["prob_lesion"])
            require(math.isfinite(probability) and 0 <= probability <= 1, "Invalid probability")
            probabilities[row["case_id"]].append(probability)
        require(seen == expected_coords[run["fold"]], "Predictions differ from exact frozen coordinates")
        require(len(rows) == done["rows"], "Inference row count mismatch")
        total_predictions += len(rows)
        for case, values in probabilities.items():
            require(len(values) == frozen["case_counts"][case], "Animal tile count mismatch")
            tile_means[(run["seed"], case)] = math.fsum(values) / len(values)
    require(total_predictions == 396096, "Three-seed prediction total mismatch")
    per_seed = read_csv(aggregate / "case_scores_per_seed.csv")
    cases = marker["spec"]["primary_case_ids"]
    require(len(cases) == len(set(cases)) == 21, "Case specification invalid")
    expected_keys = set(itertools.product(SEEDS, cases))
    require(len(per_seed) == 63 and {(int(r["seed"]), r["case_id"]) for r in per_seed} == expected_keys, "Incomplete per-seed scores")
    require(set(tile_means) == expected_keys, "Incomplete tile means")
    for row in per_seed:
        near(tile_means[(int(row["seed"]), row["case_id"])], row["mean_probability"], "Tile average mismatch", 1e-14)
    final = {r["case_id"]: r for r in read_csv(aggregate / "case_scores_equal_seed_mean.csv")}
    outcomes = {r["case_id"]: r for r in read_csv(outcomes_dir / "outcomes21.csv")}
    joined = read_csv(STAT_DIR / "joined21.csv")
    require(len(joined) == 21 and [r["case_id"] for r in joined] == sorted(cases), "Joined case set/order mismatch")
    require(set(final) == set(outcomes) == set(cases), "Outcome/score case mismatch")
    require(len({r["source_sha256"] for r in joined}) == 21, "Nonunique animal source WSI")
    for row in joined:
        case = row["case_id"]
        require(row["source_sha256"] == final[case]["source_sha256"] == outcomes[case]["source_sha256"], "Source mismatch")
        require(int(row["outer_fold"]) == int(case[-1]) and int(row["n_tiles"]) == frozen["case_counts"][case], "Joined fold/count mismatch")
        require(int(final[case]["seed_count"]) == 3, "Not all three seeds")
        means = [tile_means[(seed, case)] for seed in SEEDS]
        for seed, mean in zip(SEEDS, means, strict=True):
            near(mean, final[case][f"seed{seed}_mean_probability"], "Embedded seed score mismatch", 1e-14)
        score = math.fsum(means) / 3
        near(score, final[case]["oof_prob_mean_raw"], "Seed equal-weight mean mismatch", 1e-14)
        near(score, row["pathology_score"], "Joined score mismatch", 1e-14)
        for key in ("Area", "IL6"):
            near(row[key], outcomes[case][key], "Joined outcome mismatch", 1e-14)
    return marker, joined, tile_means, inputs


def verify_statistics(marker, joined):
    results = read_json(STAT_DIR / "association_results.json")
    require(len(results) == 8, "Require all four primary and four secondary statistics")
    exclusions = set(marker["spec"]["optional_secondary_n18"]["exclude_exact_cases"])
    require(exclusions == {f"30-30-120-N-{i}" for i in FOLDS}, "Unexpected secondary exclusion")
    expected = set(itertools.product(("primary21", "secondary18_fixed_alias_history_exclusions"), ("Area", "IL6"), ("Pearson", "Spearman")))
    require({(r["analysis"], r["outcome"], r["statistic"]) for r in results} == expected, "Missing or duplicate statistics")
    verified = []
    for record in results:
        primary = record["analysis"] == "primary21"
        rows = joined if primary else [r for r in joined if r["case_id"] not in exclusions]
        x = np.array([float(r["pathology_score"]) for r in rows])
        y = np.array([float(r[record["outcome"]]) for r in rows])
        function = stats.pearsonr if record["statistic"] == "Pearson" else stats.spearmanr
        coefficient, p_value = function(x, y)
        near(coefficient, record["coefficient"], "Association coefficient mismatch")
        near(p_value, record["p_two_sided_nominal"], "Association p mismatch")
        require(len(rows) == record["n_animals"] and record["multiple_testing_adjusted"] is False, "Statistical metadata mismatch")
        audit = {"analysis": record["analysis"], "outcome": record["outcome"], "statistic": record["statistic"], "coefficient": float(coefficient), "p": float(p_value)}
        if primary:
            saved = record["paired_animal_percentile95_CI"]
            require(saved["draws"] == 10000 and saved["seed"] == 20260920, "Unexpected bootstrap spec")
            # Independent vectorized implementation uses the same fixed integer draw stream.
            indices = np.random.default_rng(20260920).integers(0, len(rows), size=(10000, len(rows)))
            boot_x, boot_y = x[indices], y[indices]
            if record["statistic"] == "Spearman":
                boot_x = stats.rankdata(boot_x, axis=1)
                boot_y = stats.rankdata(boot_y, axis=1)
            boot_r = stats.pearsonr(boot_x, boot_y, axis=1).statistic
            finite = boot_r[np.isfinite(boot_r)]
            lo, hi = np.percentile(finite, [2.5, 97.5])
            near(lo, saved["low"], "Bootstrap lower CI mismatch")
            near(hi, saved["high"], "Bootstrap upper CI mismatch")
            require(len(finite) == saved["finite_draws"] == 10000, "Bootstrap valid draw mismatch")
            audit["ci_recomputed"] = [float(lo), float(hi)]
        verified.append(audit)
    return results, verified


