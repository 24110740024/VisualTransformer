"""Freeze the audited 21 biological outcomes before new A5 associations.

This command never reads model scores. Source tables and original WSI are read
only; a new directory holds the exact-case outcomes, identities and analysis spec.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

import pandas as pd

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent.parent
GROUPS = {"30-40-80-N", "36-40-80-N", "36-40-80-A", "30-30-120-N",
          "36-30-120-N", "36-30-120-A", "36-20-160-A"}
EXPECTED_CASES = {f"{group}-{animal}" for group in GROUPS for animal in (1, 2, 3)}
EXCLUDED_N18 = [f"30-30-120-N-{animal}" for animal in (1, 2, 3)]
FIELDS = ["case_id", "outer_fold", "source_sha256", "Area", "IL6"]
PROTOCOL = "A5_OUTCOMES_FULL_PRECISION_V1"
STATISTICS_SPEC = {
    "protocol": "A5_FINAL_DIRECT_ASSOCIATIONS_V1",
    "primary_case_ids": sorted(EXPECTED_CASES),
    "row_order": "case_id ascending Unicode order",
    "analysis_unit": "animal; one source WSI and one fixed score per animal",
    "score_column": "oof_prob_mean_raw",
    "score_definition": "all frozen tile probability means per animal and seed, then equal-weight mean of seeds42/43/44",
    "outcomes": ["Area", "IL6"],
    "outcome_transforms": "none",
    "statistics": ["Pearson", "Spearman"],
    "p_values": "SciPy default two-sided, nominal; Spearman asymptotic approximation",
    "multiple_testing_correction": "none (four primary nominal p-values)",
    "group_adjustment": "none",
    "permutation": False,
    "bootstrap": {"unit": "paired animal", "stratified": False, "draws": 10000,
                  "seed": 20260920, "reset_rng_each_statistic_outcome_call": True,
                  "interval": "np.percentile finite coefficients at [2.5,97.5]",
                  "nonfinite_replicates": "discard and report; fail if fewer than 9500 valid",
                  "retrain_or_resample_models_tiles_seeds": False},
    "optional_secondary_n18": {"exclude_exact_cases": EXCLUDED_N18,
                               "statistics": ["Pearson", "Spearman"],
                               "bootstrap": False, "replaces_primary": False},
    "model_selection_using_outcomes": False,
    "historical_evidence": "history retrievalPage21 turn2284753b-f488-444a-806e-547a95955b9f (supplied final ensemble script); see recovered_protocol/A5_biological_statistics.md",
    "new_run_differences": ["IL6 fixed to raw full-precision workbook values rather than four rounded entries in the older keyed table",
                            "sorted case_id fixes bootstrap row order",
                            "explicit failure if fewer than9500 finite bootstrap coefficients"],
}
UNITS = {
    "Area": {"source_field": "area_mm2", "verified_unit": None,
             "uncertainty": "Source field suggests mm2, but original measurement definition/unit has not been independently verified. No percent conversion or relabeling performed."},
    "IL6": {"source_field": "raw IL-6.xlsx cells; checked against il6_il6_value",
            "verified_unit": None, "uncertainty": "Concentration unit not independently documented; do not assume pg/mL."},
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def unique_keyed(rows, field, expected=None):
    keyed = {row[field]: row for row in rows}
    if len(keyed) != len(rows) or not all(keyed):
        raise ValueError(f"Duplicate or empty {field}")
    if expected is not None and set(keyed) != set(expected):
        raise ValueError(f"Exact key set mismatch for {field}")
    return keyed


def valid_hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def checked_file(path, expected_hash):
    if not valid_hash(expected_hash) or sha256_file(path) != expected_hash:
        raise ValueError(f"Source/artifact SHA256 mismatch: {path}")


def source_identities(copy_audit, coverage):
    audit, covered = read_json(copy_audit), read_json(coverage)
    if audit.get("summary", {}).get("accepted") is not True:
        raise ValueError("WSI copy audit was not accepted")
    normalize = lambda path: str(path).replace("/", "\\").casefold()
    copies = unique_keyed([{**r, "normalized_source": normalize(r["source"])} for r in audit["files"]], "normalized_source")
    cases = unique_keyed(covered["cases"], "case_id", EXPECTED_CASES)
    identities = {}
    for case, meta in cases.items():
        if meta["matching_qupath_project_images"] != 1:
            raise ValueError(f"Ambiguous QuPath source identity: {case}")
        copied = copies[normalize(meta["mac_source_tif"])]
        digest = copied["source_sha256"]
        if not valid_hash(digest) or copied["passed"] is not True or copied["destination_sha256"] != digest:
            raise ValueError(f"Unverified original WSI identity: {case}")
        identities[case] = digest
    if len(set(identities.values())) != 21:
        raise ValueError("Expected21 distinct source hashes")
    return identities


def collect_outcomes(table_audit, copy_audit, coverage):
    audit = read_json(table_audit)
    if (audit.get("sample_count") != 21 or audit.get("exact_sample_key_match") is not True
            or audit.get("raw_and_clean_il6_all_match") is not True):
        raise ValueError("Outcome table audit not accepted")
    names = ("IL-6.xlsx", "3_il6_clean.csv", "evidence_1IL6Area_clean.csv")
    records = {}
    for name in names:
        matches = [r for r in audit["sources"] if Path(r["path"]).name == name]
        if len(matches) != 1:
            raise ValueError(f"Ambiguous/missing outcome source: {name}")
        checked_file(matches[0]["path"], matches[0]["sha256"])
        records[name] = matches[0]
    raw = pd.read_excel(records[names[0]]["path"], index_col=0)
    if set(raw.columns) != GROUPS or list(raw.index) != ["mice_one", "mice_two", "mice_three"] or len(raw.columns) != 7:
        raise ValueError("Raw IL6 workbook case layout changed")
    clean = unique_keyed(read_csv(records[names[1]]["path"]), "CaseID", EXPECTED_CASES)
    area = unique_keyed(read_csv(records[names[2]]["path"]), "casekey")
    if not EXPECTED_CASES.issubset(area) or len(area) != audit["area_source_records"]:
        raise ValueError("Area exact-case source missing/changed")
    audited = unique_keyed(audit["samples"], "case_id", EXPECTED_CASES)
    identities = source_identities(copy_audit, coverage)
    rows = []
    for case in sorted(EXPECTED_CASES):
        group, animal = case.rsplit("-", 1)
        il6 = float(raw.loc[["mice_one", "mice_two", "mice_three"][int(animal)-1], group])
        il6_clean = float(clean[case]["il6_il6_value"])
        measured_area = float(area[case]["area_mm2"])
        if not all(math.isfinite(v) for v in (il6, il6_clean, measured_area)):
            raise ValueError(f"Missing/nonfinite biological outcome: {case}")
        if abs(il6 - il6_clean) >= 1e-6:
            raise ValueError(f"Raw/clean IL6 mismatch: {case}")
        if il6 != audited[case]["raw_il6"] or measured_area != audited[case]["area_mm2"]:
            raise ValueError(f"Recorded audit outcome mismatch: {case}")
        rows.append(dict(case_id=case, outer_fold=int(animal), source_sha256=identities[case], Area=measured_area, IL6=il6))
    return rows, list(records.values())


def freeze_outcomes(table_audit, copy_audit, coverage, out_dir):
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError("Refusing to overwrite frozen biological outcomes")
    rows, sources = collect_outcomes(table_audit, copy_audit, coverage)
    out_dir.mkdir(parents=True)
    write_csv(out_dir / "outcomes21.csv", rows)
    write_json(out_dir / "STATISTICS_SPEC.json", STATISTICS_SPEC)
    marker = {
        "status": "frozen_before_new_associations", "protocol": PROTOCOL,
        "created_utc": datetime.now(timezone.utc).isoformat(), "case_count": 21,
        "case_ids": sorted(EXPECTED_CASES), "source_sha256_by_case": {r["case_id"]: r["source_sha256"] for r in rows},
        "model_scores_read": False, "model_selection": False,
        "IL6_precision": "raw full-precision workbook floating-point values; clean CSV independent match tolerance1e-6",
        "Area_selection": "only original area_mm2 selected by explicit unique casekey; no old-model columns read",
        "units": UNITS,
        "limitation": "Recorded key/hash/numerical consistency does not independently establish animal labeling or laboratory provenance.",
        "sources": sources,
        "audit_inputs": {name: {"path": str(Path(path).resolve()), "sha256": sha256_file(path)}
                         for name, path in (("table_pairing", table_audit), ("verified_copy", copy_audit), ("final21_coverage", coverage))},
        "outputs_sha256": {name: sha256_file(out_dir / name) for name in ("outcomes21.csv", "STATISTICS_SPEC.json")},
        "preparer_sha256": sha256_file(__file__),
    }
    write_json(out_dir / "OUTCOMES_FROZEN.json", marker)
    return marker


def load_frozen_outcomes(directory):
    directory = Path(directory)
    marker = read_json(directory / "OUTCOMES_FROZEN.json")
    if (marker.get("status") != "frozen_before_new_associations" or marker.get("protocol") != PROTOCOL
            or marker.get("model_scores_read") is not False or marker.get("model_selection") is not False
            or marker.get("case_count") != 21 or marker.get("case_ids") != sorted(EXPECTED_CASES)
            or marker.get("units") != UNITS):
        raise ValueError("Invalid biological outcome freeze")
    for name in ("outcomes21.csv", "STATISTICS_SPEC.json"):
        checked_file(directory / name, marker["outputs_sha256"][name])
    if read_json(directory / "STATISTICS_SPEC.json") != STATISTICS_SPEC:
        raise ValueError("Statistical specification changed")
    for entry in marker["audit_inputs"].values():
        checked_file(entry["path"], entry["sha256"])
    inputs = marker["audit_inputs"]
    expected, sources = collect_outcomes(inputs["table_pairing"]["path"], inputs["verified_copy"]["path"], inputs["final21_coverage"]["path"])
    if marker["sources"] != sources:
        raise ValueError("Frozen source identities changed")
    rows = read_csv(directory / "outcomes21.csv")
    unique_keyed(rows, "case_id", EXPECTED_CASES)
    normalized = [{"case_id": r["case_id"], "outer_fold": int(r["outer_fold"]), "source_sha256": r["source_sha256"],
                   "Area": float(r["Area"]), "IL6": float(r["IL6"])} for r in rows]
    if normalized != expected or marker["source_sha256_by_case"] != {r["case_id"]: r["source_sha256"] for r in expected}:
        raise ValueError("Outcome contents/source identities differ from frozen original tables")
    return normalized, marker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-audit", type=Path, default=WORKSPACE / "recovery_audit/table_pairing_audit.json")
    parser.add_argument("--copy-audit", type=Path, default=WORKSPACE / "recovery_audit/verified_mac_wsi_copy_acceptance.json")
    parser.add_argument("--coverage", type=Path, default=WORKSPACE / "recovery_audit/final21_wsi_coverage.json")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    freeze_outcomes(args.table_audit, args.copy_audit, args.coverage, args.out_dir)
    print(f"FROZEN:21 biological outcomes; no model scores read -> {args.out_dir}")


if __name__ == "__main__":
    main()
