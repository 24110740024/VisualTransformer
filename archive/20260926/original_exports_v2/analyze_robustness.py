"""Fixed supplementary analyses; never alter the main scores or animal cohort."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import argparse
import csv
import hashlib
import json
import math
import sys
import warnings
import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from analyze_a5_associations import validate_aggregate
from prepare_outcomes import load_frozen_outcomes, checked_file, read_json, read_csv, write_csv, write_json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def holm(p):
    p = np.asarray(p, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Holm adjustment requires finite p-values in [0,1]')
    order = np.argsort(p, kind='stable')
    answer = np.empty_like(p)
    answer[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return answer


def coefficient(x, y, method):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', stats.ConstantInputWarning)
        warnings.simplefilter('ignore', stats.NearConstantInputWarning)
        r, p = (stats.pearsonr if method == 'Pearson' else stats.spearmanr)(x, y)
    return (float(r), float(p)) if math.isfinite(r) and math.isfinite(p) else (None, None)


def center_by_group(values, groups):
    values, groups = np.asarray(values, dtype=float), np.asarray(groups)
    answer = values.copy()
    for group in sorted(set(groups)):
        mask = groups == group
        answer[mask] -= values[mask].mean()
    return answer


def compute(rows):
    ids = np.array([r['case_id'] for r in rows])
    groups = np.array([i.rsplit('-', 1)[0] for i in ids])
    assert len(rows) == len(set(ids)) == 21
    assert len(set(groups)) == 7 and set(Counter(groups).values()) == {3}
    x = np.asarray([r['pathology_score'] for r in rows], dtype=float)
    main, influence, within = [], [], []
    for outcome in ('Area', 'IL6'):
        y = np.asarray([r[outcome] for r in rows], dtype=float)
        assert np.isfinite(x).all() and np.isfinite(y).all()
        for method in ('Pearson', 'Spearman'):
            r, p = coefficient(x, y, method)
            if r is None:
                raise ValueError('Primary association undefined; investigate without selecting alternative inputs')
            main.append({'outcome': outcome, 'statistic': method, 'n_animals': 21,
                         'coefficient': r, 'p_nominal': p})
            for mode, excluded_values in [('leave_one_animal_out', ids), ('leave_one_group_out', sorted(set(groups)))]:
                for excluded in excluded_values:
                    keep = (ids if mode == 'leave_one_animal_out' else groups) != excluded
                    rr, pp = coefficient(x[keep], y[keep], method)
                    influence.append({'outcome': outcome, 'statistic': method, 'analysis': mode,
                                      'excluded': str(excluded), 'n_animals': int(keep.sum()),
                                      'coefficient': rr, 'delta_from_full21': None if rr is None else rr-r,
                                      'p_nominal_descriptive': pp, 'status': 'undefined' if rr is None else 'ok'})
        # Descriptive group-centered coefficient only: ordinary pearsonr p would
        # ignore the degrees of freedom spent estimating seven group means.
        xx, yy = center_by_group(x, groups), center_by_group(y, groups)
        rr, _ = coefficient(xx, yy, 'Pearson')
        within.append({'outcome': outcome, 'statistic': 'Pearson_of_group_centered_values',
                       'n_animals': 21, 'n_groups': 7, 'animals_per_group': 3,
                       'coefficient': rr, 'inference': 'descriptive_only_no_p_or_CI',
                       'status': 'undefined' if rr is None else 'ok'})
    for row, adjusted in zip(main, holm([r['p_nominal'] for r in main]), strict=True):
        row['p_holm_family4'] = float(adjusted)
    return main, influence, within


def self_test():
    # Known Holm example, reordering invariance, and a synthetic Simpson reversal.
    p = np.array([0.01, 0.04, 0.03, 0.002])
    assert np.allclose(holm(p), [0.03, 0.06, 0.06, 0.008])
    perm = np.array([3, 1, 0, 2])
    assert np.allclose(holm(p[perm]), holm(p)[perm])
    rows = []
    for g in range(7):
        for i, residual in enumerate([-1, 0, 1], 1):
            rows.append({'case_id': f'{20+g}-40-80-N-{i}',
                         'pathology_score': (10*g+residual+1)/70,
                         'Area': 10*g-residual, 'IL6': 10*g-residual})
    main, influence, within = compute(rows)
    assert len(main) == 4 and len(influence) == 112 and len(within) == 2
    assert all(r['coefficient'] > 0.9 for r in main)
    assert all(np.isclose(r['coefficient'], -1) for r in within)
    assert all(r['n_animals'] == (20 if r['analysis'] == 'leave_one_animal_out' else 18) for r in influence)
    assert all(r['inference'] == 'descriptive_only_no_p_or_CI' for r in within)
    assert coefficient([1, 1, 1], [2, 3, 4], 'Pearson') == (None, None)
    print('PASS: Holm reference/order, Simpson reversal, all 112 deletion analyses, constant-input handling. No study outcomes read.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    state = read_json(HERE / 'jobs/pipeline_status.json')
    if state['phase'] != 'core_complete':
        raise RuntimeError('Wait for this v2 pipeline to complete; never read unfinished or v1 results')
    for path, digest in read_json(HERE / 'jobs/execution_protocol.json')['dependency_sha256'].items():
        checked_file(path, digest)
    aggregate = Path(state['aggregate']).resolve(strict=True)
    statistics = Path(state['statistics']).resolve(strict=True)
    assert aggregate.is_relative_to(HERE) and statistics.is_relative_to(HERE)
    outcomes, _ = load_frozen_outcomes(HERE.parent / 'outcomes_v1')
    rows, marker = validate_aggregate(aggregate, outcomes)
    assert all(Path(r['directory']).resolve().is_relative_to(HERE) for r in marker['input_runs'])
    primary_done = read_json(statistics / 'COMPLETE.json')
    assert primary_done['status'] == 'complete' and primary_done['primary_n'] == 21
    assert primary_done['aggregate_complete_sha256'] == sha(aggregate / 'COMPLETE.json')
    for name, digest in primary_done['outputs_sha256'].items():
        checked_file(statistics / name, digest)
    primary = [r for r in read_json(statistics / 'association_results.json') if r['analysis'] == 'primary21']
    main_rows, influence, within = compute(rows)
    for new in main_rows:
        old = next(r for r in primary if (r['outcome'], r['statistic']) == (new['outcome'], new['statistic']))
        assert math.isclose(old['coefficient'], new['coefficient'], rel_tol=0, abs_tol=1e-12)
        assert math.isclose(old['p_two_sided_nominal'], new['p_nominal'], rel_tol=0, abs_tol=1e-12)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = HERE / 'robustness' / stamp
    out.mkdir(parents=True, exist_ok=False)
    for name, table in [('holm_primary4.csv', main_rows), ('influence_all112.csv', influence), ('within_group_descriptive.csv', within)]:
        write_csv(out / name, table)
    write_json(out / 'COMPLETE.json', {
        'status': 'complete', 'protocol': 'V2_QUALITY_ADDENDUM_ROBUSTNESS',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'primary_n_unchanged': 21,
        'primary_scores_unchanged': True, 'model_or_animal_selection': False,
        'script_sha256': sha(__file__), 'addendum_sha256': sha(HERE / 'QUALITY_ADDENDUM.md'),
        'main_statistics_directory': str(statistics),
        'main_statistics_complete_sha256': sha(statistics / 'COMPLETE.json'),
        'outputs_sha256': {p.name: sha(p) for p in sorted(out.glob('*.csv'))},
        'limitations': ['Post hoc supplementary analysis; prior results known.',
                        'Group-centered coefficient is descriptive with only three animals per group.',
                        'All deletion analyses retained; none select the main animal cohort.',
                        'Four primary tests adjusted jointly; deletion diagnostic p-values are not confirmatory.']})
    print(f'Complete fixed supplementary audit: {out}')


if __name__ == '__main__':
    main()
