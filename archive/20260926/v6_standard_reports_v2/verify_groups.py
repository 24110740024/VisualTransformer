from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import itertools
from figures import read,rows,sha,close

def verify_group(GROUP):
    spec = read(GROUP / 'ANALYSIS_SPEC.json')
    assert sha(Path(spec['input'])) == spec['input_sha256']
    df = pd.read_csv(GROUP / 'animals_with_groups.csv', float_precision='round_trip')
    orig = pd.read_csv(Path(spec['input']), float_precision='round_trip')
    assert np.array_equal(df.pathology_score, orig.pathology_score)
    assert list(df.case_id) == list(orig.case_id) and len(df) == 21
    wt = df[df.group.isin(['G3', 'G4', 'G6', 'G7'])]
    a = (wt.temperature.to_numpy() - 33) / 3
    b = (wt.ozone.to_numpy() - 100) / 20
    X = np.column_stack([np.ones(12), a, b, a * b])
    y = wt.pathology_score.to_numpy()
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    mse = np.sum((y - X @ coef) ** 2) / 8
    expected = coef[1:] ** 2 / (mse * np.diag(np.linalg.inv(X.T @ X))[1:])
    got = pd.read_csv(GROUP / 'WT_factorial_ANOVA.csv', float_precision='round_trip')
    assert np.allclose(got.F, expected, rtol=0, atol=1e-12)
    assert np.allclose(got.p, stats.f.sf(expected, 1, 8), rtol=0, atol=1e-12)
    assert np.allclose(got.partial_eta2, expected / (expected + 8), rtol=0, atol=1e-12)
    contrasts = rows(GROUP / 'planned_contrasts.csv')
    assert len(contrasts) == 4
    for r in contrasts:
        x, z = [df[df.group == g].pathology_score.to_numpy() for g in (r['group_A'], r['group_B'])]
        pool = np.r_[x, z]
        observed = abs(z.mean() - x.mean())
        vals = []
        for ix in itertools.combinations(range(6), 3):
            mask = np.zeros(6, dtype=bool)
            mask[list(ix)] = True
            vals.append(abs(pool[mask].mean() - pool[~mask].mean()))
        close(r['exact_permutation_p_nominal'], np.mean(np.array(vals) >= observed - 1e-12))
        close(r['difference_B_minus_A'], z.mean() - x.mean())
        vx, vz = (np.var(x, ddof=1), np.var(z, ddof=1))
        t = (x.mean() - z.mean()) / np.sqrt(vx / 3 + vz / 3)
        dof = (vx / 3 + vz / 3) ** 2 / ((vx / 3) ** 2 / 2 + (vz / 3) ** 2 / 2)
        close(r['Welch_t'], t)
        close(r['Welch_p_nominal'], 2 * stats.t.sf(abs(t), dof))
        close(r['Hedges_g_B_minus_A'], 0.8 * (z.mean() - x.mean()) / np.sqrt((vx + vz) / 2))
    return {'all_real_group_checks_passed': True, 'animals': 21, 'anova_independent_OLS': True, 'four_contrasts_all20_assignments': True, 'Welch_closed_form_and_Hedges_checked': True, 'outputs_sha256': {p.name: sha(p) for p in GROUP.iterdir() if p.is_file()}}
