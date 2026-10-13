# Reference syntax-tree fixtures for the reported v6 implementation.
# Parsed by tests; not executable entrypoints.

def paired_bootstrap_ci(x, y, statistic, draws=10000, seed=20260920, minimum_valid=None):
    """Exact recovered sampling sequence: rng.integers(0,n,n) in each loop.

    Reset the generator per call. Nonfinite constant-input samples are skipped,
    matching the old estimator; their count is reported and inadequate runs fail.
    Parameters are exposed for meaningful synthetic tests, not CLI tuning.
    """
    x, y = (np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    if x.ndim != 1 or x.shape != y.shape or len(x) < 3 or (not np.isfinite(x).all()) or (not np.isfinite(y).all()):
        raise ValueError('Bootstrap needs paired finite one-dimensional samples')
    if draws < 1:
        raise ValueError('Bootstrap draws must be positive')
    required = math.ceil(0.95 * draws) if minimum_valid is None else minimum_valid
    rng = np.random.default_rng(seed)
    coefficients = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', stats.ConstantInputWarning)
        warnings.simplefilter('ignore', stats.NearConstantInputWarning)
        for _ in range(draws):
            index = rng.integers(0, len(x), len(x))
            coefficient = float(statistic(x[index], y[index]))
            if math.isfinite(coefficient):
                coefficients.append(coefficient)
    if len(coefficients) < required:
        raise ValueError(f'Insufficient finite bootstrap coefficients: {len(coefficients)}/{draws}')
    low, high = np.percentile(coefficients, [2.5, 97.5])
    return {'low': float(low), 'high': float(high), 'draws': draws, 'finite_draws': len(coefficients), 'discarded_nonfinite_draws': draws - len(coefficients), 'seed': seed}

def association_results(rows, label, bootstrap):
    x = np.asarray([r['pathology_score'] for r in rows], dtype=float)
    results = []
    for outcome in ('Area', 'IL6'):
        y = np.asarray([r[outcome] for r in rows], dtype=float)
        for name, function in (('Pearson', stats.pearsonr), ('Spearman', stats.spearmanr)):
            point, p = function(x, y)
            if not math.isfinite(float(point)) or not math.isfinite(float(p)):
                raise ValueError(f'Undefined {name} {outcome} association')
            result = {'analysis': label, 'n_animals': len(rows), 'outcome': outcome, 'statistic': name, 'coefficient': float(point), 'p_two_sided_nominal': float(p), 'multiple_testing_adjusted': False}
            if bootstrap:
                result['paired_animal_percentile95_CI'] = paired_bootstrap_ci(x, y, lambda a, b: function(a, b)[0])
            results.append(result)
    return results
