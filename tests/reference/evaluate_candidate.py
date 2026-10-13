# Reference syntax-tree fixtures for the reported v6 implementation.
# Parsed by tests; not executable entrypoints.

def select_validation_threshold(labels, probabilities):
    """Select the first maximum of validation Youden's J; record nonfinite optima."""
    fpr, tpr, thresholds = roc_curve(labels, probabilities, pos_label=1)
    index = int(np.argmax(tpr - fpr))
    threshold = float(thresholds[index])
    return {'method': 'validation-only Youden J; first np.argmax in sklearn roc_curve order', 'threshold': threshold if np.isfinite(threshold) else None, 'status': 'locked' if np.isfinite(threshold) else 'nonfinite_optimum_no_threshold_metrics', 'youden_j': float((tpr - fpr)[index]), 'validation_auc_lesion': float(roc_auc_score(labels, probabilities))}

def metrics(labels, probabilities, threshold):
    result = {'n': len(labels), 'lesion_n': int(labels.sum()), 'normal_n': int(len(labels) - labels.sum()), 'auc_lesion': float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else None}
    if threshold is None:
        return result
    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = map(int, confusion_matrix(labels, predicted, labels=[0, 1]).ravel())
    result.update(TN=tn, FP=fp, FN=fn, TP=tp, accuracy=(tp + tn) / len(labels), precision=tp / (tp + fp) if tp + fp else 0.0, recall=tp / (tp + fn) if tp + fn else 0.0, specificity=tn / (tn + fp) if tn + fp else None, f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return result

def bootstrap_auc(labels, probabilities, source_ids, source_order, patch_draws=PATCH_DRAWS, cluster_draws=CLUSTER_DRAWS):
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
        return {'requested_draws': requested, 'valid_draws': len(values), 'percentile_95_ci': np.percentile(values, [2.5, 97.5]).tolist() if values else None}
    return {'seed': BOOTSTRAP_SEED, 'patch_iid_descriptive': result(patch_values, patch_draws), 'source_wsi_cluster': result(cluster_values, cluster_draws), 'cluster_count': len(source_order), 'interpretation': 'Patch AUC uncertainty; 4 source clusters. Per-WSI patch AUC is not animal-level classification AUC.'}
