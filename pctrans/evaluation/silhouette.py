"""Cross-domain silhouette — the Day 10 Gate 1 secondary metric.

The silhouette is computed on the POOLED CCLE + TCGA test embeddings using the
*lineage* label (not the domain label) as the cluster assignment. A positive
score means within-lineage samples — across the petri-dish/patient boundary —
are more similar to each other than to other lineages, i.e. alignment worked. A
negative score means domain artefacts still dominate over lineage signal.

`domain_labels` is accepted (and recorded by callers) to make the cross-domain
intent explicit, but by design it is *not* used as the clustering label: scoring
on domain would reward the failure mode this project exists to remove.
"""

import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score


def cross_domain_silhouette(embeddings, lineage_labels, domain_labels=None, metric="euclidean"):
    """Overall silhouette score of pooled embeddings clustered by lineage.

    Returns 0.0 when fewer than two lineages are present (silhouette undefined).
    """
    embeddings = np.asarray(embeddings, dtype=np.float64)
    lineage_labels = np.asarray(lineage_labels)
    if len(np.unique(lineage_labels)) < 2 or len(lineage_labels) < 2:
        return 0.0
    return float(silhouette_score(embeddings, lineage_labels, metric=metric))


def silhouette_contributions(embeddings, lineage_labels, metric="euclidean"):
    """Per-sample silhouette values (the per-point contributions to the mean).

    Used for per-cell-line TFS: the CCLE slice of these contributions feeds the
    silhouette half of each cell line's Translational Fidelity Score.
    """
    embeddings = np.asarray(embeddings, dtype=np.float64)
    lineage_labels = np.asarray(lineage_labels)
    if len(np.unique(lineage_labels)) < 2 or len(lineage_labels) < 2:
        return np.zeros(len(lineage_labels), dtype=np.float64)
    return silhouette_samples(embeddings, lineage_labels, metric=metric)
