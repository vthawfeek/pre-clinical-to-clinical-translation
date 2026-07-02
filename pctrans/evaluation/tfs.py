"""Translational Fidelity Score (TFS) — the Day 10 composite metric.

TFS blends the two Gate 1 signals onto a common [0, 1] scale:

    TFS = 0.5 * knn_accuracy + 0.5 * (silhouette + 1) / 2

kNN accuracy is already in [0, 1]; the silhouette in [-1, 1] is rescaled to
[0, 1] by ``(s + 1) / 2``. Interpretation: TFS > 0.70 = high fidelity (good model
for its lineage), 0.50-0.70 = moderate (use with caution), < 0.50 = poor (domain
artefacts dominate; deprioritise for translational work).

The same formula scores the whole model (scalars) and each cell line (arrays):
per-cell-line TFS uses that sample's neighbour match fraction and its silhouette
contribution, giving the ranked "which cell lines translate best?" output.
"""

import numpy as np


def translational_fidelity_score(knn_accuracy, silhouette_score):
    """Composite TFS from an overall kNN accuracy and silhouette score."""
    return 0.5 * knn_accuracy + 0.5 * (silhouette_score + 1.0) / 2.0


def per_cell_line_tfs(match_fraction, silhouette_contribution):
    """Elementwise TFS per CCLE cell line.

    ``match_fraction``: fraction of a cell line's k neighbours matching its lineage.
    ``silhouette_contribution``: that cell line's per-sample silhouette value.
    """
    match = np.asarray(match_fraction, dtype=np.float64)
    sil = np.asarray(silhouette_contribution, dtype=np.float64)
    return 0.5 * match + 0.5 * (sil + 1.0) / 2.0
