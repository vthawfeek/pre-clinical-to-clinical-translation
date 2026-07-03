"""Day 25 tests: the Celligner head-to-head benchmark."""

import numpy as np

from pctrans.evaluation.celligner_compare import retrieval_on_embedding, run_celligner
from pctrans.evaluation.knn import knn_accuracy_from_embeddings

LINEAGES = ["LUAD", "BRCA", "SKCM"]


def _lineage_separable_embedding(n_per_lineage, n_dim, seed):
    """Perfectly lineage-separable synthetic embedding, both domains identical."""
    rng = np.random.default_rng(seed)
    rows, ids, lineage_of = [], [], {}
    for li, lineage in enumerate(LINEAGES):
        centre = np.zeros(n_dim)
        centre[li] = 10.0
        for j in range(n_per_lineage):
            sample_id = f"{lineage}_{j}"
            rows.append(centre + rng.normal(0.0, 0.1, size=n_dim))
            ids.append(sample_id)
            lineage_of[sample_id] = li
    return np.array(rows), ids, lineage_of


def test_celligner_compare_skips_without_dep():
    # `celligner` is not installed in this environment (see the module
    # docstring: its PyPI release cannot even be resolved by pip/uv), so
    # `run_celligner` must return `None` cleanly rather than raising.
    ccle_expr = np.random.default_rng(0).normal(size=(5, 10))
    tcga_expr = np.random.default_rng(1).normal(size=(8, 10))
    result = run_celligner(ccle_expr, tcga_expr)
    assert result is None


def test_retrieval_on_embedding_matches_our_metric():
    # `retrieval_on_embedding` must reproduce `knn_accuracy_from_embeddings`
    # exactly on a fixed array: the whole point of Day 25 is that Celligner
    # and the contrastive model are scored with the identical function.
    z_ccle, ccle_ids, lineage_of = _lineage_separable_embedding(4, n_dim=3, seed=10)
    z_tcga, tcga_ids, tcga_lineage_of = _lineage_separable_embedding(6, n_dim=3, seed=11)
    lineage_of.update(tcga_lineage_of)

    joint_emb = np.concatenate([z_ccle, z_tcga], axis=0)
    result = retrieval_on_embedding(joint_emb, ccle_ids, tcga_ids, lineage_of, k=5)

    y_ccle = np.array([lineage_of[i] for i in ccle_ids])
    y_tcga = np.array([lineage_of[i] for i in tcga_ids])
    expected = knn_accuracy_from_embeddings(z_ccle, y_ccle, z_tcga, y_tcga, k=5)

    assert result["overall_accuracy"] == expected["overall_accuracy"]
    assert result["per_lineage"] == expected["per_lineage"]
    assert result["overall_accuracy"] == 1.0  # perfectly separable by construction


def test_retrieval_on_embedding_shapes_and_range():
    z_ccle, ccle_ids, lineage_of = _lineage_separable_embedding(5, n_dim=3, seed=20)
    z_tcga, tcga_ids, tcga_lineage_of = _lineage_separable_embedding(7, n_dim=3, seed=21)
    lineage_of.update(tcga_lineage_of)

    joint_emb = np.concatenate([z_ccle, z_tcga], axis=0)
    result = retrieval_on_embedding(joint_emb, ccle_ids, tcga_ids, lineage_of, k=5)

    assert 0.0 <= result["overall_accuracy"] <= 1.0
    assert -1.0 <= result["silhouette"] <= 1.0
    assert set(result["per_lineage"]) <= set(LINEAGES)
