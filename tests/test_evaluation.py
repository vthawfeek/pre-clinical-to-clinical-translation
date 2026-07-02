"""Day 10 evaluation-module tests: kNN retrieval, silhouette, and TFS."""

import numpy as np
import torch
from torch.utils.data import DataLoader

from pctrans.data.dataset import LINEAGE_TO_IDX
from pctrans.evaluation.knn import (
    knn_accuracy_from_embeddings,
    knn_retrieval_accuracy,
)
from pctrans.evaluation.silhouette import (
    cross_domain_silhouette,
    silhouette_contributions,
)
from pctrans.evaluation.tfs import per_cell_line_tfs, translational_fidelity_score

LUAD, BRCA, SKCM = LINEAGE_TO_IDX["LUAD"], LINEAGE_TO_IDX["BRCA"], LINEAGE_TO_IDX["SKCM"]


def _perfect_embeddings(n_ccle_per=3, n_tcga_per=10):
    """One-hot lineage embeddings: LUAD=[1,0,0], BRCA=[0,1,0], SKCM=[0,0,1]."""
    onehot = np.eye(3, dtype=np.float64)
    labels = [LUAD, BRCA, SKCM]
    z_ccle = np.concatenate([np.tile(onehot[i], (n_ccle_per, 1)) for i in range(3)])
    y_ccle = np.array([labels[i] for i in range(3) for _ in range(n_ccle_per)])
    z_tcga = np.concatenate([np.tile(onehot[i], (n_tcga_per, 1)) for i in range(3)])
    y_tcga = np.array([labels[i] for i in range(3) for _ in range(n_tcga_per)])
    return z_ccle, y_ccle, z_tcga, y_tcga


def test_knn_accuracy_random_model():
    # Random 64-dim embeddings on 3 balanced lineages -> ~33% (chance).
    rng = np.random.default_rng(0)
    n_ccle, n_tcga = 300, 900
    z_ccle = rng.normal(size=(n_ccle, 64))
    z_tcga = rng.normal(size=(n_tcga, 64))
    y_ccle = np.array([LUAD, BRCA, SKCM] * (n_ccle // 3))
    y_tcga = np.array([LUAD, BRCA, SKCM] * (n_tcga // 3))
    res = knn_accuracy_from_embeddings(z_ccle, y_ccle, z_tcga, y_tcga, k=5)
    assert 0.23 <= res["overall_accuracy"] <= 0.43


def test_knn_accuracy_perfect_embeddings():
    z_ccle, y_ccle, z_tcga, y_tcga = _perfect_embeddings()
    res = knn_accuracy_from_embeddings(z_ccle, y_ccle, z_tcga, y_tcga, k=5)
    assert res["overall_accuracy"] == 1.0
    assert res["k_table"][5] == 1.0
    # Confusion matrix is diagonal (every cell line predicted correctly).
    cm = np.array(res["confusion_matrix"])
    assert (cm - np.diag(np.diag(cm))).sum() == 0


def test_knn_confusion_matrix_shape():
    z_ccle, y_ccle, z_tcga, y_tcga = _perfect_embeddings()
    res = knn_accuracy_from_embeddings(z_ccle, y_ccle, z_tcga, y_tcga, k=5)
    assert np.array(res["confusion_matrix"]).shape == (3, 3)
    assert res["confusion_labels"] == ["LUAD", "BRCA", "SKCM"]
    assert res["match_fraction"].shape == (len(y_ccle),)


def test_knn_retrieval_accuracy_end_to_end(small_model, tiny_ccle_dataset, tiny_tcga_dataset):
    ccle_loader = DataLoader(tiny_ccle_dataset, batch_size=8, shuffle=False)
    tcga_loader = DataLoader(tiny_tcga_dataset, batch_size=8, shuffle=False)
    with torch.no_grad():
        res = knn_retrieval_accuracy(small_model, ccle_loader, tcga_loader, k=5)
    assert 0.0 <= res["overall_accuracy"] <= 1.0
    assert set(res["embeddings"]) == {"z_ccle", "y_ccle", "z_tcga", "y_tcga"}
    assert res["embeddings"]["z_ccle"].shape[0] == len(tiny_ccle_dataset)


def test_silhouette_perfect_clusters():
    z_ccle, y_ccle, z_tcga, y_tcga = _perfect_embeddings()
    z = np.concatenate([z_ccle, z_tcga])
    y = np.concatenate([y_ccle, y_tcga])
    score = cross_domain_silhouette(z, y, domain_labels=None)
    assert score == 1.0


def test_silhouette_single_lineage_returns_zero():
    z = np.random.default_rng(1).normal(size=(10, 8))
    y = np.zeros(10, dtype=int)
    assert cross_domain_silhouette(z, y) == 0.0


def test_silhouette_contributions_length():
    z_ccle, y_ccle, z_tcga, y_tcga = _perfect_embeddings()
    z = np.concatenate([z_ccle, z_tcga])
    y = np.concatenate([y_ccle, y_tcga])
    contribs = silhouette_contributions(z, y)
    assert contribs.shape == (len(y),)
    assert np.all(contribs <= 1.0) and np.all(contribs >= -1.0)


def test_tfs_formula():
    # 0.5 * 0.8 + 0.5 * (0.4 + 1) / 2 = 0.4 + 0.35 = 0.75
    assert translational_fidelity_score(0.8, 0.4) == 0.75


def test_tfs_range():
    for acc in (0.0, 0.5, 1.0):
        for sil in (-1.0, 0.0, 1.0):
            tfs = translational_fidelity_score(acc, sil)
            assert 0.0 <= tfs <= 1.0


def test_per_cell_line_tfs_elementwise():
    match = np.array([1.0, 0.0, 0.6])
    sil = np.array([1.0, -1.0, 0.2])
    tfs = per_cell_line_tfs(match, sil)
    assert tfs.shape == (3,)
    assert np.all(tfs >= 0.0) and np.all(tfs <= 1.0)
    # First cell line: perfect match + max silhouette -> TFS 1.0.
    assert tfs[0] == 1.0
