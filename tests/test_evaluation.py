"""Day 10 evaluation-module tests: kNN retrieval, silhouette, and TFS."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from pctrans.data.dataset import LINEAGE_TO_IDX, build_lineage_maps
from pctrans.evaluation.knn import (
    KNOWN_CONFUSABLE_PAIRS,
    confusable_pair_mass,
    knn_accuracy_from_embeddings,
    knn_retrieval_accuracy,
    top_confusions,
)
from pctrans.evaluation.silhouette import (
    cross_domain_silhouette,
    silhouette_contributions,
)
from pctrans.evaluation.tfs import per_cell_line_tfs, translational_fidelity_score

LUAD, BRCA, SKCM = LINEAGE_TO_IDX["LUAD"], LINEAGE_TO_IDX["BRCA"], LINEAGE_TO_IDX["SKCM"]

_LINEAGES_15 = [
    "BLCA", "BRCA", "COAD", "GBM", "HNSC", "KIRC", "LGG", "LIHC",
    "LUAD", "LUSC", "OV", "PAAD", "READ", "SKCM", "STAD",
]


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


# --- Day 19: 15-lineage confusion-matrix generalisation + error-structure ---


def test_knn_accuracy_from_embeddings_accepts_custom_lineage_map():
    # A 5-lineage config-driven map (Day 18 build_lineage_maps) must produce a
    # 5x5 confusion matrix, not silently fall back to the 3-lineage default.
    lineages = ["BLCA", "BRCA", "GBM", "LUAD", "SKCM"]
    lineage_to_idx, idx_to_lineage = build_lineage_maps(lineages)
    onehot = np.eye(5, dtype=np.float64)
    z_ccle = np.concatenate([np.tile(onehot[i], (3, 1)) for i in range(5)])
    y_ccle = np.array([i for i in range(5) for _ in range(3)])
    z_tcga = np.concatenate([np.tile(onehot[i], (10, 1)) for i in range(5)])
    y_tcga = np.array([i for i in range(5) for _ in range(10)])

    res = knn_accuracy_from_embeddings(
        z_ccle, y_ccle, z_tcga, y_tcga, k=5, idx_to_lineage=idx_to_lineage, lineage_order=lineages
    )
    assert res["overall_accuracy"] == 1.0
    assert np.array(res["confusion_matrix"]).shape == (5, 5)
    assert res["confusion_labels"] == lineages
    assert set(res["per_lineage"]) == set(lineages)


@pytest.fixture
def eval15():
    """Synthetic 15-lineage confusion result, off-diagonal mass concentrated on
    the plan's named confusable pairs (LUAD-LUSC, GBM-LGG, COAD-READ, LUSC-HNSC)."""
    labels = list(_LINEAGES_15)
    idx = {lab: i for i, lab in enumerate(labels)}
    n = len(labels)
    cm = np.eye(n, dtype=int) * 20  # strong diagonal: mostly correct

    def bump(a, b, count):
        cm[idx[a], idx[b]] += count

    bump("LUAD", "LUSC", 6)
    bump("LUSC", "LUAD", 3)
    bump("GBM", "LGG", 5)
    bump("LGG", "GBM", 4)
    bump("COAD", "READ", 4)
    bump("READ", "COAD", 3)
    bump("LUSC", "HNSC", 3)
    # A little unrelated noise so the "majority, not all" claim is meaningful.
    bump("BRCA", "OV", 1)
    bump("KIRC", "STAD", 1)

    return {"confusion_matrix": cm.tolist(), "confusion_labels": labels}


def test_confusion_matrix_is_15x15(eval15):
    cm = np.array(eval15["confusion_matrix"])
    assert cm.shape == (15, 15)
    assert len(eval15["confusion_labels"]) == 15
    assert len(set(eval15["confusion_labels"])) == 15


def test_offdiagonal_mass_on_related_pairs(eval15):
    mass = confusable_pair_mass(
        eval15["confusion_matrix"], eval15["confusion_labels"], KNOWN_CONFUSABLE_PAIRS
    )
    # Majority of the off-diagonal (error) mass sits on the curated biologically-
    # related pairs, not scattered randomly across all 15*14 off-diagonal cells.
    assert mass > 0.5


def test_top_confusions_sorted_descending(eval15):
    top = top_confusions(eval15["confusion_matrix"], eval15["confusion_labels"], top_n=3)
    assert len(top) == 3
    counts = [t[2] for t in top]
    assert counts == sorted(counts, reverse=True)
    # The single largest confusion is the LUAD->LUSC bump (count 6).
    assert top[0][:2] == ("LUAD", "LUSC")


def test_confusable_pair_mass_zero_when_no_pairs_present():
    labels = ["LUAD", "BRCA", "SKCM"]
    cm = [[10, 1, 0], [0, 10, 1], [1, 0, 10]]  # off-diagonal misses are all unnamed pairs
    mass = confusable_pair_mass(cm, labels, KNOWN_CONFUSABLE_PAIRS)
    assert mass == 0.0


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
