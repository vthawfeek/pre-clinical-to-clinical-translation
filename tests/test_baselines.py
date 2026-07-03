"""Day 17 tests: real batch-correction baselines + supervised cross-domain ceiling."""

import numpy as np
import pandas as pd

from pctrans.data.dataset import CCLEDataset, TCGADataset
from pctrans.evaluation.baselines import (
    combat_knn,
    harmony_knn,
    pca_knn,
    scanorama_knn,
    supervised_ceiling,
)

LINEAGES = ["LUAD", "BRCA", "SKCM"]


def _lineage_separable(n_per_lineage, n_genes, seed):
    """Lineage-separable synthetic expression: each lineage lifts a 5-gene block."""
    rng = np.random.default_rng(seed)
    rows, labels = [], []
    for li, lineage in enumerate(LINEAGES):
        base = np.zeros(n_genes)
        base[li * 5 : (li + 1) * 5] = 4.0
        for _ in range(n_per_lineage):
            rows.append(base + rng.normal(0.0, 1.0, size=n_genes))
            labels.append(lineage)
    df = pd.DataFrame(rows, columns=[f"GENE{i}" for i in range(n_genes)])
    df["lineage"] = labels
    return df


def _ccle(n_per_lineage, seed):
    return CCLEDataset(_lineage_separable(n_per_lineage, 30, seed))


def _tcga(n_per_lineage, seed):
    return TCGADataset(_lineage_separable(n_per_lineage, 30, seed))


def test_supervised_ceiling_beats_random():
    ccle_train = _ccle(20, seed=10)
    tcga_test = _tcga(30, seed=12)
    acc = supervised_ceiling(ccle_train, tcga_test)
    assert 1.0 / 3.0 < acc <= 1.0


def test_baseline_knn_shapes_and_range():
    ccle_test = _ccle(15, seed=11)
    tcga_test = _tcga(30, seed=12)
    for fn in (pca_knn, harmony_knn, combat_knn, scanorama_knn):
        acc = fn(ccle_test, tcga_test, k=5)
        if acc is None:
            # Optional dep (inmoose / scanorama) not installed in this environment.
            continue
        assert 0.0 <= acc <= 1.0


def test_harmony_knn_runs_for_real():
    # harmonypy is a `dev`-extra dependency with no native build step, so this
    # must produce a real number, not silently skip.
    ccle_test = _ccle(15, seed=11)
    tcga_test = _tcga(30, seed=12)
    acc = harmony_knn(ccle_test, tcga_test, k=5)
    assert acc is not None
    assert 0.0 <= acc <= 1.0
