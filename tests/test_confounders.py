"""Day 20 tests: tumour-purity confounder analysis."""

import numpy as np
import pandas as pd
import pytest

from pctrans.evaluation.confounders import (
    CCLE_PURITY,
    domain_axis_purity_correlation,
    load_purity,
    purity_residualised_silhouette,
    purity_stratified_knn,
    residualise_purity,
)

LUAD, BRCA, SKCM = 0, 1, 2
IDX_TO_LINEAGE = {LUAD: "LUAD", BRCA: "BRCA", SKCM: "SKCM"}
LINEAGE_ORDER = ["LUAD", "BRCA", "SKCM"]


def _lineage_blobs(n_per_lineage, n_dims, seed):
    """Cross-domain-aligned, lineage-separable embeddings (unit-norm blobs)."""
    rng = np.random.default_rng(seed)
    z, y = [], []
    centers = {LUAD: 0, BRCA: 1, SKCM: 2}
    for lineage, offset in centers.items():
        base = np.zeros(n_dims)
        base[offset] = 5.0
        for _ in range(n_per_lineage):
            z.append(base + rng.normal(0.0, 0.3, size=n_dims))
            y.append(lineage)
    return np.asarray(z), np.asarray(y)


def _purity_table_path(tmp_path, barcodes, purities):
    df = pd.DataFrame({"array": barcodes, "purity": purities})
    path = tmp_path / "purity.txt"
    df.to_csv(path, sep="\t", index=False)
    return path


def test_load_purity_assigns_ccle_pure_and_joins_tcga(tmp_path):
    ids_ccle = np.array(["ACH-1", "ACH-2"])
    ids_tcga = np.array(["TCGA-AA-0001-01", "TCGA-AA-0002-01", "TCGA-AA-0003-01"])
    path = _purity_table_path(
        tmp_path,
        barcodes=["TCGA-AA-0001-01", "TCGA-AA-0002-01"],  # 0003 missing -> NaN
        purities=[0.9, 0.4],
    )
    purity_ccle, purity_tcga = load_purity(path, ids_ccle, ids_tcga)
    assert np.all(purity_ccle == CCLE_PURITY)
    assert purity_tcga[0] == pytest.approx(0.9)
    assert purity_tcga[1] == pytest.approx(0.4)
    assert np.isnan(purity_tcga[2])


def test_domain_axis_purity_correlation_detects_strong_signal():
    # Domain axis points CCLE (purity=1) -> TCGA (purity=1-projection), so a
    # sample's projection grows as its purity falls -- a strong *negative*
    # domain-axis/purity correlation is the "axis is a purity axis" signal.
    rng = np.random.default_rng(0)
    n = 60
    purity_tcga = rng.uniform(0.1, 0.9, size=n)
    z_ccle = np.column_stack([np.ones(20), np.zeros(20)])
    z_tcga = np.column_stack([purity_tcga, np.zeros(n)])
    purity_ccle = np.ones(20)

    result = domain_axis_purity_correlation(z_ccle, z_tcga, purity_ccle, purity_tcga)
    assert result["r"] < -0.9
    assert result["n"] == 80


def test_domain_axis_purity_correlation_ignores_nan_purity():
    z_ccle = np.ones((5, 2))
    z_tcga = np.zeros((5, 2))
    purity_ccle = np.ones(5)
    purity_tcga = np.array([np.nan] * 5)
    result = domain_axis_purity_correlation(z_ccle, z_tcga, purity_ccle, purity_tcga)
    # Only the 5 finite CCLE purities remain, all identical -> zero-variance guard.
    assert result["r"] == 0.0
    assert result["n"] == 5


def test_purity_stratified_knn_runs():
    z_ccle, y_ccle = _lineage_blobs(10, 8, seed=1)
    z_tcga, y_tcga = _lineage_blobs(40, 8, seed=2)
    rng = np.random.default_rng(3)
    purity_tcga = rng.uniform(0.2, 1.0, size=len(y_tcga))

    both_strata = purity_stratified_knn(
        z_ccle, y_ccle, z_tcga, y_tcga, purity_tcga, k=5,
        idx_to_lineage=IDX_TO_LINEAGE, lineage_order=LINEAGE_ORDER,
    )
    assert set(both_strata) == {"high_purity", "low_purity"}
    for entry in both_strata.values():
        assert entry["overall_accuracy"] is None or 0.0 <= entry["overall_accuracy"] <= 1.0
        assert entry["n"] > 0
    # Lineage-separable synthetic data should retrieve near-perfectly in both halves.
    assert both_strata["high_purity"]["overall_accuracy"] > 0.9
    assert both_strata["low_purity"]["overall_accuracy"] > 0.9


def test_purity_stratified_knn_handles_missing_purity():
    z_ccle, y_ccle = _lineage_blobs(5, 6, seed=4)
    z_tcga, y_tcga = _lineage_blobs(20, 6, seed=5)
    purity_tcga = np.array([np.nan] * len(y_tcga))  # no ABSOLUTE call for any sample

    result = purity_stratified_knn(z_ccle, y_ccle, z_tcga, y_tcga, purity_tcga, k=5)
    assert result["high_purity"]["overall_accuracy"] is None
    assert result["low_purity"]["overall_accuracy"] is None
    assert result["high_purity"]["n"] == 0
    assert result["low_purity"]["n"] == 0


def test_residualisation_preserves_shape():
    rng = np.random.default_rng(6)
    embeddings = rng.normal(size=(50, 16))
    purity = rng.uniform(0.1, 1.0, size=50)
    residual = residualise_purity(embeddings, purity)
    assert residual.shape == embeddings.shape
    assert np.all(np.isfinite(residual))


def test_residualisation_removes_linear_purity_trend():
    rng = np.random.default_rng(7)
    n = 100
    purity = rng.uniform(0.0, 1.0, size=n)
    noise = rng.normal(0.0, 0.05, size=n)
    # Dim 0 is purely a linear function of purity; residualising it should
    # collapse its correlation with purity to ~0.
    embeddings = np.column_stack([3.0 * purity + noise, rng.normal(size=n)])
    residual = residualise_purity(embeddings, purity)
    corr_before = np.corrcoef(embeddings[:, 0], purity)[0, 1]
    corr_after = np.corrcoef(residual[:, 0], purity)[0, 1]
    assert abs(corr_before) > 0.9
    assert abs(corr_after) < 0.05


def test_residualisation_passes_through_nonfinite_purity_unchanged():
    embeddings = np.ones((3, 4))
    purity = np.array([np.nan, np.nan, np.nan])
    residual = residualise_purity(embeddings, purity)
    assert np.array_equal(residual, embeddings)


def test_purity_residualised_silhouette_stays_positive_for_separable_lineages():
    z_ccle, y_ccle = _lineage_blobs(15, 10, seed=8)
    z_tcga, y_tcga = _lineage_blobs(45, 10, seed=9)
    rng = np.random.default_rng(10)
    purity_ccle = np.ones(len(y_ccle))
    purity_tcga = rng.uniform(0.2, 1.0, size=len(y_tcga))

    score = purity_residualised_silhouette(
        z_ccle, y_ccle, z_tcga, y_tcga, purity_ccle, purity_tcga
    )
    assert score > 0.0
