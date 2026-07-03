"""Day 11 visualisation tests: UMAP projection + plotly/matplotlib figure builders.

The plotly / matplotlib builders are exercised on synthetic coordinates (fast).
`umap_projection` triggers the heavy numba JIT, so it is marked ``slow`` and kept
out of the default quality-gate suite.
"""

import matplotlib
import numpy as np
import plotly.graph_objects as go
import pytest

matplotlib.use("Agg")  # headless: no display in CI

from pctrans.evaluation.viz import (  # noqa: E402
    LINEAGE_COLORS,
    before_after_panel,
    braf_casestudy_panel,
    braf_casestudy_panel_interactive,
    confusion_matrix_heatmap,
    lineage_domain_scatter,
    lineage_domain_scatter_static,
    permutation_null_panel,
    purity_confounder_panel,
    tfs_ranking_bar,
    umap_projection,
)

LUAD, BRCA, SKCM = 0, 1, 2


def _synthetic(n_per=8):
    """Three lineage blobs, each with CCLE (domain 0) and TCGA (domain 1) points."""
    rng = np.random.default_rng(0)
    coords, lineage, domain, ids = [], [], [], []
    centers = {LUAD: (0, 0), BRCA: (10, 0), SKCM: (5, 10)}
    for lin, c in centers.items():
        for dom in (0, 1):
            pts = rng.normal(c, 0.5, size=(n_per, 2))
            coords.append(pts)
            lineage += [lin] * n_per
            domain += [dom] * n_per
            ids += [f"{'ACH' if dom == 0 else 'TCGA'}-{lin}{dom}{i}" for i in range(n_per)]
    return (
        np.concatenate(coords),
        np.array(lineage),
        np.array(domain),
        np.array(ids),
    )


def test_lineage_domain_scatter_is_figure_with_all_points():
    coords, lineage, domain, ids = _synthetic()
    fig = lineage_domain_scatter(coords, lineage, domain, "t", sample_ids=ids)
    assert isinstance(fig, go.Figure)
    # Every input point appears in exactly one (lineage, domain) trace.
    assert sum(len(tr.x) for tr in fig.data) == len(coords)


def test_lineage_domain_scatter_accepts_string_labels():
    coords, lineage, domain, ids = _synthetic()
    lin_str = np.array(["LUAD", "BRCA", "SKCM"])[lineage]
    dom_str = np.array(["CCLE", "TCGA"])[domain]
    fig = lineage_domain_scatter(coords, lin_str, dom_str, "t", sample_ids=ids)
    assert sum(len(tr.x) for tr in fig.data) == len(coords)
    # Colours come from the lineage palette.
    used = {tr.marker.color for tr in fig.data}
    assert used <= set(LINEAGE_COLORS.values())


def test_tfs_ranking_bar_selects_top_and_bottom():
    rng = np.random.default_rng(1)
    ids = np.array([f"ACH-{i:04d}" for i in range(30)])
    tfs = rng.uniform(0.3, 0.95, size=30)
    fig = tfs_ranking_bar(ids, tfs, top_n=5)
    assert isinstance(fig, go.Figure)
    bar = fig.data[0]
    assert len(bar.x) == 10  # bottom 5 + top 5
    # Rendered ascending -> the smallest selected value is the global minimum.
    assert bar.x[0] == pytest.approx(tfs.min())
    assert bar.x[-1] == pytest.approx(tfs.max())


def test_tfs_ranking_bar_small_input_keeps_all():
    ids = np.array([f"ACH-{i}" for i in range(6)])
    tfs = np.linspace(0.4, 0.9, 6)
    fig = tfs_ranking_bar(ids, tfs, top_n=10)
    assert len(fig.data[0].x) == 6


def test_before_after_panel_two_axes():
    coords, lineage, domain, _ = _synthetic()
    raw = coords + np.array([100.0, 0.0]) * domain[:, None]  # fake domain gap
    fig = before_after_panel(raw, domain, coords, lineage, domain)
    assert len(fig.axes) == 2


def test_lineage_domain_scatter_static_returns_figure():
    coords, lineage, domain, _ = _synthetic()
    fig = lineage_domain_scatter_static(coords, lineage, domain, "t")
    assert fig.axes  # a drawable matplotlib figure


@pytest.mark.slow
def test_umap_projection_shape():
    rng = np.random.default_rng(2)
    z = rng.normal(size=(40, 16)).astype(np.float32)
    coords = umap_projection(z, n_neighbors=15, seed=42)
    assert coords.shape == (40, 2)


# --- Day 19: 5+-lineage overrides + confusion-matrix heatmap ---


def test_lineage_domain_scatter_accepts_lineage_order_override():
    # A 5-lineage set outside the Phase-1 default LINEAGE_ORDER/IDX_TO_LINEAGE
    # must render every point, proving the override (not the module default) drives it.
    order = ["BLCA", "BRCA", "GBM", "LUAD", "SKCM"]
    idx_to_lineage = dict(enumerate(order))
    rng = np.random.default_rng(3)
    n_per = 4
    coords = rng.normal(size=(n_per * len(order) * 2, 2))
    lineage = np.repeat(np.arange(len(order)), n_per * 2)
    domain = np.tile(np.repeat([0, 1], n_per), len(order))

    fig = lineage_domain_scatter(
        coords, lineage, domain, "t", lineage_order=order, idx_to_lineage=idx_to_lineage
    )
    assert sum(len(tr.x) for tr in fig.data) == len(coords)
    used = {tr.marker.color for tr in fig.data}
    assert used <= {LINEAGE_COLORS[lineage_name] for lineage_name in order}


def test_confusion_matrix_heatmap_returns_figure():
    labels = ["LUAD", "LUSC", "BRCA"]
    cm = [[8, 2, 0], [1, 9, 0], [0, 0, 10]]
    fig = confusion_matrix_heatmap(cm, labels, title="t")
    assert fig.axes
    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == labels


def test_confusion_matrix_heatmap_normalizes_rows():
    labels = ["A", "B"]
    cm = [[5, 5], [2, 8]]
    fig = confusion_matrix_heatmap(cm, labels, normalize=True)
    im = fig.axes[0].images[0]
    normalized = im.get_array()
    np.testing.assert_allclose(normalized, [[0.5, 0.5], [0.2, 0.8]])


def test_purity_confounder_panel_returns_two_axes():
    rng = np.random.default_rng(0)
    projection = rng.normal(size=20)
    purity = rng.uniform(0.2, 1.0, size=20)
    domain = np.array([0] * 10 + [1] * 10)
    strata = {
        "high_purity": {"overall_accuracy": 0.9, "n": 15},
        "low_purity": {"overall_accuracy": 0.8, "n": 14},
    }
    fig = purity_confounder_panel(projection, purity, domain, strata, reference_line=0.85)
    assert len(fig.axes) == 2


def test_permutation_null_panel_one_axis_per_variant():
    results = {
        "eval-only": {"null_values": [0.05, 0.07, 0.1, 0.06], "real_value": 0.78,
                      "p_value": 0.02, "n_perm": 4},
        "retrain": {"null_values": [0.04, 0.09, 0.05, 0.08], "real_value": 0.78,
                    "p_value": 0.02, "n_perm": 4},
    }
    fig = permutation_null_panel(results, chance_level=1 / 15)
    assert len(fig.axes) == 2


def _braf_casestudy_inputs():
    rng = np.random.default_rng(0)
    coords = rng.normal(size=(10, 2))
    braf_status = np.array(["mutant"] * 5 + ["WT"] * 5)
    domain = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 1])  # 0=CCLE, 1=TCGA
    sample_ids = np.array([f"S{i}" for i in range(10)])
    proximity = rng.normal(size=6)
    auc = rng.uniform(0, 1, size=6)
    response_status = np.array(["mutant"] * 3 + ["WT"] * 3)
    response_ids = np.array([f"R{i}" for i in range(6)])
    placement_result = {"p_value": 0.01, "effect_size": 0.9}
    response_result = {"rho": -0.5, "ci_low": -0.8, "ci_high": -0.1, "n": 6}
    return (
        coords, braf_status, domain, sample_ids, proximity, auc, response_status, response_ids,
        placement_result, response_result,
    )


def test_braf_casestudy_panel_returns_two_axes():
    (coords, braf_status, domain, _sample_ids, proximity, auc, response_status, _response_ids,
     placement_result, response_result) = _braf_casestudy_inputs()
    fig = braf_casestudy_panel(
        coords, braf_status, domain, proximity, auc, response_status, placement_result, response_result,
    )
    assert len(fig.axes) == 2


def test_braf_casestudy_panel_adds_third_axis_with_drug_signal_result():
    (coords, braf_status, domain, _sample_ids, proximity, auc, response_status, _response_ids,
     placement_result, response_result) = _braf_casestudy_inputs()
    drug_signal_result = {
        "braf_status": {"r2": -0.05, "rho": 0.1, "p_value": 0.6, "n": 20, "n_splits": 5},
        "raw_expression": {"r2": 0.2, "rho": 0.4, "p_value": 0.05, "n": 20, "n_splits": 5},
        "embedding": {"r2": 0.02, "rho": 0.15, "p_value": 0.5, "n": 20, "n_splits": 5},
    }
    fig = braf_casestudy_panel(
        coords, braf_status, domain, proximity, auc, response_status, placement_result, response_result,
        drug_signal_result=drug_signal_result,
    )
    assert len(fig.axes) == 3


def test_braf_casestudy_panel_interactive_has_two_subplots():
    (coords, braf_status, domain, sample_ids, proximity, auc, response_status, response_ids,
     _placement_result, _response_result) = _braf_casestudy_inputs()
    fig = braf_casestudy_panel_interactive(
        coords, braf_status, domain, sample_ids, proximity, auc, response_status, response_ids,
    )
    assert len(fig.data) > 0
    xaxis_titles = {fig.layout[k].title.text for k in fig.layout if k.startswith("xaxis")}
    assert "UMAP-1" in xaxis_titles
