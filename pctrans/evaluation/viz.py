"""Day 11 visualisation: UMAP projection + publication figures.

`umap_projection` embeds the pooled test embeddings into 2-D. The plotly builders
(`lineage_domain_scatter`, `tfs_ranking_bar`) produce the interactive figures
saved as HTML for the notebook and Streamlit app; the matplotlib builders
(`lineage_domain_scatter_static`, `before_after_panel`) produce the static PNGs
used in the blog / LinkedIn posts (no kaleido binary needed).

Encoding is constant across every figure so the reader can read them at a glance:
colour = lineage (LUAD blue / BRCA pink / SKCM brown); marker shape = domain
(TCGA patient = circle, CCLE cell line = cross). A cell line landing *inside* its
lineage's patient cloud is the whole point of the project made visible.
"""

import numpy as np

from pctrans.data.dataset import IDX_TO_LINEAGE

# Lineage colours (plan: LUAD=blue, BRCA=pink, SKCM=brown). Extended on Day 19 to
# cover all 15 Phase-2 lineages (`configs/data_15.yaml`); the original 3 keep
# their Phase-1 hex values so old figures are unaffected. `lineage_domain_scatter`
# and friends only ever iterate `lineage_order` (default: the 3-lineage
# `LINEAGE_ORDER` below), so the extra keys are a no-op unless a caller passes a
# longer `lineage_order`.
LINEAGE_COLORS = {
    "LUAD": "#1f77b4",
    "BRCA": "#e377c2",
    "SKCM": "#8c564b",
    "BLCA": "#ff7f0e",
    "COAD": "#2ca02c",
    "GBM": "#d62728",
    "HNSC": "#9467bd",
    "KIRC": "#17becf",
    "LGG": "#bcbd22",
    "LIHC": "#7f7f7f",
    "LUSC": "#aec7e8",
    "OV": "#f7b6d2",
    "PAAD": "#c49c94",
    "READ": "#98df8a",
    "STAD": "#ffbb78",
}
LINEAGE_ORDER = ["LUAD", "BRCA", "SKCM"]

# Domain 0 = CCLE cell line, 1 = TCGA patient (matches evaluate.py's pooling).
DOMAIN_NAMES = {0: "CCLE", 1: "TCGA"}
_DOMAIN_SYMBOL = {"CCLE": "x", "TCGA": "circle"}  # plotly marker symbols
_DOMAIN_MARKER = {"CCLE": "X", "TCGA": "o"}  # matplotlib markers

# TFS colour bands (shared with the Streamlit gauge, Day 12).
_TFS_GREEN, _TFS_YELLOW, _TFS_RED = "#2ca02c", "#ff7f0e", "#d62728"


def _tfs_color(value):
    if not np.isfinite(value):
        return "#7f7f7f"
    if value >= 0.70:
        return _TFS_GREEN
    if value >= 0.50:
        return _TFS_YELLOW
    return _TFS_RED


def _lineage_names(labels, idx_to_lineage=None):
    """Normalise lineage labels (int codes or strings) to lineage-name strings."""
    labels = np.asarray(labels)
    if labels.dtype.kind in "iu":
        mapping = idx_to_lineage if idx_to_lineage is not None else IDX_TO_LINEAGE
        return np.array([mapping[int(v)] for v in labels])
    return labels.astype(str)


def _domain_names(labels):
    """Normalise domain labels (0/1 codes or strings) to 'CCLE'/'TCGA'."""
    labels = np.asarray(labels)
    if labels.dtype.kind in "iu":
        return np.array([DOMAIN_NAMES[int(v)] for v in labels])
    return labels.astype(str)


def umap_projection(embeddings, n_neighbors=15, min_dist=0.1, n_components=2, seed=42):
    """Fit UMAP on the pooled embeddings, returning the ``(n, n_components)`` coords.

    ``n_neighbors`` is clamped to ``n_samples - 1`` so the projection still runs on
    the small held-out test set (and in unit tests) without a UMAP error.
    """
    import umap  # lazy: heavy numba JIT on first import

    embeddings = np.asarray(embeddings, dtype=np.float32)
    n = len(embeddings)
    n_neighbors = max(2, min(n_neighbors, n - 1))
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        random_state=seed,
        metric="euclidean",
    )
    return np.asarray(reducer.fit_transform(embeddings))


def lineage_domain_scatter(
    coords,
    lineage_labels,
    domain_labels,
    title,
    sample_ids=None,
    tfs_scores=None,
    lineage_order=None,
    idx_to_lineage=None,
):
    """Interactive plotly scatter, colour = lineage, marker = domain.

    Hover shows the sample ID, its lineage/domain, and (for CCLE cell lines) its
    TFS. One trace per (lineage, domain) pair so the legend toggles cleanly.
    ``lineage_order``/``idx_to_lineage`` default to the Phase-1 3-lineage module
    constants; pass the Day 18 ``build_lineage_maps`` output (and its lineage
    list) to render an arbitrary-size lineage set, e.g. the Day 19 15-lineage UMAP.
    """
    import plotly.graph_objects as go

    order = lineage_order if lineage_order is not None else LINEAGE_ORDER
    coords = np.asarray(coords)
    lin = _lineage_names(lineage_labels, idx_to_lineage)
    dom = _domain_names(domain_labels)
    n = len(coords)
    ids = np.asarray(sample_ids) if sample_ids is not None else np.array([""] * n)
    tfs = (
        np.asarray(tfs_scores, dtype=float)
        if tfs_scores is not None
        else np.full(n, np.nan)
    )

    fig = go.Figure()
    for lineage in order:
        for domain in ("TCGA", "CCLE"):  # patients first so cell lines sit on top
            mask = (lin == lineage) & (dom == domain)
            if not mask.any():
                continue
            m_tfs = tfs[mask]
            customdata = np.column_stack([ids[mask], m_tfs])
            tfs_line = "TFS: %{customdata[1]:.3f}<br>" if domain == "CCLE" else ""
            fig.add_trace(
                go.Scatter(
                    x=coords[mask, 0],
                    y=coords[mask, 1],
                    mode="markers",
                    name=f"{lineage} · {domain}",
                    marker={
                        "color": LINEAGE_COLORS[lineage],
                        "symbol": _DOMAIN_SYMBOL[domain],
                        "size": 11 if domain == "CCLE" else 6,
                        "opacity": 0.95 if domain == "CCLE" else 0.45,
                        "line": {
                            "width": 1.2 if domain == "CCLE" else 0,
                            "color": "black",
                        },
                    },
                    customdata=customdata,
                    hovertemplate=(
                        "%{customdata[0]}<br>"
                        f"{lineage} · {domain}<br>"
                        f"{tfs_line}"
                        "<extra></extra>"
                    ),
                )
            )
    fig.update_layout(
        title=title,
        xaxis_title="UMAP-1",
        yaxis_title="UMAP-2",
        template="plotly_white",
        legend_title="Lineage · Domain",
        width=900,
        height=700,
    )
    return fig


def tfs_ranking_bar(cell_line_ids, tfs_scores, top_n=10):
    """Horizontal bar chart of the top ``top_n`` and bottom ``top_n`` cell lines by TFS.

    Bars are coloured by TFS band (green > 0.70, yellow 0.50-0.70, red < 0.50).
    """
    import plotly.graph_objects as go

    ids = np.asarray(cell_line_ids).astype(str)
    tfs = np.asarray(tfs_scores, dtype=float)
    order = np.argsort(tfs)  # ascending
    if len(ids) <= 2 * top_n:
        sel = order
    else:
        sel = np.concatenate([order[:top_n], order[-top_n:]])
    ids_s, tfs_s = ids[sel], tfs[sel]  # ascending -> highest at top of a horizontal bar

    fig = go.Figure(
        go.Bar(
            x=tfs_s,
            y=ids_s,
            orientation="h",
            marker_color=[_tfs_color(v) for v in tfs_s],
            text=[f"{v:.3f}" for v in tfs_s],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=f"Translational Fidelity Score — bottom {top_n} / top {top_n} cell lines",
        xaxis_title="TFS (0-1)",
        yaxis_title="CCLE cell line",
        xaxis_range=[0, 1],
        template="plotly_white",
        width=800,
        height=max(400, 26 * len(ids_s)),
    )
    return fig


def lineage_domain_scatter_static(
    coords, lineage_labels, domain_labels, title, ax=None, lineage_order=None, idx_to_lineage=None
):
    """Static matplotlib version of `lineage_domain_scatter` (for PNG export)."""
    import matplotlib.pyplot as plt

    order = lineage_order if lineage_order is not None else LINEAGE_ORDER
    coords = np.asarray(coords)
    lin = _lineage_names(lineage_labels, idx_to_lineage)
    dom = _domain_names(domain_labels)

    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(8, 7))
    for lineage in order:
        for domain in ("TCGA", "CCLE"):
            mask = (lin == lineage) & (dom == domain)
            if not mask.any():
                continue
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                c=LINEAGE_COLORS[lineage],
                marker=_DOMAIN_MARKER[domain],
                s=70 if domain == "CCLE" else 14,
                alpha=0.95 if domain == "CCLE" else 0.4,
                edgecolors="black" if domain == "CCLE" else "none",
                linewidths=0.6,
                label=f"{lineage} · {domain}",
            )
    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(fontsize=8, framealpha=0.9)
    return ax.figure


def before_after_panel(
    raw_coords,
    raw_domain_labels,
    emb_coords,
    emb_lineage_labels,
    emb_domain_labels,
    raw_title="Before — PCA of raw expression (domain gap)",
    emb_title="After — UMAP of aligned embeddings (lineage clusters)",
    lineage_order=None,
    idx_to_lineage=None,
):
    """Two-panel matplotlib figure for Blog Post 2.

    Left: PCA of the raw (unaligned) pooled features coloured by *domain* — CCLE
    and TCGA form two separated clouds (the gap the project removes). Right: UMAP
    of the trained embeddings coloured by *lineage*, marker by domain — the three
    lineages cluster with cell lines sitting inside their patient cloud.
    """
    import matplotlib.pyplot as plt

    raw_coords = np.asarray(raw_coords)
    raw_dom = _domain_names(raw_domain_labels)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    domain_colors = {"CCLE": "#d62728", "TCGA": "#1f77b4"}
    for domain in ("TCGA", "CCLE"):
        mask = raw_dom == domain
        if not mask.any():
            continue
        axes[0].scatter(
            raw_coords[mask, 0],
            raw_coords[mask, 1],
            c=domain_colors[domain],
            marker=_DOMAIN_MARKER[domain],
            s=60 if domain == "CCLE" else 14,
            alpha=0.9 if domain == "CCLE" else 0.4,
            edgecolors="black" if domain == "CCLE" else "none",
            linewidths=0.6,
            label=domain,
        )
    axes[0].set_title(raw_title)
    axes[0].set_xlabel("PC-1")
    axes[0].set_ylabel("PC-2")
    axes[0].legend(fontsize=9, framealpha=0.9)

    lineage_domain_scatter_static(
        emb_coords,
        emb_lineage_labels,
        emb_domain_labels,
        emb_title,
        ax=axes[1],
        lineage_order=lineage_order,
        idx_to_lineage=idx_to_lineage,
    )
    fig.tight_layout()
    return fig


def purity_confounder_panel(
    projection,
    purity,
    domain_labels,
    knn_strata,
    reference_line=None,
    title="Tumour-purity confounder analysis",
):
    """Two-panel matplotlib figure for the Day 20 purity-confounder analysis.

    Left: each sample's projection onto the CCLE->TCGA domain axis plotted
    against its purity, marker = domain (CCLE cross / TCGA circle) -- shows
    how much of the residual domain gap purity explains. Right: cross-domain
    kNN@5 accuracy in the high- vs. low-purity TCGA strata (the
    ``pctrans.evaluation.confounders.purity_stratified_knn`` output), with an
    optional dashed ``reference_line`` (e.g. the unstratified overall
    accuracy or the random baseline).
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    dom = _domain_names(domain_labels)
    purity = np.asarray(purity, dtype=float)
    projection = np.asarray(projection, dtype=float)
    for domain in ("TCGA", "CCLE"):
        mask = dom == domain
        if not mask.any():
            continue
        axes[0].scatter(
            purity[mask],
            projection[mask],
            marker=_DOMAIN_MARKER[domain],
            s=50 if domain == "CCLE" else 14,
            alpha=0.9 if domain == "CCLE" else 0.4,
            edgecolors="black" if domain == "CCLE" else "none",
            linewidths=0.5,
            label=domain,
        )
    axes[0].set_xlabel("Tumour purity (1.0 = pure)")
    axes[0].set_ylabel("Projection onto CCLE→TCGA domain axis")
    axes[0].set_title("Domain axis vs. purity")
    axes[0].legend(fontsize=9)

    strata_labels, values, ns = [], [], []
    for name in ("high_purity", "low_purity"):
        entry = knn_strata.get(name, {})
        acc = entry.get("overall_accuracy")
        strata_labels.append(name.replace("_", " "))
        values.append(acc if acc is not None else 0.0)
        ns.append(entry.get("n", 0))
    bars = axes[1].bar(strata_labels, values, color=["#1f77b4", "#d62728"])
    for bar, n in zip(bars, ns):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"n={n}",
            ha="center",
            fontsize=9,
        )
    if reference_line is not None:
        axes[1].axhline(reference_line, color="black", linestyle="--", linewidth=1, label="reference")
        axes[1].legend(fontsize=9)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("kNN@5 accuracy")
    axes[1].set_title("Retrieval by purity stratum")

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def confusion_matrix_heatmap(confusion_matrix, labels, title="Confusion matrix", normalize=True):
    """Static matplotlib heatmap of a square confusion matrix (Day 19: 15-lineage read).

    Rows = true lineage, columns = predicted lineage (matches
    ``knn_accuracy_from_embeddings``'s convention). Row-normalised by default so
    lineages with very different anchor counts (e.g. READ N=13 vs. BRCA N=1215)
    are still visually comparable; pass ``normalize=False`` for raw counts.
    """
    import matplotlib.pyplot as plt

    cm = np.asarray(confusion_matrix, dtype=float)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)

    n = len(labels)
    fig, ax = plt.subplots(figsize=(0.55 * n + 3, 0.5 * n + 2.5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1.0 if normalize else cm.max())
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted lineage")
    ax.set_ylabel("True lineage")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fmt = "{:.2f}" if normalize else "{:.0f}"
    threshold = (1.0 if normalize else cm.max()) / 2
    for i in range(n):
        for j in range(n):
            val = cm[i, j]
            if val > 0:
                ax.text(
                    j, i, fmt.format(val), ha="center", va="center", fontsize=7,
                    color="white" if val > threshold else "black",
                )
    fig.tight_layout()
    return fig


_BRAF_COLORS = {"mutant": "#d62728", "WT": "#1f77b4"}


def _bootstrap_fit_band(x, y, n_boot=500, seed=0, alpha=0.05, n_grid=100):
    """OLS fit line + percentile bootstrap CI band, evaluated on a grid over ``x``."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)
    grid = np.linspace(x.min(), x.max(), n_grid) if n and x.max() > x.min() else np.zeros(n_grid)

    slope, intercept = np.polyfit(x, y, 1) if n >= 2 else (0.0, float(y.mean()) if n else 0.0)
    fit = slope * grid + intercept

    rng = np.random.default_rng(seed)
    preds = np.empty((n_boot, n_grid))
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xb, yb = x[idx], y[idx]
        if np.std(xb) == 0:
            preds[b] = np.mean(yb)
            continue
        s, i = np.polyfit(xb, yb, 1)
        preds[b] = s * grid + i
    lo = np.percentile(preds, 100.0 * alpha / 2.0, axis=0)
    hi = np.percentile(preds, 100.0 * (1.0 - alpha / 2.0), axis=0)
    return grid, fit, lo, hi


def braf_casestudy_panel(
    coords,
    braf_status,
    domain_labels,
    proximity,
    vemurafenib_auc,
    response_status,
    placement_result,
    response_result,
    drug_signal_result=None,
    title="BRAF / vemurafenib case study (Day 23)",
):
    """Static matplotlib figure for the Day 23 (+ Day 26) case study.

    Left: 2-D projection of the pooled SKCM cell-line + patient embeddings,
    coloured by BRAF status (mutant red / WT blue), marker = domain (cell
    line cross / patient circle) -- the Part-A placement question made
    visible. Middle: cell-line proximity to the BRAF-mutant-patient centroid
    vs. vemurafenib AUC, coloured by BRAF status, with an OLS fit + bootstrap
    CI band -- the Part-B response-link question. Right (only when
    ``drug_signal_result`` -- the Day 26 `drug_signal_retained` output -- is
    given): out-of-fold R^2 for predicting vemurafenib AUC from BRAF status
    alone vs. raw HVG expression vs. the 64-d embedding, so a reader can see
    at a glance whether alignment discarded drug-response signal or the
    Part-B proximity probe was simply the wrong readout.
    """
    import matplotlib.pyplot as plt

    n_panels = 3 if drug_signal_result is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 6))

    coords = np.asarray(coords)
    status = np.asarray(braf_status).astype(str)
    dom = _domain_names(domain_labels)
    domain_to_dom_name = {"patient": "TCGA", "cell_line": "CCLE"}
    for braf in ("mutant", "WT"):
        for domain in ("patient", "cell_line"):
            mask = (status == braf) & (dom == domain_to_dom_name[domain])
            if not mask.any():
                continue
            axes[0].scatter(
                coords[mask, 0],
                coords[mask, 1],
                c=_BRAF_COLORS[braf],
                marker="X" if domain == "cell_line" else "o",
                s=70 if domain == "cell_line" else 20,
                alpha=0.9 if domain == "cell_line" else 0.45,
                edgecolors="black" if domain == "cell_line" else "none",
                linewidths=0.6,
                label=f"{braf} · {'cell line' if domain == 'cell_line' else 'patient'}",
            )
    axes[0].set_title(
        f"Part A — placement (p={placement_result['p_value']:.3g}, "
        f"effect={placement_result['effect_size']:.2f})"
    )
    axes[0].set_xlabel("UMAP-1")
    axes[0].set_ylabel("UMAP-2")
    axes[0].legend(fontsize=8, framealpha=0.9)

    proximity = np.asarray(proximity, dtype=float)
    auc = np.asarray(vemurafenib_auc, dtype=float)
    resp_status = np.asarray(response_status).astype(str)
    grid, fit, lo, hi = _bootstrap_fit_band(proximity, auc)
    axes[1].fill_between(grid, lo, hi, color="#7f7f7f", alpha=0.25, label="bootstrap 95% CI")
    axes[1].plot(grid, fit, color="black", linewidth=1.5, label="OLS fit")
    for braf in ("mutant", "WT"):
        mask = resp_status == braf
        if not mask.any():
            continue
        axes[1].scatter(
            proximity[mask], auc[mask], c=_BRAF_COLORS[braf], s=45, alpha=0.85,
            edgecolors="black", linewidths=0.5, label=braf,
        )
    axes[1].set_title(
        f"Part B — response link (ρ={response_result['rho']:.2f}, "
        f"95% CI [{response_result['ci_low']:.2f}, {response_result['ci_high']:.2f}], "
        f"n={response_result['n']})"
    )
    axes[1].set_xlabel("Proximity to BRAF-mutant-patient centroid (−distance)")
    axes[1].set_ylabel("Vemurafenib AUC (lower = more sensitive)")
    axes[1].legend(fontsize=8, framealpha=0.9)

    if drug_signal_result is not None:
        blocks = ["braf_status", "raw_expression", "embedding"]
        labels = ["BRAF status\nalone", "raw HVG\nexpression", "64-d\nembedding"]
        r2 = [drug_signal_result[b]["r2"] for b in blocks]
        rho = [drug_signal_result[b]["rho"] for b in blocks]
        x = np.arange(len(blocks))
        width = 0.35
        axes[2].bar(x - width / 2, r2, width, color="#4c72b0", label="R²")
        axes[2].bar(x + width / 2, rho, width, color="#dd8452", label="Spearman ρ")
        axes[2].axhline(0.0, color="black", linewidth=0.8)
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(labels, fontsize=9)
        axes[2].set_title("Drug-signal retained (Day 26, within-CCLE CV)")
        axes[2].set_ylabel("Out-of-fold score")
        axes[2].legend(fontsize=8, framealpha=0.9)

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def braf_casestudy_panel_interactive(
    coords,
    braf_status,
    domain_labels,
    sample_ids,
    proximity,
    vemurafenib_auc,
    response_status,
    response_sample_ids,
):
    """Interactive plotly two-panel version of `braf_casestudy_panel` (for HTML export)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    coords = np.asarray(coords)
    status = np.asarray(braf_status).astype(str)
    dom = _domain_names(domain_labels)
    ids = np.asarray(sample_ids).astype(str)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Part A — placement", "Part B — response link"),
    )

    for braf in ("mutant", "WT"):
        for domain, symbol, size, opacity in (("TCGA", "circle", 7, 0.5), ("CCLE", "x", 11, 0.95)):
            mask = (status == braf) & (dom == domain)
            if not mask.any():
                continue
            fig.add_trace(
                go.Scatter(
                    x=coords[mask, 0], y=coords[mask, 1], mode="markers",
                    name=f"{braf} · {domain}",
                    marker={"color": _BRAF_COLORS[braf], "symbol": symbol, "size": size, "opacity": opacity},
                    text=ids[mask], hovertemplate="%{text}<br>%{customdata}<extra></extra>",
                    customdata=[f"{braf} · {domain}"] * int(mask.sum()),
                ),
                row=1, col=1,
            )

    proximity = np.asarray(proximity, dtype=float)
    auc = np.asarray(vemurafenib_auc, dtype=float)
    resp_status = np.asarray(response_status).astype(str)
    resp_ids = np.asarray(response_sample_ids).astype(str)
    for braf in ("mutant", "WT"):
        mask = resp_status == braf
        if not mask.any():
            continue
        fig.add_trace(
            go.Scatter(
                x=proximity[mask], y=auc[mask], mode="markers",
                name=f"{braf} (vemurafenib)",
                marker={"color": _BRAF_COLORS[braf], "size": 9, "opacity": 0.85},
                text=resp_ids[mask], hovertemplate="%{text}<extra></extra>",
            ),
            row=1, col=2,
        )

    fig.update_xaxes(title_text="UMAP-1", row=1, col=1)
    fig.update_yaxes(title_text="UMAP-2", row=1, col=1)
    fig.update_xaxes(title_text="Proximity to BRAF-mutant-patient centroid", row=1, col=2)
    fig.update_yaxes(title_text="Vemurafenib AUC (lower = more sensitive)", row=1, col=2)
    fig.update_layout(
        title="BRAF / vemurafenib case study (Day 23)",
        template="plotly_white", width=1300, height=650,
    )
    return fig


def permutation_null_panel(results, chance_level=None, title="Label-shuffle negative control"):
    """One histogram per Day 21 permutation-test variant, real value marked.

    ``results`` is ``{variant_name: permutation_test(...) output}`` -- each
    entry contributes one subplot: a histogram of that variant's
    ``null_values`` with a solid vertical line at ``real_value`` and, if
    given, a dashed line at ``chance_level`` (e.g. ``1 / n_lineages``). The
    subplot title carries the empirical p-value so the figure is
    self-contained without reading the companion JSON.
    """
    import matplotlib.pyplot as plt

    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.5), squeeze=False)
    axes = axes[0]

    for ax, (name, result) in zip(axes, results.items()):
        null_values = np.asarray(result["null_values"], dtype=float)
        ax.hist(null_values, bins=min(10, max(3, len(null_values) // 2)),
                color="#7f7f7f", edgecolor="white", alpha=0.85)
        ax.axvline(result["real_value"], color="#d62728", linewidth=2, label="real value")
        if chance_level is not None:
            ax.axvline(chance_level, color="black", linestyle="--", linewidth=1, label="chance (1/n)")
        ax.set_xlabel("kNN@5 accuracy")
        ax.set_ylabel("Permutations")
        ax.set_title(f"{name}\np = {result['p_value']:.4f}  (n={result['n_perm']})")
        ax.legend(fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def celligner_comparison_panel(results, title="Day 25 — Celligner head-to-head"):
    """Figure F7: one grouped bar chart per variant in `celligner_compare`'s output.

    ``results`` is the `reports/celligner_comparison.json` dict (one entry per
    ``"3-lineage"``/``"15-lineage"`` variant). Each subplot bars every method
    with a real number (random / PCA / Harmony / Celligner / supervised
    ceiling / contrastive), skipping any entry that is ``None`` (e.g.
    Celligner when the dependency is not installed) rather than plotting a
    fabricated zero.
    """
    import matplotlib.pyplot as plt

    _METHOD_LABELS = {
        "random": "Random",
        "pca_knn": "PCA+kNN",
        "harmony_knn": "Harmony",
        "celligner_knn5": "Celligner",
        "supervised_ceiling": "Supervised\nceiling",
        "contrastive_knn5": "Contrastive\n(ours)",
    }
    _METHOD_COLORS = {
        "random": "#7f7f7f",
        "pca_knn": "#aec7e8",
        "harmony_knn": "#ff7f0e",
        "celligner_knn5": "#9467bd",
        "supervised_ceiling": "#2ca02c",
        "contrastive_knn5": "#d62728",
    }

    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), squeeze=False)
    axes = axes[0]

    for ax, (name, res) in zip(axes, results.items()):
        values = dict(res.get("reference", {}))
        values["celligner_knn5"] = res.get("celligner_knn5")
        values["contrastive_knn5"] = res.get("contrastive_knn5")

        methods = [m for m in _METHOD_LABELS if values.get(m) is not None]
        heights = [values[m] * 100 for m in methods]
        colors = [_METHOD_COLORS[m] for m in methods]
        labels = [_METHOD_LABELS[m] for m in methods]

        bars = ax.bar(labels, heights, color=colors, edgecolor="white")
        for bar, h in zip(bars, heights):
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{h:.1f}%",
                    ha="center", va="bottom", fontsize=9)
        if not res.get("celligner_available", True):
            ax.text(0.5, 0.5, "Celligner: dep not installed", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8, color="#9467bd", style="italic")

        ax.set_ylabel("kNN@5 retrieval accuracy (%)")
        ax.set_ylim(0, 108)
        ax.set_title(f"{name} (n={res['n_lineages']} lineages)")

    fig.suptitle(title)
    fig.tight_layout()
    return fig
