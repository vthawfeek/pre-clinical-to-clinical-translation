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

# Lineage colours (plan: LUAD=blue, BRCA=pink, SKCM=brown).
LINEAGE_COLORS = {"LUAD": "#1f77b4", "BRCA": "#e377c2", "SKCM": "#8c564b"}
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


def _lineage_names(labels):
    """Normalise lineage labels (int codes or strings) to lineage-name strings."""
    labels = np.asarray(labels)
    if labels.dtype.kind in "iu":
        return np.array([IDX_TO_LINEAGE[int(v)] for v in labels])
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
    coords, lineage_labels, domain_labels, title, sample_ids=None, tfs_scores=None
):
    """Interactive plotly scatter, colour = lineage, marker = domain.

    Hover shows the sample ID, its lineage/domain, and (for CCLE cell lines) its
    TFS. One trace per (lineage, domain) pair so the legend toggles cleanly.
    """
    import plotly.graph_objects as go

    coords = np.asarray(coords)
    lin = _lineage_names(lineage_labels)
    dom = _domain_names(domain_labels)
    n = len(coords)
    ids = np.asarray(sample_ids) if sample_ids is not None else np.array([""] * n)
    tfs = (
        np.asarray(tfs_scores, dtype=float)
        if tfs_scores is not None
        else np.full(n, np.nan)
    )

    fig = go.Figure()
    for lineage in LINEAGE_ORDER:
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


def lineage_domain_scatter_static(coords, lineage_labels, domain_labels, title, ax=None):
    """Static matplotlib version of `lineage_domain_scatter` (for PNG export)."""
    import matplotlib.pyplot as plt

    coords = np.asarray(coords)
    lin = _lineage_names(lineage_labels)
    dom = _domain_names(domain_labels)

    created = ax is None
    if created:
        _, ax = plt.subplots(figsize=(8, 7))
    for lineage in LINEAGE_ORDER:
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
        emb_coords, emb_lineage_labels, emb_domain_labels, emb_title, ax=axes[1]
    )
    fig.tight_layout()
    return fig
