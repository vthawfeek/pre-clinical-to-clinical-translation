"""Streamlit demo: where does a CCLE cell line land in human patient space?

Pick a CCLE cell line and the app answers the project's core question visually:
its position in a UMAP of TCGA patients, its Translational Fidelity Score, and its
five nearest human patients by cosine similarity in the 64-dim aligned manifold.

Everything is precomputed (Day 11 ``embeddings_test.npz`` + Day 12 ``app_meta.json``
/ ``ccle_embeddings.npz``); the app never loads the model or the raw expression
files, so it fits inside Streamlit Cloud's free-tier memory. The UMAP, neighbours,
and TFS gauge all run on the *held-out test set* — the honest, evaluated subset
with Gate 1 TFS scores.

All data/plot logic lives in module-level pure functions so it can be imported and
smoke-tested without a Streamlit runtime; the UI in ``main()`` is guarded by
``if __name__ == "__main__"`` (true under ``streamlit run``, false on import).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pctrans.data.dataset import IDX_TO_LINEAGE
from pctrans.evaluation.viz import LINEAGE_COLORS, LINEAGE_ORDER, umap_projection

DATA_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")

REPO_URL = "https://github.com/vthawfeek/pre-clinical-to-clinical-translation"

# TFS colour bands (plan Day 12: green > 0.70, yellow 0.50-0.70, red < 0.50).
_TFS_GREEN, _TFS_YELLOW, _TFS_RED, _TFS_GREY = "#2ca02c", "#f0a500", "#d62728", "#7f7f7f"
# Domain 0 = CCLE cell line, 1 = TCGA patient (matches the pooled embedding order).
_CCLE, _TCGA = 0, 1


def tfs_color(value):
    """Colour for a TFS value on the plan's fidelity bands."""
    if value is None or not np.isfinite(value):
        return _TFS_GREY
    if value >= 0.70:
        return _TFS_GREEN
    if value >= 0.50:
        return _TFS_YELLOW
    return _TFS_RED


def tfs_band_label(value):
    """Human-readable fidelity band for a TFS value."""
    if value is None or not np.isfinite(value):
        return "not evaluated"
    if value >= 0.70:
        return "High fidelity"
    if value >= 0.50:
        return "Moderate fidelity"
    return "Poor fidelity"


def load_bundle(data_dir=DATA_DIR, reports_dir=REPORTS_DIR):
    """Load every precomputed artefact the app needs into one dict.

    Reads the pooled test embeddings, the Gate 1 per-cell-line TFS, and the
    deploy-safe metadata (cell-line names + TCGA clinical annotation). Raises
    ``FileNotFoundError`` with a runnable hint if the embeddings are missing.
    """
    data_dir, reports_dir = Path(data_dir), Path(reports_dir)
    emb_path = data_dir / "embeddings_test.npz"
    if not emb_path.exists():
        raise FileNotFoundError(
            f"{emb_path} not found - run `pctrans-visualize` (Day 11) then "
            "`pctrans-precompute` (Day 12) to build the app artefacts."
        )
    with np.load(emb_path, allow_pickle=True) as d:
        z_ccle = d["z_ccle"].astype(np.float32)
        y_ccle = d["y_ccle"].astype(int)
        ids_ccle = d["ids_ccle"].astype(str)
        z_tcga = d["z_tcga"].astype(np.float32)
        y_tcga = d["y_tcga"].astype(int)
        ids_tcga = d["ids_tcga"].astype(str)

    tfs_by_id = {}
    summary_path = reports_dir / "eval_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        tfs_by_id = {r["id"]: float(r["tfs"]) for r in summary.get("per_cell_line_tfs", [])}

    ccle_names, tcga_meta = {}, {}
    meta_path = data_dir / "app_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        ccle_names = meta.get("ccle_names", {})
        tcga_meta = meta.get("tcga_meta", {})

    return {
        "z_ccle": z_ccle,
        "y_ccle": y_ccle,
        "ids_ccle": ids_ccle,
        "z_tcga": z_tcga,
        "y_tcga": y_tcga,
        "ids_tcga": ids_tcga,
        "tfs_by_id": tfs_by_id,
        "ccle_names": ccle_names,
        "tcga_meta": tcga_meta,
    }


def pooled_umap(bundle, seed=42):
    """Fit UMAP on the pooled test embeddings (CCLE first, then TCGA).

    Returns ``coords`` aligned so ``coords[:n_ccle]`` are the cell lines and
    ``coords[n_ccle:]`` the patients, matching the arrays in ``bundle``.
    """
    pooled = np.concatenate([bundle["z_ccle"], bundle["z_tcga"]], axis=0)
    return umap_projection(pooled, seed=seed)


def cell_line_options(bundle):
    """Ordered (label, ccle_id) pairs for the dropdown, grouped by lineage.

    Sorted by lineage (LUAD, BRCA, SKCM) then display name so the selectbox reads
    as three visually grouped blocks. Labels carry the lineage, display name, and
    ModelID (e.g. ``LUAD - NCIH1975 (ACH-000021)``).
    """
    ids, y = bundle["ids_ccle"], bundle["y_ccle"]
    names = bundle["ccle_names"]
    rows = []
    for cid, code in zip(ids, y):
        lineage = IDX_TO_LINEAGE[int(code)]
        name = names.get(cid, cid)
        rows.append((LINEAGE_ORDER.index(lineage), lineage, name, cid))
    rows.sort(key=lambda r: (r[0], r[2].upper()))
    return [(f"{lineage} - {name} ({cid})", cid) for _, lineage, name, cid in rows]


def nearest_patients(bundle, ccle_id, k=5):
    """The ``k`` nearest TCGA test patients to ``ccle_id`` by cosine similarity.

    Embeddings are already L2-normalised, so cosine similarity is the dot product.
    Returns a list of dicts (rank, id, lineage, cosine, stage, histology, coords
    index into the pooled UMAP) ordered most-similar first.
    """
    idx = int(np.where(bundle["ids_ccle"] == ccle_id)[0][0])
    z_i = bundle["z_ccle"][idx]
    sims = bundle["z_tcga"] @ z_i
    k = min(k, len(sims))
    top = np.argsort(-sims)[:k]
    n_ccle = len(bundle["ids_ccle"])
    out = []
    for rank, j in enumerate(top, start=1):
        tcga_id = bundle["ids_tcga"][j]
        meta = bundle["tcga_meta"].get(tcga_id, {})
        out.append(
            {
                "rank": rank,
                "id": tcga_id,
                "lineage": IDX_TO_LINEAGE[int(bundle["y_tcga"][j])],
                "cosine": float(sims[j]),
                "stage": meta.get("stage", ""),
                "histology": meta.get("histology", ""),
                "coord_idx": n_ccle + int(j),
            }
        )
    return out


def neighbours_dataframe(neighbours):
    """Format `nearest_patients` output as the display table."""
    return pd.DataFrame(
        [
            {
                "Rank": n["rank"],
                "TCGA sample": n["id"],
                "Lineage": n["lineage"],
                "Cosine similarity": round(n["cosine"], 3),
                "Tumour stage": n["stage"] or "-",
                "Histology": n["histology"] or "-",
            }
            for n in neighbours
        ]
    )


def build_umap_figure(bundle, coords, ccle_id, neighbours):
    """Interactive UMAP: full test set dimmed, selected cell line + neighbours lit.

    Base layer = every test point coloured by lineage, marker by domain, faded.
    The selected cell line is drawn as a gold-ringed star; its ``k`` nearest
    patients as ringed hexagons. Hover carries the sample id, lineage/domain, and
    (for the cell line) its TFS.
    """
    import plotly.graph_objects as go

    n_ccle = len(bundle["ids_ccle"])
    lineage_codes = np.concatenate([bundle["y_ccle"], bundle["y_tcga"]])
    domain = np.array([_CCLE] * n_ccle + [_TCGA] * len(bundle["ids_tcga"]))
    ids = np.concatenate([bundle["ids_ccle"], bundle["ids_tcga"]])
    names = np.array([bundle["ccle_names"].get(i, i) for i in ids])

    fig = go.Figure()

    # -- Base layer: dimmed context, one trace per (lineage, domain) ----------
    for lineage in LINEAGE_ORDER:
        for dom, symbol, size in ((_TCGA, "circle", 6), (_CCLE, "x", 9)):
            mask = (lineage_codes == LINEAGE_ORDER.index(lineage)) & (domain == dom)
            if not mask.any():
                continue
            fig.add_trace(
                go.Scatter(
                    x=coords[mask, 0],
                    y=coords[mask, 1],
                    mode="markers",
                    name=f"{lineage} - {'CCLE' if dom == _CCLE else 'TCGA'}",
                    marker={
                        "color": LINEAGE_COLORS[lineage],
                        "symbol": symbol,
                        "size": size,
                        "opacity": 0.28,
                        "line": {"width": 0},
                    },
                    customdata=np.stack([ids[mask], names[mask]], axis=1),
                    hovertemplate="%{customdata[1]} (%{customdata[0]})<br>"
                    f"{lineage}<extra></extra>",
                )
            )

    # -- Nearest patients: ringed hexagons ------------------------------------
    if neighbours:
        nb_idx = np.array([n["coord_idx"] for n in neighbours])
        nb_lineage = [n["lineage"] for n in neighbours]
        fig.add_trace(
            go.Scatter(
                x=coords[nb_idx, 0],
                y=coords[nb_idx, 1],
                mode="markers",
                name="Nearest 5 patients",
                marker={
                    "color": [LINEAGE_COLORS[lin] for lin in nb_lineage],
                    "symbol": "hexagon",
                    "size": 16,
                    "opacity": 0.95,
                    "line": {"width": 2, "color": "#111111"},
                },
                customdata=np.array(
                    [[n["id"], n["cosine"]] for n in neighbours], dtype=object
                ),
                hovertemplate="%{customdata[0]}<br>cosine %{customdata[1]:.3f}"
                "<extra>nearest patient</extra>",
            )
        )

    # -- Selected cell line: gold-ringed star ---------------------------------
    sel = int(np.where(bundle["ids_ccle"] == ccle_id)[0][0])
    lineage = IDX_TO_LINEAGE[int(bundle["y_ccle"][sel])]
    fig.add_trace(
        go.Scatter(
            x=[coords[sel, 0]],
            y=[coords[sel, 1]],
            mode="markers",
            name="Selected cell line",
            marker={
                "color": LINEAGE_COLORS[lineage],
                "symbol": "star",
                "size": 26,
                "opacity": 1.0,
                "line": {"width": 2.5, "color": "#d4af37"},
            },
            customdata=[[bundle["ccle_names"].get(ccle_id, ccle_id), ccle_id]],
            hovertemplate="%{customdata[0]} (%{customdata[1]})<br>"
            f"{lineage} - selected<extra></extra>",
        )
    )

    fig.update_layout(
        title="Test-set manifold: selected cell line (star) among TCGA patients",
        xaxis_title="UMAP-1",
        yaxis_title="UMAP-2",
        template="plotly_white",
        legend_title="Lineage - domain",
        height=620,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )
    return fig


def build_tfs_gauge(tfs):
    """TFS dial (0-1) coloured by fidelity band, with a threshold at 0.70."""
    import plotly.graph_objects as go

    value = tfs if (tfs is not None and np.isfinite(tfs)) else 0.0
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"valueformat": ".3f"},
            title={"text": "Translational Fidelity Score"},
            gauge={
                "axis": {"range": [0, 1]},
                "bar": {"color": tfs_color(tfs)},
                "steps": [
                    {"range": [0.0, 0.50], "color": "#f8d7da"},
                    {"range": [0.50, 0.70], "color": "#fff3cd"},
                    {"range": [0.70, 1.0], "color": "#d4edda"},
                ],
                "threshold": {
                    "line": {"color": "#333333", "width": 3},
                    "thickness": 0.75,
                    "value": 0.70,
                },
            },
        )
    )
    fig.update_layout(height=300, margin={"l": 20, "r": 20, "t": 50, "b": 10})
    return fig


def main():
    """Render the Streamlit UI (runs under ``streamlit run``, not on import)."""
    import streamlit as st

    st.set_page_config(page_title="Pre-Clinical to Clinical Translation", layout="wide")

    @st.cache_data(show_spinner=False)
    def _bundle():
        return load_bundle()

    @st.cache_resource(show_spinner="Fitting UMAP on the test manifold...")
    def _coords():
        return pooled_umap(_bundle())

    try:
        bundle = _bundle()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    coords = _coords()
    options = cell_line_options(bundle)

    st.title("Pre-Clinical to Clinical Translation")
    st.caption(
        "Where does a CCLE cancer cell line land in human (TCGA) patient space? "
        "Dual-tower InfoNCE contrastive alignment, held-out test set."
    )

    with st.sidebar:
        st.header("Select a CCLE cell line")
        label = st.selectbox(
            "Grouped by lineage (LUAD, BRCA, SKCM)",
            options=[lbl for lbl, _ in options],
            index=0,
        )
        ccle_id = dict(options)[label]
        st.markdown(f"**ModelID:** `{ccle_id}`")
        st.markdown(
            "TCGA patients = circles, CCLE cell lines = crosses. "
            "The selected cell line is a star; its 5 nearest patients are hexagons."
        )

    tfs = bundle["tfs_by_id"].get(ccle_id)
    neighbours = nearest_patients(bundle, ccle_id, k=5)

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Live UMAP")
        st.plotly_chart(
            build_umap_figure(bundle, coords, ccle_id, neighbours),
            width="stretch",
        )
    with right:
        st.subheader("Translational Fidelity")
        st.plotly_chart(build_tfs_gauge(tfs), width="stretch")
        band = tfs_band_label(tfs)
        tfs_txt = f"{tfs:.3f}" if tfs is not None else "not evaluated"
        st.markdown(
            f"**{band}** (TFS = {tfs_txt}). TFS measures how faithfully this cell "
            "line maps to its human counterpart in the 64-dimensional latent space. "
            "Green > 0.70, yellow 0.50-0.70, red < 0.50."
        )

    st.subheader("Nearest human patient neighbours")
    st.dataframe(
        neighbours_dataframe(neighbours), hide_index=True, width="stretch"
    )

    st.divider()
    st.markdown(
        f"[GitHub]({REPO_URL})  |  Model: dual-tower MLP "
        "[2000 -> 1024 -> 512 -> 256 -> 128 -> 64] x 2, L2-normalised, "
        "SupCon-InfoNCE loss with learnable temperature.  |  "
        "Method: contrastive alignment of CCLE cell lines to TCGA patients "
        "over 2,000 highly variable genes."
    )


if __name__ == "__main__":
    main()
