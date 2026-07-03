"""``pctrans-visualize`` CLI: Day 11 UMAP + TFS figures.

Loads the trained dual-tower model and the held-out test split, embeds the CCLE
cell lines and TCGA patients, and produces the Day 11 deliverables:

  * ``data/processed/embeddings_test.npz`` — pooled test embeddings + ids/labels,
    consumed by the Day 12 Streamlit app (no model inference needed at serve time).
  * ``reports/umap_test_set.html`` / ``.png`` — interactive + static UMAP of the
    test embeddings, colour = lineage, marker = domain.
  * ``reports/umap_before_after.png`` — PCA of raw features (domain gap) beside
    the UMAP of aligned embeddings (lineage clusters), for Blog Post 2.
  * ``reports/tfs_ranking.html`` — top/bottom cell lines by TFS (from
    ``reports/eval_summary.json``, written by ``pctrans-evaluate``).

Per-lineage cluster tightness (mean intra-lineage cosine similarity) is printed so
the biological-interpretation questions in the daily report can be answered with
numbers rather than eyeballing the plot.
"""

import json
import pickle
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
import typer
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

matplotlib.use("Agg")  # headless PNG rendering, no display required

from pctrans.data.dataset import CCLEDataset, TCGADataset, build_lineage_maps  # noqa: E402
from pctrans.data.preprocessor import DataSplitter  # noqa: E402
from pctrans.evaluation.knn import embed_loader  # noqa: E402
from pctrans.evaluation.viz import (  # noqa: E402
    before_after_panel,
    lineage_domain_scatter,
    lineage_domain_scatter_static,
    tfs_ranking_bar,
    umap_projection,
)
from pctrans.scripts.evaluate import _build_model, _load_yaml  # noqa: E402

app = typer.Typer()

_DEFAULT_LINEAGES = ["LUAD", "BRCA", "SKCM"]


def _intra_lineage_cosine(z, y, idx_to_lineage):
    """Mean within-lineage cosine similarity per lineage (cluster tightness)."""
    z = np.asarray(z, dtype=np.float64)
    z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
    y = np.asarray(y)
    out = {}
    for idx, lineage in idx_to_lineage.items():
        mask = y == idx
        if mask.sum() < 2:
            continue
        sub = z[mask]
        sim = sub @ sub.T
        iu = np.triu_indices(len(sub), k=1)
        out[lineage] = float(sim[iu].mean())
    return out


@app.command()
def main(
    model: str = "models/best_model.pt",
    data_dir: str = "data/processed/",
    ccle_file: str = "ccle_2k.parquet",
    tcga_file: str = "tcga_2k.parquet",
    splits_file: str = "splits.json",
    scalers_file: str = "scalers.pkl",
    model_config: str = "configs/model.yaml",
    data_config: str = "configs/data.yaml",
    eval_summary: str = "reports/eval_summary.json",
    reports_dir: str = "reports/",
    output_suffix: str = "",
    seed: int = 42,
):
    """Render the Day 11 UMAP / before-after / TFS-ranking figures.

    ``--ccle-file/--tcga-file/--splits-file/--scalers-file/--data-config`` let this
    render an alternate artefact set (e.g. Day 18's ``*_15`` 15-lineage outputs)
    without touching the Phase-1 defaults. ``--output-suffix`` (e.g. ``_15``) keeps
    the written figures/embeddings alongside the Phase-1 ones instead of
    overwriting them.
    """
    data_dir = Path(data_dir)
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    model_cfg = _load_yaml(model_config)
    data_cfg = _load_yaml(data_config) if Path(data_config).exists() else {}
    lineages = data_cfg.get("lineages", _DEFAULT_LINEAGES)
    lineage_to_idx, idx_to_lineage = build_lineage_maps(lineages)

    ccle_df = pd.read_parquet(data_dir / ccle_file)
    tcga_df = pd.read_parquet(data_dir / tcga_file)
    with open(data_dir / splits_file, encoding="utf-8") as f:
        splits = json.load(f)
    with open(data_dir / scalers_file, "rb") as f:
        scalers = pickle.load(f)

    splitter = DataSplitter()

    def scaled(df, ids):
        return splitter.apply_scalers(df.loc[ids], scalers)

    ccle_test = CCLEDataset(scaled(ccle_df, splits["ccle"]["test"]), lineage_to_idx=lineage_to_idx)
    tcga_test = TCGADataset(scaled(tcga_df, splits["tcga"]["test"]), lineage_to_idx=lineage_to_idx)
    typer.echo(f"Test set: CCLE {len(ccle_test)} + TCGA {len(tcga_test)}")

    net = _build_model(model_cfg)
    checkpoint = torch.load(model, map_location="cpu", weights_only=False)
    net.load_state_dict(checkpoint["model_state_dict"])
    net.eval()

    ccle_loader = DataLoader(ccle_test, batch_size=256, shuffle=False)
    tcga_loader = DataLoader(tcga_test, batch_size=256, shuffle=False)
    z_ccle, y_ccle = embed_loader(net.encode_ccle, ccle_loader)
    z_tcga, y_tcga = embed_loader(net.encode_tcga, tcga_loader)
    ids_ccle = ccle_test.ids.astype(str)
    ids_tcga = tcga_test.ids.astype(str)

    # -- Save pooled embeddings for the Streamlit app -------------------------
    emb_path = data_dir / f"embeddings_test{output_suffix}.npz"
    np.savez(
        emb_path,
        z_ccle=z_ccle,
        y_ccle=y_ccle,
        ids_ccle=ids_ccle,
        z_tcga=z_tcga,
        y_tcga=y_tcga,
        ids_tcga=ids_tcga,
    )
    typer.echo(f"Wrote {emb_path}")

    # -- Per-cell-line TFS (from the Gate 1 summary) --------------------------
    tfs_by_id = {}
    if Path(eval_summary).exists():
        with open(eval_summary, encoding="utf-8") as f:
            summary = json.load(f)
        tfs_by_id = {r["id"]: r["tfs"] for r in summary.get("per_cell_line_tfs", [])}

    # -- Pool + UMAP ----------------------------------------------------------
    pooled_z = np.concatenate([z_ccle, z_tcga], axis=0)
    pooled_lineage = np.concatenate([y_ccle, y_tcga])
    pooled_domain = np.array([0] * len(y_ccle) + [1] * len(y_tcga))
    pooled_ids = np.concatenate([ids_ccle, ids_tcga])
    pooled_tfs = np.concatenate(
        [np.array([tfs_by_id.get(i, np.nan) for i in ids_ccle]), np.full(len(y_tcga), np.nan)]
    )

    coords = umap_projection(pooled_z, seed=seed)
    title = f"CCLE cell lines aligned to TCGA patients (test set, {len(lineages)}-lineage, 64-d embeddings)"
    fig = lineage_domain_scatter(
        coords,
        pooled_lineage,
        pooled_domain,
        title,
        sample_ids=pooled_ids,
        tfs_scores=pooled_tfs,
        lineage_order=lineages,
        idx_to_lineage=idx_to_lineage,
    )
    html_path = reports_dir / f"umap_test_set{output_suffix}.html"
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    typer.echo(f"Wrote {html_path}")

    static_fig = lineage_domain_scatter_static(
        coords, pooled_lineage, pooled_domain, title, lineage_order=lineages, idx_to_lineage=idx_to_lineage
    )
    png_path = reports_dir / f"umap_test_set{output_suffix}.png"
    static_fig.savefig(png_path, dpi=150, bbox_inches="tight")
    typer.echo(f"Wrote {png_path}")

    # -- Before/after panel: raw PCA (domain gap) vs UMAP embeddings ----------
    raw_pooled = np.concatenate(
        [ccle_test.features.numpy(), tcga_test.features.numpy()], axis=0
    )
    raw_coords = PCA(n_components=2, random_state=seed).fit_transform(raw_pooled)
    ba_fig = before_after_panel(
        raw_coords,
        pooled_domain,
        coords,
        pooled_lineage,
        pooled_domain,
        lineage_order=lineages,
        idx_to_lineage=idx_to_lineage,
    )
    ba_path = reports_dir / f"umap_before_after{output_suffix}.png"
    ba_fig.savefig(ba_path, dpi=150, bbox_inches="tight")
    typer.echo(f"Wrote {ba_path}")

    # -- TFS ranking bar ------------------------------------------------------
    if tfs_by_id:
        ids_arr = np.array(list(tfs_by_id.keys()))
        tfs_arr = np.array(list(tfs_by_id.values()))
        bar = tfs_ranking_bar(ids_arr, tfs_arr)
        bar_path = reports_dir / f"tfs_ranking{output_suffix}.html"
        bar.write_html(str(bar_path), include_plotlyjs="cdn")
        typer.echo(f"Wrote {bar_path}")

    # -- Biological-interpretation numbers ------------------------------------
    tightness = _intra_lineage_cosine(pooled_z, pooled_lineage, idx_to_lineage)
    typer.echo("")
    typer.echo("Per-lineage cluster tightness (mean within-lineage cosine, pooled):")
    for lineage in lineages:
        if lineage in tightness:
            typer.echo(f"  {lineage}: {tightness[lineage]:+.3f}")

    return {
        "coords_shape": tuple(coords.shape),
        "tightness": tightness,
        "n_ccle": len(y_ccle),
        "n_tcga": len(y_tcga),
    }


if __name__ == "__main__":
    app()
