"""``pctrans-confounders`` CLI: the Day 20 tumour-purity confounder analysis.

Loads the Day 11 test embeddings (`data/processed/embeddings_test.npz`) and
TCGA ABSOLUTE purity calls, then runs the three Day 20 analyses: does the
CCLE->TCGA domain axis track purity, does cross-domain retrieval hold in both
high- and low-purity TCGA strata, and does lineage cohesion survive
regressing purity out of the pooled embeddings. Writes
`reports/confounder_purity.json` + `reports/confounder_purity.png`.
"""

import json
from pathlib import Path

import matplotlib
import numpy as np
import typer

matplotlib.use("Agg")  # headless PNG rendering, no display required

from pctrans.data.dataset import build_lineage_maps  # noqa: E402
from pctrans.data.tcga_client import TCGAClient  # noqa: E402
from pctrans.evaluation.confounders import (  # noqa: E402
    domain_axis_purity_correlation,
    load_purity,
    purity_residualised_silhouette,
    purity_stratified_knn,
)
from pctrans.evaluation.knn import knn_accuracy_from_embeddings  # noqa: E402
from pctrans.evaluation.silhouette import cross_domain_silhouette  # noqa: E402
from pctrans.evaluation.viz import purity_confounder_panel  # noqa: E402
from pctrans.scripts.evaluate import _load_yaml  # noqa: E402

app = typer.Typer()

_DEFAULT_LINEAGES = ["LUAD", "BRCA", "SKCM"]


@app.command()
def main(
    embeddings: str = "data/processed/embeddings_test.npz",
    purity_dir: str = "data/raw/tcga/",
    data_config: str = "configs/data.yaml",
    k: int = 5,
    output: str = "reports/confounder_purity.json",
    figure: str = "reports/confounder_purity.png",
):
    """Run the Day 20 purity-confounder analysis on the held-out test embeddings."""
    data_cfg = _load_yaml(data_config) if Path(data_config).exists() else {}
    lineages = data_cfg.get("lineages", _DEFAULT_LINEAGES)
    _, idx_to_lineage = build_lineage_maps(lineages)

    emb_path = Path(embeddings)
    data = np.load(emb_path, allow_pickle=True)
    z_ccle, y_ccle, ids_ccle = data["z_ccle"], data["y_ccle"], data["ids_ccle"]
    z_tcga, y_tcga, ids_tcga = data["z_tcga"], data["y_tcga"], data["ids_tcga"]
    typer.echo(f"Loaded {emb_path}: CCLE {len(ids_ccle)} + TCGA {len(ids_tcga)}")

    purity_path = TCGAClient().download_purity(purity_dir)
    purity_ccle, purity_tcga = load_purity(purity_path, ids_ccle, ids_tcga)
    n_with_purity = int(np.isfinite(purity_tcga).sum())
    typer.echo(
        f"TCGA patients with an ABSOLUTE purity call: {n_with_purity}/{len(ids_tcga)}"
    )

    # (a) domain axis vs. purity.
    axis_corr = domain_axis_purity_correlation(z_ccle, z_tcga, purity_ccle, purity_tcga)

    # (b) purity-stratified retrieval.
    strata = purity_stratified_knn(
        z_ccle, y_ccle, z_tcga, y_tcga, purity_tcga, k=k,
        idx_to_lineage=idx_to_lineage, lineage_order=lineages,
    )
    overall = knn_accuracy_from_embeddings(
        z_ccle, y_ccle, z_tcga, y_tcga, k=k, idx_to_lineage=idx_to_lineage, lineage_order=lineages
    )["overall_accuracy"]

    # (c) purity-residualised silhouette.
    silhouette_before = cross_domain_silhouette(
        np.concatenate([z_ccle, z_tcga], axis=0), np.concatenate([y_ccle, y_tcga])
    )
    silhouette_after = purity_residualised_silhouette(
        z_ccle, y_ccle, z_tcga, y_tcga, purity_ccle, purity_tcga
    )

    bar = "=" * 52
    typer.echo("")
    typer.echo(bar)
    typer.echo("       DAY 20 — TUMOUR-PURITY CONFOUNDER REPORT")
    typer.echo(bar)
    typer.echo(
        f"(a) corr(domain-axis projection, purity): r = {axis_corr['r']:+.3f}  (n={axis_corr['n']})"
    )
    typer.echo(f"(b) Overall kNN@{k} (unstratified): {overall * 100:5.1f}%")
    for name in ("high_purity", "low_purity"):
        entry = strata[name]
        shown = (
            f"{entry['overall_accuracy'] * 100:5.1f}%"
            if entry["overall_accuracy"] is not None
            else "  n/a"
        )
        typer.echo(f"    {name:12s}: {shown}   (n={entry['n']})")
    typer.echo(
        f"(c) Silhouette before purity residualisation: {silhouette_before:+.3f}"
    )
    typer.echo(
        f"    Silhouette after purity residualisation:  {silhouette_after:+.3f}"
    )
    typer.echo(bar)

    # -- Figure: domain-axis-vs-purity scatter + stratified-retrieval bars ----
    z_pooled = np.concatenate([z_ccle, z_tcga], axis=0)
    purity_pooled = np.concatenate([purity_ccle, purity_tcga])
    domain_pooled = np.array([0] * len(z_ccle) + [1] * len(z_tcga))
    axis = z_tcga.mean(axis=0) - z_ccle.mean(axis=0)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    projection = z_pooled @ axis
    finite = np.isfinite(purity_pooled)
    fig = purity_confounder_panel(
        projection[finite],
        purity_pooled[finite],
        domain_pooled[finite],
        strata,
        reference_line=overall,
        title="Day 20 — tumour-purity confounder analysis",
    )
    fig_path = Path(figure)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    typer.echo(f"Wrote {fig_path}")

    summary = {
        "embeddings": str(emb_path),
        "purity_table": str(purity_path),
        "n_tcga_with_purity": n_with_purity,
        "n_tcga_total": int(len(ids_tcga)),
        "domain_axis_purity_correlation": axis_corr,
        "overall_knn_accuracy": overall,
        "purity_stratified_knn": strata,
        "silhouette_before_residualisation": silhouette_before,
        "silhouette_after_residualisation": silhouette_after,
    }
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    typer.echo(f"Wrote {out_path}")
    return summary


if __name__ == "__main__":
    app()
