"""``pctrans-celligner-compare`` CLI: the Day 25 Celligner head-to-head.

Runs the Day 25 prior-art benchmark on both the 3-lineage and 15-lineage test
sets: attempt Celligner (Warren et al., Nat. Commun. 2021) on the same raw
HVG-filtered CCLE+TCGA matrices used elsewhere in this project, score its
aligned embedding with the identical cross-domain kNN@k + silhouette metric,
and place it next to our already-computed contrastive/Harmony/supervised-
ceiling numbers (`reports/eval_summary*.json`, `reports/baselines.json`).
Writes `reports/celligner_comparison.json`.

`celligner` is not resolvable by pip/uv at all (see
`pctrans/evaluation/celligner_compare.py`) without a hand-built install, so on
a stock environment this reports the gap honestly -- `celligner: n/a (dep not
installed)` -- alongside the real numbers for every other method, the same
pattern Day 17 used for ComBat/Scanorama.
"""

import json
from pathlib import Path

import matplotlib
import pandas as pd
import typer

matplotlib.use("Agg")  # headless PNG rendering, no display required

from pctrans.data.dataset import build_lineage_maps  # noqa: E402
from pctrans.evaluation.celligner_compare import retrieval_on_embedding, run_celligner  # noqa: E402
from pctrans.evaluation.viz import celligner_comparison_panel  # noqa: E402
from pctrans.scripts.evaluate import _load_yaml  # noqa: E402

app = typer.Typer()

_VARIANTS = {
    "3-lineage": {
        "ccle_file": "ccle_2k.parquet",
        "tcga_file": "tcga_2k.parquet",
        "splits_file": "splits.json",
        "data_config": "configs/data.yaml",
        "eval_summary": "reports/eval_summary.json",
        "baselines": "reports/baselines.json",
    },
    "15-lineage": {
        "ccle_file": "ccle_2k_15.parquet",
        "tcga_file": "tcga_2k_15.parquet",
        "splits_file": "splits_15.json",
        "data_config": "configs/data_15.yaml",
        "eval_summary": "reports/eval_summary_15.json",
        "baselines": None,  # Day 17 real-baseline sweep was 3-lineage only.
    },
}


def _load_json(path):
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _run_variant(name, cfg, data_dir, k):
    ccle_df = pd.read_parquet(data_dir / cfg["ccle_file"])
    tcga_df = pd.read_parquet(data_dir / cfg["tcga_file"])
    with open(data_dir / cfg["splits_file"], encoding="utf-8") as f:
        splits = json.load(f)
    data_cfg = _load_yaml(cfg["data_config"]) if Path(cfg["data_config"]).exists() else {}
    lineages = data_cfg.get("lineages", ["LUAD", "BRCA", "SKCM"])
    lineage_to_idx, idx_to_lineage = build_lineage_maps(lineages)

    ccle_ids = splits["ccle"]["test"]
    tcga_ids = splits["tcga"]["test"]
    ccle_test = ccle_df.loc[ccle_ids]
    tcga_test = tcga_df.loc[tcga_ids]
    gene_cols = [c for c in ccle_df.columns if c != "lineage"]

    id_to_lineage_idx = {}
    id_to_lineage_idx.update(
        {i: lineage_to_idx[lin] for i, lin in ccle_test["lineage"].items()}
    )
    id_to_lineage_idx.update(
        {i: lineage_to_idx[lin] for i, lin in tcga_test["lineage"].items()}
    )

    joint_emb = run_celligner(ccle_test[gene_cols].to_numpy(), tcga_test[gene_cols].to_numpy())

    celligner_result = None
    if joint_emb is not None:
        celligner_result = retrieval_on_embedding(
            joint_emb,
            ccle_ids,
            tcga_ids,
            id_to_lineage_idx,
            k=k,
            idx_to_lineage=idx_to_lineage,
            lineage_order=lineages,
        )

    eval_summary = _load_json(cfg["eval_summary"])
    contrastive = eval_summary.get("overall_knn_accuracy")
    contrastive_ci = eval_summary.get("knn_wilson_ci")

    reference = {"random": 1.0 / len(lineages) if lineages else None}
    if cfg["baselines"] is not None:
        base = _load_json(cfg["baselines"])
        reference.update(
            {
                "pca_knn": base.get("pca_knn"),
                "harmony_knn": base.get("harmony_knn"),
                "supervised_ceiling": base.get("supervised_ceiling"),
            }
        )
    elif eval_summary.get("baselines"):
        reference["pca_knn"] = eval_summary["baselines"].get("pca_knn")

    return {
        "n_lineages": len(lineages),
        "test_sizes": {"ccle": len(ccle_ids), "tcga": len(tcga_ids)},
        "celligner_knn5": celligner_result["overall_accuracy"] if celligner_result else None,
        "celligner_silhouette": celligner_result["silhouette"] if celligner_result else None,
        "celligner_per_lineage": celligner_result["per_lineage"] if celligner_result else None,
        "celligner_available": celligner_result is not None,
        "contrastive_knn5": contrastive,
        "contrastive_wilson_ci": contrastive_ci,
        "reference": reference,
    }


@app.command()
def main(
    data_dir: str = "data/processed/",
    k: int = 5,
    output: str = "reports/celligner_comparison.json",
    figure: str = "reports/celligner_comparison.png",
):
    """Run the Day 25 Celligner head-to-head on both lineage variants; write `output`."""
    data_dir = Path(data_dir)
    results = {name: _run_variant(name, cfg, data_dir, k) for name, cfg in _VARIANTS.items()}

    bar = "=" * 58
    typer.echo("")
    typer.echo(bar)
    typer.echo("      DAY 25 — CELLIGNER HEAD-TO-HEAD (identical kNN@{} metric)".format(k))
    typer.echo(bar)
    for name, res in results.items():
        typer.echo(f"\n{name} (CCLE {res['test_sizes']['ccle']} + TCGA {res['test_sizes']['tcga']}):")
        ref = res["reference"]
        if ref.get("random") is not None:
            typer.echo(f"  Random               {ref['random'] * 100:5.1f}%")
        if ref.get("pca_knn") is not None:
            typer.echo(f"  PCA+kNN              {ref['pca_knn'] * 100:5.1f}%")
        if ref.get("harmony_knn") is not None:
            typer.echo(f"  Harmony+kNN          {ref['harmony_knn'] * 100:5.1f}%")
        if res["celligner_available"]:
            typer.echo(f"  Celligner+kNN        {res['celligner_knn5'] * 100:5.1f}%")
        else:
            typer.echo("  Celligner+kNN          n/a (dep not installed -- see module docstring)")
        if ref.get("supervised_ceiling") is not None:
            typer.echo(f"  Supervised ceiling   {ref['supervised_ceiling'] * 100:5.1f}%")
        if res["contrastive_knn5"] is not None:
            ci = res["contrastive_wilson_ci"]
            ci_str = f"  (Wilson {ci['low'] * 100:.1f}-{ci['high'] * 100:.1f}%)" if ci else ""
            typer.echo(f"  Contrastive (ours)   {res['contrastive_knn5'] * 100:5.1f}%{ci_str}")
    typer.echo("")
    typer.echo(bar)

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    typer.echo(f"Wrote {out_path}")

    fig = celligner_comparison_panel(results)
    fig_path = Path(figure)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    typer.echo(f"Wrote {fig_path}")

    return results


if __name__ == "__main__":
    app()
