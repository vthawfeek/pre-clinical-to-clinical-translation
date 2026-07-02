import typer

from pctrans.data.preprocessor import FeatureSynchroniser

app = typer.Typer()


@app.command()
def main(
    raw_dir: str = "data/raw/",
    out_dir: str = "data/processed/",
    n_hvgs: int = 2000,
    split: bool = False,
):
    """Synchronise CCLE + TCGA onto a shared top-`n_hvgs` HVG feature space."""
    if split:
        raise NotImplementedError("--split is implemented on Day 5 (DataSplitter)")

    fs = FeatureSynchroniser()

    ccle_expr, ccle_meta = fs.load_ccle(raw_dir)
    typer.echo(f"CCLE: {ccle_expr.shape[0]} cell lines x {ccle_expr.shape[1] - 1} genes")
    for lineage, count in ccle_expr["lineage"].value_counts().items():
        typer.echo(f"  {lineage}: {count} cell lines")

    tcga_expr, tcga_pheno = fs.load_tcga(raw_dir)
    typer.echo(f"TCGA: {tcga_expr.shape[0]} patients x {tcga_expr.shape[1] - 1} genes")
    for lineage, count in tcga_expr["lineage"].value_counts().items():
        typer.echo(f"  {lineage}: {count} patients")

    ccle_genes = [c for c in ccle_expr.columns if c != "lineage"]
    tcga_genes = [c for c in tcga_expr.columns if c != "lineage"]
    common_genes = fs.find_common_genes(ccle_genes, tcga_genes)
    typer.echo(f"Common genes (CCLE & TCGA): {len(common_genes)}")

    hvgs = fs.select_hvgs(ccle_expr, tcga_expr, common_genes, n_hvgs=n_hvgs)
    typer.echo(f"Selected top {len(hvgs)} HVGs (union-rank method)")
    typer.echo(f"Top 10 HVGs by mean rank: {hvgs[:10]}")

    fs.save_filtered(ccle_expr, tcga_expr, hvgs, out_dir)
    typer.echo(f"Saved ccle_2k.parquet, tcga_2k.parquet, gene_list.txt to {out_dir}")

    return {
        "ccle_shape": ccle_expr.shape,
        "tcga_shape": tcga_expr.shape,
        "n_common_genes": len(common_genes),
        "n_hvgs": len(hvgs),
        "top_10_hvgs": hvgs[:10],
    }


if __name__ == "__main__":
    app()
