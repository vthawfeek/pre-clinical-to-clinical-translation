import typer

from pctrans.data.preprocessor import DataSplitter, FeatureSynchroniser

app = typer.Typer()


@app.command()
def main(
    raw_dir: str = "data/raw/",
    out_dir: str = "data/processed/",
    n_hvgs: int = 2000,
    split: bool = False,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
):
    """Synchronise CCLE + TCGA onto a shared top-`n_hvgs` HVG feature space.

    With ``--split``, additionally produce lineage-stratified train/val/test
    splits (``splits.json``) and per-gene z-score scalers fit on the pooled
    training expression only (``scalers.pkl``).
    """
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

    result = {
        "ccle_shape": ccle_expr.shape,
        "tcga_shape": tcga_expr.shape,
        "n_common_genes": len(common_genes),
        "n_hvgs": len(hvgs),
        "top_10_hvgs": hvgs[:10],
    }

    if split:
        keep_cols = [*hvgs, "lineage"]
        ccle_hvg = ccle_expr[keep_cols]
        tcga_hvg = tcga_expr[keep_cols]

        splitter = DataSplitter()
        splits = splitter.stratified_split(
            ccle_hvg, tcga_hvg, val_frac=val_frac, test_frac=test_frac, seed=seed
        )
        ccle_train = ccle_hvg.loc[splits["ccle"]["train"]]
        tcga_train = tcga_hvg.loc[splits["tcga"]["train"]]
        scalers = splitter.fit_scalers(ccle_train, tcga_train)
        splitter.save_splits(splits, scalers, out_dir)

        typer.echo("Stratified split (train / val / test):")
        for domain in ("ccle", "tcga"):
            counts = {k: len(v) for k, v in splits[domain].items()}
            typer.echo(
                f"  {domain.upper()}: "
                f"train {counts['train']} | val {counts['val']} | test {counts['test']}"
            )
        typer.echo(
            f"Scalers fit on {len(ccle_train) + len(tcga_train)} pooled train samples "
            f"({len(scalers['feature_cols'])} genes)"
        )
        typer.echo(f"Saved splits.json, scalers.pkl to {out_dir}")

        result["splits"] = {
            domain: {k: len(v) for k, v in splits[domain].items()}
            for domain in ("ccle", "tcga")
        }

    return result


if __name__ == "__main__":
    app()
