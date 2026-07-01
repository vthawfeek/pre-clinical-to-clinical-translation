import pandas as pd
import typer

from pctrans.data.ccle_client import CCLEClient, filter_lineages

app = typer.Typer()

LINEAGES = ["LUAD", "BRCA", "SKCM"]


@app.command()
def ccle(out_dir: str = "data/raw/ccle/"):
    """Download DepMap 24Q4 CCLE expression matrix + Model.csv, report lineage counts."""
    client = CCLEClient()
    expr_path = client.download_expression(out_dir)
    meta_path = client.download_metadata(out_dir)

    meta = pd.read_csv(meta_path)
    typer.echo(f"Metadata: {meta_path} shape={meta.shape}")
    lineage_counts = {lineage: int(filter_lineages(meta, [lineage]).sum()) for lineage in LINEAGES}
    for lineage, count in lineage_counts.items():
        typer.echo(f"  {lineage}: {count} cell lines")

    expr = pd.read_csv(expr_path, index_col=0)
    n_nan = int(expr.isna().sum().sum())
    typer.echo(f"Expression matrix: {expr_path} shape={expr.shape}")
    typer.echo(f"NaN values in expression matrix: {n_nan}")

    return {
        "expression_path": str(expr_path),
        "metadata_path": str(meta_path),
        "expression_shape": expr.shape,
        "lineage_counts": lineage_counts,
        "n_nan": n_nan,
    }


@app.command()
def tcga(out_dir: str = "data/raw/tcga/"):
    raise NotImplementedError


if __name__ == "__main__":
    app()
