import pandas as pd
import typer

from pctrans.data.ccle_client import CCLEClient, filter_lineages
from pctrans.data.tcga_client import TCGAClient, filter_tcga_lineages

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
    """Download TCGA Pan-Cancer expression matrix + phenotype table from UCSC Xena, report lineage counts."""
    client = TCGAClient()
    expr_path = client.download_expression(out_dir)
    pheno_path = client.download_phenotype(out_dir)

    pheno = pd.read_csv(pheno_path, sep="\t")
    typer.echo(f"Phenotype: {pheno_path} shape={pheno.shape}")
    lineage_counts = {lineage: int(filter_tcga_lineages(pheno, [lineage]).sum()) for lineage in LINEAGES}
    for lineage, count in lineage_counts.items():
        typer.echo(f"  {lineage}: {count} patients")

    # The expression matrix is ~1.5 GB uncompressed (genes as rows); avoid loading it
    # fully into memory just to report its shape.
    with open(expr_path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
    n_samples = len(header) - 1
    with open(expr_path, encoding="utf-8") as f:
        n_genes = sum(1 for _ in f) - 1  # minus header row
    first_genes = pd.read_csv(expr_path, sep="\t", usecols=[0], nrows=3).iloc[:, 0].tolist()

    typer.echo(f"Expression matrix: {expr_path} genes={n_genes} samples={n_samples}")
    typer.echo(f"First 3 genes: {first_genes}")

    return {
        "expression_path": str(expr_path),
        "phenotype_path": str(pheno_path),
        "n_genes": n_genes,
        "n_samples": n_samples,
        "first_genes": first_genes,
        "lineage_counts": lineage_counts,
    }


@app.command()
def purity(out_dir: str = "data/raw/tcga/"):
    """Download TCGA ABSOLUTE consensus purity/ploidy calls (Day 20 confounder analysis)."""
    client = TCGAClient()
    purity_path = client.download_purity(out_dir)

    table = pd.read_csv(purity_path, sep="\t")
    n_called = int((table["call status"] == "called").sum())
    typer.echo(f"Purity table: {purity_path} shape={table.shape}")
    typer.echo(f"Samples with a called purity estimate: {n_called}/{len(table)}")
    typer.echo(f"Purity range: [{table['purity'].min():.2f}, {table['purity'].max():.2f}]")

    return {
        "purity_path": str(purity_path),
        "shape": table.shape,
        "n_called": n_called,
    }


if __name__ == "__main__":
    app()
