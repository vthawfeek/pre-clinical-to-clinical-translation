from pathlib import Path
from typing import Optional

import typer
import yaml

from pctrans.data.preprocessor import DataSplitter, FeatureSynchroniser

app = typer.Typer()

_DEFAULT_LINEAGES = ["LUAD", "BRCA", "SKCM"]


def _load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _artefact_names(hvg_on: str, output_suffix: Optional[str]) -> dict:
    """Output filenames for this run.

    ``output_suffix=None`` (default) derives the Phase-1/Day-16 convention from
    ``hvg_on`` alone: ``all`` keeps the bare Phase-1 names, ``train`` appends
    ``_trainhvg`` -- byte-for-byte the same as the old static
    ``_ARTEFACT_NAMES`` dict. An explicit suffix (e.g. ``_15`` for Day 18's
    15-lineage run) overrides that and is used verbatim, independent of
    ``hvg_on``, so a different lineage set never collides with the Phase-1
    artefacts.
    """
    if output_suffix is None:
        output_suffix = "_trainhvg" if hvg_on == "train" else ""
    return {
        "ccle": f"ccle_2k{output_suffix}.parquet",
        "tcga": f"tcga_2k{output_suffix}.parquet",
        "gene_list": f"gene_list{output_suffix}.txt",
        "splits": f"splits{output_suffix}.json",
        "scalers": f"scalers{output_suffix}.pkl",
    }


@app.command()
def main(
    raw_dir: str = "data/raw/",
    out_dir: str = "data/processed/",
    data_config: str = "configs/data.yaml",
    n_hvgs: Optional[int] = None,
    split: bool = False,
    val_frac: Optional[float] = None,
    test_frac: Optional[float] = None,
    seed: Optional[int] = None,
    hvg_on: str = "all",
    output_suffix: Optional[str] = None,
):
    """Synchronise CCLE + TCGA onto a shared top-`n_hvgs` HVG feature space.

    With ``--split``, additionally produce lineage-stratified train/val/test
    splits (``splits.json``) and per-gene z-score scalers fit on the pooled
    training expression only (``scalers.pkl``).

    ``--data-config`` (Day 18) supplies the lineage list (and ``n_hvgs``/
    ``val_frac``/``test_frac``/``seed`` defaults) instead of the 3-lineage
    hardcode -- point it at ``configs/data_15.yaml`` for the 15-lineage run.
    Explicit CLI flags still override the config when given.

    ``--hvg-on`` chooses the feature-selection scope (Day 16):

    - ``all`` (default): variance ranked over every sample, then split — the
      Phase-1 pipeline, kept byte-for-byte reproducible.
    - ``train``: split first, then rank variance on the train slice only, so HVG
      selection never sees val/test samples. Implies ``--split`` and writes a
      parallel ``*_trainhvg`` artefact set rather than overwriting the Phase-1 one.

    ``--output-suffix`` names the output artefact set explicitly (e.g. ``_15``);
    see `_artefact_names`.
    """
    if hvg_on not in ("all", "train"):
        raise typer.BadParameter("--hvg-on must be 'all' or 'train'")
    train_only = hvg_on == "train"
    if train_only:
        split = True  # train-only HVG selection requires the split up front
    names = _artefact_names(hvg_on, output_suffix)

    data_cfg = _load_yaml(data_config) if Path(data_config).exists() else {}
    lineages = data_cfg.get("lineages", _DEFAULT_LINEAGES)
    n_hvgs = n_hvgs if n_hvgs is not None else data_cfg.get("n_hvgs", 2000)
    val_frac = val_frac if val_frac is not None else data_cfg.get("val_frac", 0.15)
    test_frac = test_frac if test_frac is not None else data_cfg.get("test_frac", 0.15)
    seed = seed if seed is not None else data_cfg.get("seed", 42)

    fs = FeatureSynchroniser(lineages=lineages)

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

    complete_genes = fs.drop_incomplete_genes(ccle_expr, tcga_expr, common_genes)
    n_dropped = len(common_genes) - len(complete_genes)
    if n_dropped:
        typer.echo(
            f"Dropped {n_dropped} gene(s) with missing values in at least one "
            f"domain; {len(complete_genes)} complete common genes remain"
        )
    common_genes = complete_genes

    # Train-only HVG needs the split before ranking variance; the split depends
    # only on sample IDs + lineage + seed, so computing it on the full-gene frame
    # here is identical to computing it on the HVG-restricted frame.
    splitter = DataSplitter()
    splits = None
    train_ids = None
    if split:
        splits = splitter.stratified_split(
            ccle_expr, tcga_expr, val_frac=val_frac, test_frac=test_frac, seed=seed
        )
        if train_only:
            train_ids = {
                "ccle": splits["ccle"]["train"],
                "tcga": splits["tcga"]["train"],
            }

    hvgs = fs.select_hvgs(
        ccle_expr, tcga_expr, common_genes, n_hvgs=n_hvgs, train_ids=train_ids
    )
    scope = "train slice only" if train_only else "all samples"
    typer.echo(f"Selected top {len(hvgs)} HVGs (union-rank method, variance on {scope})")
    typer.echo(f"Top 10 HVGs by mean rank: {hvgs[:10]}")

    fs.save_filtered(
        ccle_expr,
        tcga_expr,
        hvgs,
        out_dir,
        ccle_name=names["ccle"],
        tcga_name=names["tcga"],
        gene_list_name=names["gene_list"],
    )
    typer.echo(
        f"Saved {names['ccle']}, {names['tcga']}, {names['gene_list']} to {out_dir}"
    )

    result = {
        "hvg_on": hvg_on,
        "ccle_shape": ccle_expr.shape,
        "tcga_shape": tcga_expr.shape,
        "n_common_genes": len(common_genes),
        "n_hvgs": len(hvgs),
        "top_10_hvgs": hvgs[:10],
    }

    if split:
        keep_cols = [*hvgs, "lineage"]
        ccle_train = ccle_expr[keep_cols].loc[splits["ccle"]["train"]]
        tcga_train = tcga_expr[keep_cols].loc[splits["tcga"]["train"]]
        scalers = splitter.fit_scalers(ccle_train, tcga_train)
        splitter.save_splits(
            splits,
            scalers,
            out_dir,
            splits_name=names["splits"],
            scalers_name=names["scalers"],
        )

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
        typer.echo(f"Saved {names['splits']}, {names['scalers']} to {out_dir}")

        result["splits"] = {
            domain: {k: len(v) for k, v in splits[domain].items()}
            for domain in ("ccle", "tcga")
        }

    return result


if __name__ == "__main__":
    app()
