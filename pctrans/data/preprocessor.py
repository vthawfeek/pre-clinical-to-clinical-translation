"""Gene-ID harmonisation, common-gene intersection, and union-rank HVG selection.

Merges CCLE (DepMap) and TCGA (Xena) expression matrices onto a shared HUGO-symbol
gene space so that both domains can be fed through the dual-tower encoders on Day 6+.
"""

import re
from pathlib import Path

import pandas as pd

from pctrans.data.ccle_client import (
    EXPRESSION_FILENAME as CCLE_EXPRESSION_FILENAME,
)
from pctrans.data.ccle_client import (
    METADATA_FILENAME as CCLE_METADATA_FILENAME,
)
from pctrans.data.ccle_client import filter_lineages
from pctrans.data.tcga_client import (
    EXPRESSION_FILENAME as TCGA_EXPRESSION_FILENAME,
)
from pctrans.data.tcga_client import (
    PHENOTYPE_FILENAME as TCGA_PHENOTYPE_FILENAME,
)
from pctrans.data.tcga_client import filter_tcga_lineages

LINEAGES = ["LUAD", "BRCA", "SKCM"]

# CCLE column headers are "SYMBOL (ENTREZ_ID)", e.g. "EGFR (1956)".
CCLE_ENTREZ_SUFFIX = re.compile(r" \(\d+\)$")


class FeatureSynchroniser:
    """Loads CCLE/TCGA raw data and reduces both to a shared top-N HVG feature space."""

    def load_ccle(self, raw_dir) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load CCLE expression + metadata, filter to LUAD/BRCA/SKCM, strip Entrez IDs.

        Returns (expr, meta): `expr` is indexed by ModelID with one column per gene
        symbol plus a trailing "lineage" column; `meta` is the filtered Model.csv.
        """
        raw_dir = Path(raw_dir) / "ccle"
        meta = pd.read_csv(raw_dir / CCLE_METADATA_FILENAME)
        meta = meta.loc[filter_lineages(meta, LINEAGES)].reset_index(drop=True)

        lineage = pd.Series(index=meta.index, dtype=object)
        for code in LINEAGES:
            lineage[filter_lineages(meta, [code])] = code
        meta["lineage"] = lineage

        expr = pd.read_csv(raw_dir / CCLE_EXPRESSION_FILENAME, index_col=0)
        expr.columns = [CCLE_ENTREZ_SUFFIX.sub("", c) for c in expr.columns]

        model_lineage = meta.set_index("ModelID")["lineage"]
        expr = expr.loc[expr.index.intersection(model_lineage.index)].copy()
        expr["lineage"] = model_lineage.loc[expr.index]

        return expr, meta

    def load_tcga(self, raw_dir) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load TCGA expression + phenotype, filter to LUAD/BRCA/SKCM.

        Returns (expr, pheno): `expr` is indexed by TCGA sample ID with one column
        per gene symbol plus a trailing "lineage" column; `pheno` is the filtered
        phenotype table. Only the phenotype-matched sample columns are read from the
        ~1.2 GB expression file to keep this fast.
        """
        raw_dir = Path(raw_dir) / "tcga"
        pheno = pd.read_csv(raw_dir / TCGA_PHENOTYPE_FILENAME, sep="\t")
        pheno = pheno.loc[filter_tcga_lineages(pheno, LINEAGES)].reset_index(drop=True)

        expr_path = raw_dir / TCGA_EXPRESSION_FILENAME
        with open(expr_path, encoding="utf-8") as f:
            header = f.readline().rstrip("\n").split("\t")
        gene_id_col = header[0]
        wanted_samples = set(header) & set(pheno["sample"])
        usecols = wanted_samples | {gene_id_col}

        expr = pd.read_csv(expr_path, sep="\t", usecols=usecols, index_col=0)
        # A handful of gene rows (e.g. "SLC35E2") are duplicated in the Xena matrix;
        # keep the first occurrence for a unique gene index.
        expr = expr[~expr.index.duplicated(keep="first")]
        expr = expr.T
        expr.index.name = "sample"

        lineage = pheno.set_index("sample")["cancer type abbreviation"]
        expr = expr.loc[expr.index.intersection(lineage.index)].copy()
        expr["lineage"] = lineage.loc[expr.index]

        return expr, pheno

    def find_common_genes(self, ccle_genes, tcga_genes) -> list[str]:
        """Set intersection of gene symbols, sorted for reproducibility."""
        return sorted(set(ccle_genes) & set(tcga_genes))

    def select_hvgs(self, ccle_expr, tcga_expr, common_genes, n_hvgs=2000) -> list[str]:
        """Union-rank HVG selection: average each gene's within-domain variance rank
        across CCLE and TCGA, then take the top `n_hvgs`. This gives equal weight to
        variability in each domain instead of letting TCGA's larger sample count
        dominate a pooled-variance ranking. Ties are broken alphabetically by gene
        symbol so the result is deterministic.
        """
        var_ccle = ccle_expr[common_genes].var(axis=0, ddof=1)
        var_tcga = tcga_expr[common_genes].var(axis=0, ddof=1)
        mean_rank = (var_ccle.rank() + var_tcga.rank()) / 2

        ranked = mean_rank.rename("mean_rank").rename_axis("gene").reset_index()
        ranked = ranked.sort_values(["mean_rank", "gene"], ascending=[False, True], kind="mergesort")
        return ranked["gene"].iloc[:n_hvgs].tolist()

    def save_filtered(self, ccle_expr, tcga_expr, hvg_list, out_dir) -> None:
        """Save both domains restricted to `hvg_list` (+ lineage) as parquet, and the
        gene list itself (one HUGO symbol per line, HVG rank order) as a text file.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        keep_cols = [*hvg_list, "lineage"]
        ccle_expr[keep_cols].to_parquet(out_dir / "ccle_2k.parquet")
        tcga_expr[keep_cols].to_parquet(out_dir / "tcga_2k.parquet")

        with open(out_dir / "gene_list.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(hvg_list) + "\n")


class DataSplitter:
    def stratified_split(self, ccle_df, tcga_df, val_frac=0.15, test_frac=0.15, seed=42):
        raise NotImplementedError

    def fit_scalers(self, ccle_train_expr, tcga_train_expr):
        raise NotImplementedError

    def apply_scalers(self, expr_df, scaler):
        raise NotImplementedError

    def save_splits(self, splits, scalers, out_dir):
        raise NotImplementedError
