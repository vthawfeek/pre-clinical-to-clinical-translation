"""Gene-ID harmonisation, common-gene intersection, and union-rank HVG selection.

Merges CCLE (DepMap) and TCGA (Xena) expression matrices onto a shared HUGO-symbol
gene space so that both domains can be fed through the dual-tower encoders on Day 6+.
"""

import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

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

    def select_hvgs(
        self, ccle_expr, tcga_expr, common_genes, n_hvgs=2000, train_ids=None
    ) -> list[str]:
        """Union-rank HVG selection: average each gene's within-domain variance rank
        across CCLE and TCGA, then take the top `n_hvgs`. This gives equal weight to
        variability in each domain instead of letting TCGA's larger sample count
        dominate a pooled-variance ranking. Ties are broken alphabetically by gene
        symbol so the result is deterministic.

        `train_ids` closes the last leakage path (Day 16). When ``None`` (default)
        variance is computed over *all* samples — the Phase-1 behaviour, kept behind
        ``pctrans-preprocess --hvg-on all`` so the original gene list stays
        reproducible. When a ``{"ccle": [...], "tcga": [...]}`` dict of TRAIN sample
        IDs is passed, variance is computed on the train slice of each domain only,
        so gene selection never sees val/test samples (``--hvg-on train``).
        """
        if train_ids is not None:
            ccle_src = ccle_expr.loc[train_ids["ccle"]]
            tcga_src = tcga_expr.loc[train_ids["tcga"]]
        else:
            ccle_src, tcga_src = ccle_expr, tcga_expr

        var_ccle = ccle_src[common_genes].var(axis=0, ddof=1)
        var_tcga = tcga_src[common_genes].var(axis=0, ddof=1)
        mean_rank = (var_ccle.rank() + var_tcga.rank()) / 2

        ranked = mean_rank.rename("mean_rank").rename_axis("gene").reset_index()
        ranked = ranked.sort_values(["mean_rank", "gene"], ascending=[False, True], kind="mergesort")
        return ranked["gene"].iloc[:n_hvgs].tolist()

    def save_filtered(
        self,
        ccle_expr,
        tcga_expr,
        hvg_list,
        out_dir,
        ccle_name="ccle_2k.parquet",
        tcga_name="tcga_2k.parquet",
        gene_list_name="gene_list.txt",
    ) -> None:
        """Save both domains restricted to `hvg_list` (+ lineage) as parquet, and the
        gene list itself (one HUGO symbol per line, HVG rank order) as a text file.

        The default names reproduce the Phase-1 artefacts; the train-only HVG run
        (Day 16) passes ``*_trainhvg`` names so it never overwrites them.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        keep_cols = [*hvg_list, "lineage"]
        ccle_expr[keep_cols].to_parquet(out_dir / ccle_name)
        tcga_expr[keep_cols].to_parquet(out_dir / tcga_name)

        with open(out_dir / gene_list_name, "w", encoding="utf-8") as f:
            f.write("\n".join(hvg_list) + "\n")


class DataSplitter:
    """Lineage-stratified train/val/test splits + per-gene z-score scalers.

    The split is stratified by lineage *within each domain separately* so every
    split preserves the LUAD/BRCA/SKCM proportions of both CCLE and TCGA. The
    scaler is fit on the pooled CCLE_train + TCGA_train expression only, then
    applied to val/test — this is the data-leakage boundary for Day 5.
    """

    def stratified_split(self, ccle_df, tcga_df, val_frac=0.15, test_frac=0.15, seed=42):
        """Return `{"ccle": {"train"/"val"/"test": [ids]}, "tcga": {...}}`.

        IDs are the DataFrame index values. Each domain is split independently
        but with the same seed; within a domain each lineage is shuffled and
        partitioned so the fractions hold per lineage.
        """
        return {
            "ccle": self._split_one(ccle_df, val_frac, test_frac, seed),
            "tcga": self._split_one(tcga_df, val_frac, test_frac, seed),
        }

    def _split_one(self, df, val_frac, test_frac, seed):
        rng = np.random.default_rng(seed)
        train, val, test = [], [], []
        # groupby sorts lineage keys → deterministic iteration order.
        for _, group in df.groupby("lineage", sort=True):
            ids = rng.permutation(group.index.to_numpy())
            n = len(ids)
            n_test = int(round(n * test_frac))
            n_val = int(round(n * val_frac))
            test.extend(ids[:n_test].tolist())
            val.extend(ids[n_test : n_test + n_val].tolist())
            train.extend(ids[n_test + n_val :].tolist())
        return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}

    def fit_scalers(self, ccle_train_expr, tcga_train_expr, lineage_col="lineage"):
        """Fit one `StandardScaler` (per-gene z-score) on pooled train expression.

        Returns `{"scaler": StandardScaler, "feature_cols": [...]}`. The feature
        column order is captured so `apply_scalers` can realign any frame before
        transforming.
        """
        feature_cols = [c for c in ccle_train_expr.columns if c != lineage_col]
        pooled = pd.concat(
            [ccle_train_expr[feature_cols], tcga_train_expr[feature_cols]], axis=0
        )
        scaler = StandardScaler().fit(pooled.to_numpy(dtype=np.float64))
        return {"scaler": scaler, "feature_cols": feature_cols}

    def apply_scalers(self, expr_df, scalers, lineage_col="lineage"):
        """Z-score `expr_df`'s gene columns with a fitted scaler, keeping lineage."""
        scaler = scalers["scaler"]
        feature_cols = scalers["feature_cols"]
        scaled = scaler.transform(expr_df[feature_cols].to_numpy(dtype=np.float64))
        out = pd.DataFrame(scaled, index=expr_df.index, columns=feature_cols)
        if lineage_col in expr_df.columns:
            out[lineage_col] = expr_df[lineage_col].to_numpy()
        return out

    def save_splits(
        self,
        splits,
        scalers,
        out_dir,
        splits_name="splits.json",
        scalers_name="scalers.pkl",
    ) -> None:
        """Persist `splits.json` (sample IDs by domain/split) and `scalers.pkl`.

        Names are overridable so the train-only HVG run (Day 16) writes
        ``splits_trainhvg.json`` / ``scalers_trainhvg.pkl`` alongside the Phase-1
        artefacts instead of clobbering them.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / splits_name, "w", encoding="utf-8") as f:
            json.dump(splits, f, indent=2)
        with open(out_dir / scalers_name, "wb") as f:
            pickle.dump(scalers, f)
