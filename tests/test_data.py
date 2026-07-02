from pathlib import Path

import pandas as pd
import pytest

from pctrans.data.ccle_client import filter_lineages
from pctrans.data.preprocessor import FeatureSynchroniser
from pctrans.data.tcga_client import filter_tcga_lineages


def test_tiny_ccle_shape(tiny_ccle):
    assert tiny_ccle.shape == (10, 51)


def test_ccle_client_filter_lineages(tiny_ccle_meta):
    filtered = filter_lineages(tiny_ccle_meta, ["LUAD", "BRCA", "SKCM"])
    assert filtered.sum() > 0


def test_ccle_client_filter_lineages_excludes_other_diseases(tiny_ccle_meta):
    filtered = filter_lineages(tiny_ccle_meta, ["LUAD", "BRCA", "SKCM"])
    # Glioblastoma (row 3) and Lung Squamous Cell Carcinoma (row 5) are not LUAD/BRCA/SKCM
    assert filtered.tolist() == [True, True, True, False, True, False]


@pytest.mark.integration
def test_ccle_expression_no_nan():
    path = Path("data/raw/ccle/OmicsExpressionProteinCodingGenesTPMLogp1.csv")
    if not path.exists():
        pytest.skip("CCLE expression matrix not downloaded")
    head = pd.read_csv(path, index_col=0, nrows=10)
    assert not head.isna().any().any()


def test_tcga_client_filter_lineages(tiny_tcga_meta):
    filtered = filter_tcga_lineages(tiny_tcga_meta, ["LUAD", "BRCA", "SKCM"])
    assert filtered.tolist() == [True, True, True, False, True, False]


def test_tcga_client_filter_lineages_matched_labels(tiny_tcga_meta):
    # PLAN.md's test asserts the *matched* labels are a subset of {LUAD, BRCA, SKCM};
    # filter_tcga_lineages returns the boolean mask (consistent with filter_lineages
    # above), so apply it before checking the label set.
    filtered = filter_tcga_lineages(tiny_tcga_meta, ["LUAD", "BRCA", "SKCM"])
    matched = tiny_tcga_meta.loc[filtered, "cancer type abbreviation"]
    assert set(matched.unique()).issubset({"LUAD", "BRCA", "SKCM"})


@pytest.mark.integration
def test_tcga_expression_gene_count():
    path = Path("data/raw/tcga/EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena")
    if not path.exists():
        pytest.skip("TCGA expression matrix not downloaded")
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
    assert len(header) > 1000


@pytest.mark.integration
def test_tcga_phenotype_has_target_lineages():
    path = Path("data/raw/tcga/Survival_SupplementalTable_S1_20171025_xena_sp.tsv")
    if not path.exists():
        pytest.skip("TCGA phenotype table not downloaded")
    pheno = pd.read_csv(path, sep="\t")
    filtered = filter_tcga_lineages(pheno, ["LUAD", "BRCA", "SKCM"])
    assert filtered.sum() > 0


def test_find_common_genes_sorted():
    fs = FeatureSynchroniser()
    common = fs.find_common_genes(["B", "A", "C"], ["C", "B", "D"])
    assert common == ["B", "C"]


def _common_genes(fs, tiny_ccle, tiny_tcga):
    ccle_genes = [c for c in tiny_ccle.columns if c != "lineage"]
    tcga_genes = [c for c in tiny_tcga.columns if c != "lineage"]
    return fs.find_common_genes(ccle_genes, tcga_genes)


def test_hvg_selection_count(tiny_ccle, tiny_tcga):
    fs = FeatureSynchroniser()
    common_genes = _common_genes(fs, tiny_ccle, tiny_tcga)
    hvgs = fs.select_hvgs(tiny_ccle, tiny_tcga, common_genes, n_hvgs=10)
    assert len(hvgs) == 10


def test_hvg_selection_deterministic(tiny_ccle, tiny_tcga):
    # HVG variance is computed on ALL samples (no split yet) — this is correct,
    # since the split only needs to happen before fitting scalers (Day 5).
    fs = FeatureSynchroniser()
    common_genes = _common_genes(fs, tiny_ccle, tiny_tcga)
    hvgs_1 = fs.select_hvgs(tiny_ccle, tiny_tcga, common_genes, n_hvgs=10)
    hvgs_2 = fs.select_hvgs(tiny_ccle, tiny_tcga, common_genes, n_hvgs=10)
    assert hvgs_1 == hvgs_2


def test_hvg_tie_break_is_alphabetical():
    # GENEA and GENEB have identical variance in both domains, so mean_rank ties;
    # the deterministic tie-break must prefer the alphabetically-first symbol.
    ccle = pd.DataFrame({"GENEB": [1.0, 2.0, 3.0], "GENEA": [1.0, 2.0, 3.0]})
    tcga = pd.DataFrame({"GENEB": [4.0, 5.0, 6.0], "GENEA": [4.0, 5.0, 6.0]})
    fs = FeatureSynchroniser()
    hvgs = fs.select_hvgs(ccle, tcga, ["GENEA", "GENEB"], n_hvgs=1)
    assert hvgs == ["GENEA"]
