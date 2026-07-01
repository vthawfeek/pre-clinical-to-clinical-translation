from pathlib import Path

import pandas as pd
import pytest

from pctrans.data.ccle_client import filter_lineages


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
