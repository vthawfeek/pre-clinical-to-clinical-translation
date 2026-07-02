import numpy as np
import pandas as pd
import pytest

LINEAGES = ["LUAD", "BRCA", "SKCM"]


@pytest.fixture
def tiny_ccle():
    rng = np.random.default_rng(42)
    n_samples, n_genes = 10, 50
    expr = rng.normal(size=(n_samples, n_genes))
    columns = [f"GENE{i}" for i in range(n_genes)]
    df = pd.DataFrame(expr, columns=columns)
    df["lineage"] = [LINEAGES[i % len(LINEAGES)] for i in range(n_samples)]
    return df


@pytest.fixture
def tiny_tcga():
    rng = np.random.default_rng(43)
    n_samples, n_genes = 20, 50
    expr = rng.normal(size=(n_samples, n_genes))
    columns = [f"GENE{i}" for i in range(n_genes)]
    df = pd.DataFrame(expr, columns=columns)
    df["lineage"] = [LINEAGES[i % len(LINEAGES)] for i in range(n_samples)]
    return df


@pytest.fixture
def tiny_model():
    from pctrans.models.dual_tower import DualTowerModel
    from pctrans.models.encoders import CCLEEncoder, TCGAEncoder

    ccle = CCLEEncoder(input_dim=2000, hidden_dims=(1024, 512, 256, 128), embed_dim=64)
    tcga = TCGAEncoder(input_dim=2000, hidden_dims=(1024, 512, 256, 128), embed_dim=64)
    return DualTowerModel(ccle, tcga).eval()


@pytest.fixture
def tiny_ccle_meta():
    return pd.DataFrame(
        {
            "ModelID": [f"ACH-{i:06d}" for i in range(6)],
            "OncotreePrimaryDisease": [
                "Melanoma",
                "Non-Small Cell Lung Cancer",
                "Invasive Breast Carcinoma",
                "Glioblastoma",
                "Melanoma",
                "Non-Small Cell Lung Cancer",
            ],
            "OncotreeSubtype": [
                "Melanoma",
                "Lung Adenocarcinoma",
                "Invasive Breast Carcinoma",
                "Glioblastoma",
                "Cutaneous Melanoma",
                "Lung Squamous Cell Carcinoma",
            ],
        }
    )


@pytest.fixture
def tiny_tcga_meta():
    return pd.DataFrame(
        {
            "sample": [f"TCGA-{i:02d}-0001-01" for i in range(6)],
            "cancer type abbreviation": ["LUAD", "BRCA", "SKCM", "GBM", "LUAD", "COAD"],
        }
    )
