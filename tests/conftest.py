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
    return {"input_dim": 5, "hidden_dims": [16, 8], "embed_dim": 8, "dropout": 0.1}
