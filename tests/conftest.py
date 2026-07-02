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
def tiny_ccle_dataset(tiny_ccle):
    from pctrans.data.dataset import CCLEDataset

    return CCLEDataset(tiny_ccle, lineage_col="lineage")


@pytest.fixture
def tiny_tcga_dataset(tiny_tcga):
    from pctrans.data.dataset import TCGADataset

    return TCGADataset(tiny_tcga, lineage_col="lineage")


@pytest.fixture
def tiny_sampler(tiny_ccle_dataset, tiny_tcga_dataset):
    from pctrans.data.sampler import StratifiedContrastiveBatchSampler

    # batch_size=6 -> 1 CCLE + 1 TCGA per lineage across LUAD/BRCA/SKCM.
    return StratifiedContrastiveBatchSampler(
        tiny_ccle_dataset, tiny_tcga_dataset, batch_size=6
    )


@pytest.fixture
def small_model():
    """A dual-tower model sized to the 50-gene tiny fixtures (not the 2000-gene one).

    The plan's `test_one_training_epoch` snippet names `tiny_model`, but that
    fixture is 2000-dim to match the Day 6 encoder tests; the tiny datasets have
    50 genes, so training needs an input-matched model. Small hidden dims keep a
    forward+backward pass well under a second.
    """
    from pctrans.models.dual_tower import DualTowerModel
    from pctrans.models.encoders import CCLEEncoder, TCGAEncoder

    ccle = CCLEEncoder(input_dim=50, hidden_dims=(16, 8), embed_dim=8, dropout=0.1, dropout_low=0.1)
    tcga = TCGAEncoder(input_dim=50, hidden_dims=(16, 8), embed_dim=8, dropout=0.1, dropout_low=0.1)
    return DualTowerModel(ccle, tcga)


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


# --- Day 13: end-to-end pipeline fixture (train -> evaluate -> visualize -> query) ---

_PIPELINE_LINEAGES = ["LUAD", "BRCA", "SKCM"]
_PIPELINE_MODEL_YAML = """\
input_dim: 40
hidden_dims: [32, 16]
embed_dim: 64
dropout_high: 0.1
dropout_low: 0.1
init_tau: 0.07
"""


def _synthetic_expr(n_per_lineage, n_genes, id_fn, seed):
    """Lineage-separable synthetic expression frame (genes + trailing lineage col).

    Each lineage lifts a distinct 5-gene block so the towers can learn a real
    (non-degenerate) alignment on a tiny, fast dataset.
    """
    rng = np.random.default_rng(seed)
    rows, labels, ids = [], [], []
    counter = 0
    for li, lineage in enumerate(_PIPELINE_LINEAGES):
        base = np.zeros(n_genes)
        base[li * 5 : (li + 1) * 5] = 4.0
        for _ in range(n_per_lineage):
            rows.append(base + rng.normal(0.0, 1.0, size=n_genes))
            labels.append(lineage)
            ids.append(id_fn(counter))
            counter += 1
    df = pd.DataFrame(rows, columns=[f"GENE{i}" for i in range(n_genes)], index=ids)
    df["lineage"] = labels
    return df


@pytest.fixture(scope="session")
def pipeline(tmp_path_factory):
    """A fully processed data dir + trained checkpoint + test embeddings.

    Builds synthetic ``ccle_2k.parquet`` / ``tcga_2k.parquet`` / ``splits.json`` /
    ``scalers.pkl``, writes tiny model/training configs, runs ``pctrans-train`` for
    one epoch to produce ``best_model.pt``, and writes ``embeddings_test.npz`` (the
    Day 11 app artefact) — everything the Day 10-13 CLIs read. Session-scoped so the
    one training pass is shared across all script/inference tests.
    """
    import numpy as _np
    import torch
    import yaml
    from torch.utils.data import DataLoader

    from pctrans.data.dataset import CCLEDataset, TCGADataset
    from pctrans.data.preprocessor import DataSplitter
    from pctrans.evaluation.knn import embed_loader
    from pctrans.scripts.evaluate import _build_model, _load_yaml
    from pctrans.scripts.train import main as train_main

    data_dir = tmp_path_factory.mktemp("pipeline")

    ccle_df = _synthetic_expr(20, 40, lambda c: f"ACH-{c:06d}", seed=1)
    tcga_df = _synthetic_expr(60, 40, lambda c: f"TCGA-{c:04d}-01", seed=2)

    splitter = DataSplitter()
    splits = splitter.stratified_split(ccle_df, tcga_df, val_frac=0.2, test_frac=0.2, seed=42)
    ccle_train = ccle_df.loc[splits["ccle"]["train"]]
    tcga_train = tcga_df.loc[splits["tcga"]["train"]]
    scalers = splitter.fit_scalers(ccle_train, tcga_train)

    ccle_df.to_parquet(data_dir / "ccle_2k.parquet")
    tcga_df.to_parquet(data_dir / "tcga_2k.parquet")
    splitter.save_splits(splits, scalers, data_dir)

    model_yaml = data_dir / "model.yaml"
    model_yaml.write_text(_PIPELINE_MODEL_YAML, encoding="utf-8")
    train_yaml = data_dir / "training.yaml"
    ckpt_path = data_dir / "best_model.pt"
    train_cfg = {
        "n_epochs": 1,
        "batch_size": 6,
        "lr": 1.0e-3,
        "warmup_epochs": 1,
        "grad_clip_norm": 1.0,
        "knn_k": 5,
        "early_stop_patience": 5,
        "checkpoint_path": str(ckpt_path),
        "eval_batch_size": 64,
    }
    train_yaml.write_text(yaml.safe_dump(train_cfg), encoding="utf-8")

    train_main(
        config=str(train_yaml),
        model_config=str(model_yaml),
        data_dir=str(data_dir),
        epochs=1,
        mlflow=False,
    )

    # Write embeddings_test.npz (Day 11 app artefact) directly, without the UMAP
    # step, so the fast script/inference tests do not trigger the numba JIT.
    net = _build_model(_load_yaml(model_yaml))
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    net.load_state_dict(checkpoint["model_state_dict"])
    net.eval()

    def _scaled(df, ids):
        return splitter.apply_scalers(df.loc[ids], scalers)

    ccle_test = CCLEDataset(_scaled(ccle_df, splits["ccle"]["test"]))
    tcga_test = TCGADataset(_scaled(tcga_df, splits["tcga"]["test"]))
    z_ccle, y_ccle = embed_loader(net.encode_ccle, DataLoader(ccle_test, batch_size=64))
    z_tcga, y_tcga = embed_loader(net.encode_tcga, DataLoader(tcga_test, batch_size=64))
    _np.savez(
        data_dir / "embeddings_test.npz",
        z_ccle=z_ccle,
        y_ccle=y_ccle,
        ids_ccle=ccle_test.ids.astype(str),
        z_tcga=z_tcga,
        y_tcga=y_tcga,
        ids_tcga=tcga_test.ids.astype(str),
    )

    return {
        "data_dir": str(data_dir),
        "model": str(ckpt_path),
        "model_config": str(model_yaml),
        "training_config": str(train_yaml),
        "eval_summary": str(data_dir / "eval_summary.json"),
        "reports_dir": str(data_dir / "reports"),
        "any_ccle_id": str(ccle_df.index[0]),
        "n_ccle": int(len(ccle_df)),
    }
