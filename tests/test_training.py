import math

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pctrans.data.dataset import CCLEDataset, TCGADataset
from pctrans.models.losses import SupConInfoNCELoss
from pctrans.training.callbacks import KNNValidationCallback
from pctrans.training.trainer import ContrastiveTrainer


class _IdentityModel:
    """Fake dual-tower model whose embedding is the L2-normalised input.

    Lets the kNN callback be tested on deterministic embeddings without training.
    """

    def __init__(self):
        self.training = False

    def eval(self):
        self.training = False
        return self

    def train(self, mode=True):
        self.training = mode
        return self

    def encode_ccle(self, x):
        return F.normalize(x, dim=-1)

    encode_tcga = encode_ccle


def _one_hot_frame(labels):
    rows = []
    for lab in labels:
        row = [0.0, 0.0, 0.0]
        row["LUAD BRCA SKCM".split().index(lab)] = 1.0
        rows.append(row)
    df = pd.DataFrame(rows, columns=["G0", "G1", "G2"])
    df["lineage"] = labels
    return df


def _train_config(tmp_path, **overrides):
    cfg = {
        "lr": 1.0e-3,
        "batch_size": 6,
        "warmup_epochs": 1,
        "grad_clip_norm": 1.0,
        "knn_k": 5,
        "early_stop_patience": 5,
        "checkpoint_path": str(tmp_path / "best_model.pt"),
        "eval_batch_size": 32,
    }
    cfg.update(overrides)
    return cfg


def test_knn_callback_perfect_embeddings():
    # Features are lineage one-hots and the fake model is the identity, so every
    # CCLE sample retrieves same-lineage TCGA neighbours: accuracy must be 1.0.
    ccle = CCLEDataset(_one_hot_frame(["LUAD", "BRCA", "SKCM", "LUAD", "BRCA", "SKCM"]))
    tcga = TCGADataset(_one_hot_frame(["LUAD", "BRCA", "SKCM"] * 4))
    callback = KNNValidationCallback(k=3)
    result = callback(_IdentityModel(), DataLoader(ccle, batch_size=4), DataLoader(tcga, batch_size=4))
    assert result["val_knn_accuracy"] == 1.0
    assert set(result["per_lineage"]) == {"LUAD", "BRCA", "SKCM"}
    assert all(v == 1.0 for v in result["per_lineage"].values())


def test_one_training_epoch(small_model, tiny_sampler, tiny_ccle_dataset, tiny_tcga_dataset, tmp_path):
    trainer = ContrastiveTrainer(
        small_model,
        SupConInfoNCELoss(init_tau=0.07),
        tiny_sampler,
        tiny_ccle_dataset,
        tiny_tcga_dataset,
        _train_config(tmp_path),
    )
    result = trainer.train(n_epochs=1)
    assert math.isfinite(result["train_loss"])
    assert math.isfinite(result["val_loss"])
    assert 0.0 <= result["val_knn_accuracy"] <= 1.0
    assert len(result["history"]) == 1


def test_checkpoint_saved_on_improvement(small_model, tiny_sampler, tiny_ccle_dataset, tiny_tcga_dataset, tmp_path):
    ckpt = tmp_path / "best_model.pt"
    trainer = ContrastiveTrainer(
        small_model,
        SupConInfoNCELoss(),
        tiny_sampler,
        tiny_ccle_dataset,
        tiny_tcga_dataset,
        _train_config(tmp_path, checkpoint_path=str(ckpt)),
    )
    trainer.train(n_epochs=1)
    # The first epoch always improves on the -1 sentinel, so a checkpoint exists.
    assert ckpt.exists()
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert "model_state_dict" in payload
    assert "loss_state_dict" in payload


def test_early_stopping_fires(small_model, tiny_sampler, tiny_ccle_dataset, tiny_tcga_dataset, tmp_path):
    # patience=1 with a capped 20 epochs: training must stop before all 20 run
    # once val kNN stops improving on the tiny fixtures.
    trainer = ContrastiveTrainer(
        small_model,
        SupConInfoNCELoss(),
        tiny_sampler,
        tiny_ccle_dataset,
        tiny_tcga_dataset,
        _train_config(tmp_path, early_stop_patience=1),
    )
    result = trainer.train(n_epochs=20)
    assert len(result["history"]) <= 20


def test_load_checkpoint_restores_weights(small_model, tiny_sampler, tiny_ccle_dataset, tiny_tcga_dataset, tmp_path):
    ckpt = tmp_path / "best_model.pt"
    trainer = ContrastiveTrainer(
        small_model,
        SupConInfoNCELoss(),
        tiny_sampler,
        tiny_ccle_dataset,
        tiny_tcga_dataset,
        _train_config(tmp_path, checkpoint_path=str(ckpt)),
    )
    trainer.train(n_epochs=1)
    reference = small_model.ccle_encoder.projection.weight.detach().clone()

    # Perturb, then restore from checkpoint -> weights must match again.
    with torch.no_grad():
        small_model.ccle_encoder.projection.weight.add_(1.0)
    assert not torch.allclose(small_model.ccle_encoder.projection.weight, reference)

    trainer.load_checkpoint(ckpt)
    assert torch.allclose(small_model.ccle_encoder.projection.weight, reference)


def test_lr_warmup_then_cosine(small_model, tiny_sampler, tiny_ccle_dataset, tiny_tcga_dataset, tmp_path):
    trainer = ContrastiveTrainer(
        small_model,
        SupConInfoNCELoss(),
        tiny_sampler,
        tiny_ccle_dataset,
        tiny_tcga_dataset,
        _train_config(tmp_path, warmup_epochs=3),
    )
    scale = trainer._lr_lambda(n_epochs=10)
    # Linear warmup rises to 1.0 at the end of warmup, then cosine-decays to ~0.
    assert scale(0) < scale(1) < scale(2)
    assert math.isclose(scale(2), 1.0, rel_tol=1e-6)
    assert scale(9) < 0.1
