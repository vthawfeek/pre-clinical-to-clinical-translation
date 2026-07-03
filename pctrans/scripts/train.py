"""``pctrans-train`` CLI: fit the dual-tower contrastive model on processed data.

Loads the Day 4-5 artefacts (``ccle_2k.parquet``, ``tcga_2k.parquet``,
``splits.json``, ``scalers.pkl``), z-scores each split with the train-fit scaler,
builds the stratified sampler, and runs `ContrastiveTrainer`. Prints the Gate 0
summary (epoch-1 loss finite, val kNN > 0, temperature logged) so Day 7's manual
gate can be read straight off the CLI output.
"""

import json
import math
import pickle
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
import yaml

from pctrans.data.dataset import CCLEDataset, TCGADataset, build_lineage_maps
from pctrans.data.preprocessor import DataSplitter
from pctrans.data.sampler import StratifiedContrastiveBatchSampler
from pctrans.models.dual_tower import DualTowerModel
from pctrans.models.encoders import CCLEEncoder, TCGAEncoder
from pctrans.models.losses import SupConInfoNCELoss
from pctrans.training.trainer import ContrastiveTrainer

app = typer.Typer()

_DEFAULT_LINEAGES = ["LUAD", "BRCA", "SKCM"]


def _load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_model(model_cfg):
    hidden = tuple(model_cfg.get("hidden_dims", [1024, 512, 256, 128]))
    kwargs = {
        "input_dim": model_cfg.get("input_dim", 2000),
        "hidden_dims": hidden,
        "embed_dim": model_cfg.get("embed_dim", 64),
        "dropout": model_cfg.get("dropout_high", 0.3),
        "dropout_low": model_cfg.get("dropout_low", 0.2),
    }
    model = DualTowerModel(CCLEEncoder(**kwargs), TCGAEncoder(**kwargs))
    loss_fn = SupConInfoNCELoss(init_tau=model_cfg.get("init_tau", 0.07))
    return model, loss_fn


@app.command()
def main(
    config: str = "configs/training.yaml",
    model_config: str = "configs/model.yaml",
    data_config: str = "configs/data.yaml",
    data_dir: str = "data/processed/",
    ccle_file: str = "ccle_2k.parquet",
    tcga_file: str = "tcga_2k.parquet",
    splits_file: str = "splits.json",
    scalers_file: str = "scalers.pkl",
    checkpoint_path: Optional[str] = None,
    epochs: Optional[int] = None,
    mlflow: bool = True,
):
    """Train the dual-tower model; pass ``--epochs 1`` for the Gate 0 smoke run.

    ``--ccle-file/--tcga-file/--splits-file/--scalers-file`` let a run point at an
    alternate artefact set (e.g. Day 16's ``*_trainhvg`` or Day 18's ``*_15``
    outputs) without touching the Phase-1 defaults. ``--checkpoint-path``
    overrides the config's save path for the same reason (so it never
    overwrites ``models/best_model.pt``). ``--data-config`` (Day 18) supplies
    the lineage list used to build the label-id map -- point it at
    ``configs/data_15.yaml`` for the 15-lineage run so labels 0..14 are built
    from that config instead of the hardcoded 3-lineage default.
    """
    data_dir = Path(data_dir)
    train_cfg = _load_yaml(config)
    model_cfg = _load_yaml(model_config)
    data_cfg = _load_yaml(data_config) if Path(data_config).exists() else {}
    lineages = data_cfg.get("lineages", _DEFAULT_LINEAGES)
    lineage_to_idx, idx_to_lineage = build_lineage_maps(lineages)
    n_epochs = epochs if epochs is not None else train_cfg.get("n_epochs", 30)
    if checkpoint_path is not None:
        train_cfg = dict(train_cfg)
        train_cfg["checkpoint_path"] = checkpoint_path

    ccle_df = pd.read_parquet(data_dir / ccle_file)
    tcga_df = pd.read_parquet(data_dir / tcga_file)
    with open(data_dir / splits_file, encoding="utf-8") as f:
        splits = json.load(f)
    with open(data_dir / scalers_file, "rb") as f:
        scalers = pickle.load(f)

    splitter = DataSplitter()

    def scaled(df, ids):
        return splitter.apply_scalers(df.loc[ids], scalers)

    def dataset(cls, df, ids):
        return cls(scaled(df, ids), lineage_to_idx=lineage_to_idx)

    ccle_train = dataset(CCLEDataset, ccle_df, splits["ccle"]["train"])
    ccle_val = dataset(CCLEDataset, ccle_df, splits["ccle"]["val"])
    tcga_train = dataset(TCGADataset, tcga_df, splits["tcga"]["train"])
    tcga_val = dataset(TCGADataset, tcga_df, splits["tcga"]["val"])

    typer.echo(
        f"Train: CCLE {len(ccle_train)} + TCGA {len(tcga_train)} | "
        f"Val: CCLE {len(ccle_val)} + TCGA {len(tcga_val)}"
    )

    sampler = StratifiedContrastiveBatchSampler(
        ccle_train, tcga_train, batch_size=train_cfg.get("batch_size", 48)
    )
    typer.echo(f"Sampler: {len(sampler)} batches/epoch, {sampler.per_lineage} per lineage/domain")

    model, loss_fn = _build_model(model_cfg)
    n_params = sum(p.numel() for p in model.parameters())
    typer.echo(f"Model: {n_params:,} params, init tau={loss_fn.tau.item():.4f}")

    run_name = train_cfg.get("mlflow_experiment", "pctrans-v1") if mlflow else None
    trainer = ContrastiveTrainer(
        model, loss_fn, sampler, ccle_val, tcga_val, train_cfg,
        mlflow_run_name=run_name, idx_to_lineage=idx_to_lineage,
    )

    typer.echo(f"Training for {n_epochs} epoch(s)...")
    result = trainer.train(n_epochs=n_epochs)

    first = result["history"][0]
    typer.echo("")
    typer.echo("=" * 46)
    typer.echo("           GATE 0 SANITY CHECK")
    typer.echo("=" * 46)
    typer.echo(f"Epoch 1 train loss:   {first['train_loss']:.4f}")
    typer.echo(f"Epoch 1 val loss:     {first['val_loss']:.4f}")
    typer.echo(f"Epoch 1 val kNN@{trainer.knn_k}:   {first['val_knn_accuracy']:.4f}")
    typer.echo(f"Epoch 1 temperature:  {first['temperature']:.4f}")
    finite = math.isfinite(first["train_loss"]) and math.isfinite(first["val_loss"])
    knn_positive = first["val_knn_accuracy"] > 0.0
    passed = finite and knn_positive
    typer.echo("-" * 46)
    typer.echo(f"Best val kNN@{trainer.knn_k}:      {result['best_val_knn_accuracy']:.4f} "
               f"(epoch {result['best_epoch'] + 1})")
    typer.echo(f"Final temperature:    {result['temperature']:.4f}")
    typer.echo(f"DECISION: {'PASS -> proceed to Week 2' if passed else 'FAIL -> debug'}")
    typer.echo("=" * 46)

    return {
        "gate0_pass": passed,
        "epoch1_train_loss": first["train_loss"],
        "epoch1_val_knn": first["val_knn_accuracy"],
        "epoch1_temperature": first["temperature"],
        "best_val_knn": result["best_val_knn_accuracy"],
        "best_epoch": result["best_epoch"],
    }


if __name__ == "__main__":
    app()
