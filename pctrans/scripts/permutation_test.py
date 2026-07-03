"""``pctrans-permutation-test`` CLI: the Day 21 label-shuffle negative control.

Every prior Phase-2 result (Days 15-20) has assumed the model's high cross-domain
kNN@5 reflects real CCLE<->TCGA lineage correspondence. This is the control that
tests that assumption directly: destroy the correspondence by shuffling which
TCGA lineage label goes with which sample, then check whether the metric
collapses to chance. If a "working model" can still hit a high score once the
labels no longer mean anything, the earlier results would be suspect.

Two variants, both built on ``pctrans.evaluation.stats.permutation_test``:

- **eval-only** (cheap): reuse the already-trained model's test embeddings
  (``embeddings_test_15.npz``), shuffle the TCGA test labels, and recompute
  kNN@5 with no retraining. This isolates *metric-level* chance -- how much
  apparent accuracy the majority-vote kNN formula alone hands out when the
  labels are meaningless.
- **retrain** (expensive): shuffle the TCGA lineage column before training,
  fit a short schedule from scratch, and evaluate on the (consistently)
  shuffled test split. This is the stronger claim: even given a genuine
  training run, the model cannot learn a correspondence that isn't there.

Both nulls are compared against the same real value (the actual trained
model's real, unshuffled test kNN@5) via ``permutation_test``'s empirical
p-value. Writes ``reports/permutation_test.json`` + ``reports/permutation_null.png``.
"""

import json
import pickle
import tempfile
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import typer
from torch.utils.data import DataLoader

matplotlib.use("Agg")  # headless PNG rendering, no display required

from pctrans.data.dataset import CCLEDataset, TCGADataset, build_lineage_maps  # noqa: E402
from pctrans.data.preprocessor import DataSplitter  # noqa: E402
from pctrans.data.sampler import StratifiedContrastiveBatchSampler  # noqa: E402
from pctrans.evaluation.knn import (  # noqa: E402
    knn_accuracy_from_embeddings,
    knn_retrieval_accuracy,
)
from pctrans.evaluation.stats import permutation_test  # noqa: E402
from pctrans.evaluation.viz import permutation_null_panel  # noqa: E402
from pctrans.scripts.train import _build_model, _load_yaml  # noqa: E402
from pctrans.training.trainer import ContrastiveTrainer  # noqa: E402

app = typer.Typer()

_DEFAULT_LINEAGES = ["LUAD", "BRCA", "SKCM"]


def _shuffle_lineage(df, rng, col="lineage"):
    """Return a copy of ``df`` with ``col`` reassigned by a random permutation.

    Reassigns which *sample* carries which lineage label; the multiset of
    labels (and therefore per-lineage counts) is unchanged, only the
    CCLE<->TCGA correspondence the contrastive loss relies on is destroyed.
    """
    shuffled = df.copy()
    shuffled[col] = rng.permutation(df[col].to_numpy())
    return shuffled


def _eval_only_null_generator(z_ccle, y_ccle, z_tcga, y_tcga, k, idx_to_lineage, lineage_order):
    def generator(rng):
        y_tcga_shuffled = rng.permutation(y_tcga)
        result = knn_accuracy_from_embeddings(
            z_ccle, y_ccle, z_tcga, y_tcga_shuffled,
            k=k, idx_to_lineage=idx_to_lineage, lineage_order=lineage_order,
        )
        return result["overall_accuracy"]

    return generator


def _retrain_null_generator(
    ccle_train, ccle_val, ccle_test,
    tcga_train_df, tcga_val_df, tcga_test_df,
    lineage_to_idx, idx_to_lineage, lineage_order,
    model_cfg, train_cfg, short_epochs, k, tmpdir,
):
    counter = {"n": 0}

    def generator(rng):
        counter["n"] += 1
        ckpt_path = Path(tmpdir) / f"perm_{counter['n']}.pt"

        tcga_train = TCGADataset(_shuffle_lineage(tcga_train_df, rng), lineage_to_idx=lineage_to_idx)
        tcga_val = TCGADataset(_shuffle_lineage(tcga_val_df, rng), lineage_to_idx=lineage_to_idx)
        tcga_test = TCGADataset(_shuffle_lineage(tcga_test_df, rng), lineage_to_idx=lineage_to_idx)

        sampler = StratifiedContrastiveBatchSampler(
            ccle_train, tcga_train, batch_size=train_cfg.get("batch_size", 48)
        )
        model, loss_fn = _build_model(model_cfg)
        run_cfg = dict(train_cfg)
        run_cfg["checkpoint_path"] = str(ckpt_path)
        # A short, fixed schedule: no early stopping inside it (a working model
        # must fail on the shuffled task even given the full budget).
        run_cfg["early_stop_patience"] = short_epochs + 1
        trainer = ContrastiveTrainer(
            model, loss_fn, sampler, ccle_val, tcga_val, run_cfg,
            mlflow_run_name=None, idx_to_lineage=idx_to_lineage,
        )
        trainer.train(n_epochs=short_epochs)
        trainer.load_checkpoint(ckpt_path)
        model.eval()

        ccle_loader = DataLoader(ccle_test, batch_size=256, shuffle=False)
        tcga_loader = DataLoader(tcga_test, batch_size=256, shuffle=False)
        knn = knn_retrieval_accuracy(
            model, ccle_loader, tcga_loader, k=k,
            idx_to_lineage=idx_to_lineage, lineage_order=lineage_order,
        )
        typer.echo(f"    retrain perm {counter['n']:2d}: kNN@{k} {knn['overall_accuracy'] * 100:5.1f}%")
        return knn["overall_accuracy"]

    return generator


@app.command()
def main(
    model_config: str = "configs/model.yaml",
    training_config: str = "configs/training_15.yaml",
    data_config: str = "configs/data_15.yaml",
    data_dir: str = "data/processed/",
    ccle_file: str = "ccle_2k_15.parquet",
    tcga_file: str = "tcga_2k_15.parquet",
    splits_file: str = "splits_15.json",
    scalers_file: str = "scalers_15.pkl",
    embeddings: str = "data/processed/embeddings_test_15.npz",
    n_perm: int = 20,
    short_epochs: int = 5,
    k: int = 5,
    seed: int = 0,
    output: str = "reports/permutation_test.json",
    figure: str = "reports/permutation_null.png",
):
    """Run the Day 21 label-shuffle negative control (eval-only + retrain variants).

    Defaults target the 15-lineage config (the current headline evaluation,
    Days 18-19) so the null sits at ~1/15 chance rather than the saturated
    3-lineage 100%; pass the ``*_2k.parquet``/``configs/data.yaml``/
    ``configs/training.yaml`` triple to run the same control on the Phase-1
    3-lineage pipeline instead.
    """
    data_dir = Path(data_dir)
    model_cfg = _load_yaml(model_config)
    train_cfg = _load_yaml(training_config)
    data_cfg = _load_yaml(data_config) if Path(data_config).exists() else {}
    lineages = data_cfg.get("lineages", _DEFAULT_LINEAGES)
    lineage_to_idx, idx_to_lineage = build_lineage_maps(lineages)
    chance_level = 1.0 / len(lineages)

    # -- Real value: the actual trained model's real (unshuffled) test kNN@5 --
    emb_path = Path(embeddings)
    emb = np.load(emb_path, allow_pickle=True)
    z_ccle, y_ccle = emb["z_ccle"], emb["y_ccle"]
    z_tcga, y_tcga = emb["z_tcga"], emb["y_tcga"]
    real_result = knn_accuracy_from_embeddings(
        z_ccle, y_ccle, z_tcga, y_tcga, k=k, idx_to_lineage=idx_to_lineage, lineage_order=lineages
    )
    real_value = real_result["overall_accuracy"]
    typer.echo(f"Loaded {emb_path}: CCLE {len(y_ccle)} + TCGA {len(y_tcga)}")
    typer.echo(f"Real (unshuffled) test kNN@{k}: {real_value * 100:.1f}%   (chance = {chance_level * 100:.1f}%)")

    # -- Variant 1: eval-only label shuffle (cheap, no retraining) -------------
    typer.echo(f"\nRunning eval-only label shuffle ({n_perm} permutations)...")
    eval_only_generator = _eval_only_null_generator(
        z_ccle, y_ccle, z_tcga, y_tcga, k, idx_to_lineage, lineages
    )
    eval_only_result = permutation_test(real_value, eval_only_generator, n_perm=n_perm, seed=seed)

    # -- Variant 2: retrain-based label shuffle (expensive, short schedule) ---
    typer.echo(f"\nRunning retrain-based label shuffle ({n_perm} permutations x {short_epochs} epochs)...")
    ccle_df = pd.read_parquet(data_dir / ccle_file)
    tcga_df = pd.read_parquet(data_dir / tcga_file)
    with open(data_dir / splits_file, encoding="utf-8") as f:
        splits = json.load(f)
    with open(data_dir / scalers_file, "rb") as f:
        scalers = pickle.load(f)

    splitter = DataSplitter()

    def scaled(df, ids):
        return splitter.apply_scalers(df.loc[ids], scalers)

    ccle_train = CCLEDataset(scaled(ccle_df, splits["ccle"]["train"]), lineage_to_idx=lineage_to_idx)
    ccle_val = CCLEDataset(scaled(ccle_df, splits["ccle"]["val"]), lineage_to_idx=lineage_to_idx)
    ccle_test = CCLEDataset(scaled(ccle_df, splits["ccle"]["test"]), lineage_to_idx=lineage_to_idx)
    tcga_train_df = scaled(tcga_df, splits["tcga"]["train"])
    tcga_val_df = scaled(tcga_df, splits["tcga"]["val"])
    tcga_test_df = scaled(tcga_df, splits["tcga"]["test"])

    with tempfile.TemporaryDirectory() as tmpdir:
        retrain_generator = _retrain_null_generator(
            ccle_train, ccle_val, ccle_test,
            tcga_train_df, tcga_val_df, tcga_test_df,
            lineage_to_idx, idx_to_lineage, lineages,
            model_cfg, train_cfg, short_epochs, k, tmpdir,
        )
        retrain_result = permutation_test(real_value, retrain_generator, n_perm=n_perm, seed=seed + 1000)

    # -- Report -----------------------------------------------------------------
    bar = "=" * 58
    typer.echo("")
    typer.echo(bar)
    typer.echo("     DAY 21 — LABEL-SHUFFLE NEGATIVE CONTROL REPORT")
    typer.echo(bar)
    typer.echo(f"Lineages: {len(lineages)}   Chance level: {chance_level * 100:.1f}%")
    typer.echo(f"Real test kNN@{k}: {real_value * 100:.1f}%")
    for name, result in (("Eval-only shuffle", eval_only_result), ("Retrain shuffle", retrain_result)):
        typer.echo(
            f"{name:20s} null mean {result['null_mean'] * 100:5.1f}%  "
            f"max {result['null_max'] * 100:5.1f}%  "
            f"(n_perm={result['n_perm']})  ->  p = {result['p_value']:.4f}"
        )
    decision = "PASS" if retrain_result["p_value"] < 0.01 else "INCONCLUSIVE"
    typer.echo(bar)
    typer.echo(f"DECISION: {decision}   [target p < 0.01 on the retrain variant]")
    typer.echo(bar)

    summary = {
        "lineages": lineages,
        "chance_level": chance_level,
        "k": k,
        "real_value": real_value,
        "short_epochs": short_epochs,
        "eval_only": eval_only_result,
        "retrain": retrain_result,
        "decision": decision,
    }
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    typer.echo(f"Wrote {out_path}")

    fig = permutation_null_panel(
        {"Eval-only label shuffle": eval_only_result, "Retrain label shuffle": retrain_result},
        chance_level=chance_level,
        title=f"Day 21 — label-shuffle negative control ({len(lineages)} lineages)",
    )
    fig_path = Path(figure)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    typer.echo(f"Wrote {fig_path}")

    return summary


if __name__ == "__main__":
    app()
