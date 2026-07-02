"""``pctrans-multiseed`` CLI: reproducibility across data splits (Day 15).

Phase 1's headline (test kNN@5 = 100%) came from a single lineage-stratified split
whose test set is only 38 cell lines. Varying the split seed changes *which* cell
lines land in test, so re-running the whole pipeline over many seeds is the honest
test of small-test-set stability: if 100% only held for one lucky split, the seed
sweep exposes it.

For each seed this re-runs **split -> fit scalers -> train -> test-eval**
end-to-end on the fixed 2,000-HVG feature space, reusing the default training
config with no per-seed tuning. (Per-split, train-only HVG *re-selection* is the
Day 16 refinement; here the HVG set is held fixed so the seed sweep isolates the
split effect.) It collects test kNN@{1,5}, silhouette, and TFS per seed and writes
`reports/multiseed_results.json` with per-seed rows plus aggregate mean/sd and a
bootstrap CI of the across-seed mean.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from torch.utils.data import DataLoader

from pctrans.data.dataset import CCLEDataset, TCGADataset
from pctrans.data.preprocessor import DataSplitter
from pctrans.data.sampler import StratifiedContrastiveBatchSampler
from pctrans.evaluation.knn import knn_retrieval_accuracy
from pctrans.evaluation.silhouette import cross_domain_silhouette
from pctrans.evaluation.stats import aggregate_seeds
from pctrans.evaluation.tfs import translational_fidelity_score
from pctrans.scripts.train import _build_model, _load_yaml
from pctrans.training.trainer import ContrastiveTrainer

app = typer.Typer()


def _eval_test(net, ccle_test, tcga_test, k=5):
    """kNN@{1,5} + silhouette + TFS for a trained model on a held-out test split."""
    ccle_loader = DataLoader(ccle_test, batch_size=256, shuffle=False)
    tcga_loader = DataLoader(tcga_test, batch_size=256, shuffle=False)
    knn = knn_retrieval_accuracy(net, ccle_loader, tcga_loader, k=k)

    emb = knn["embeddings"]
    pooled_z = np.concatenate([emb["z_ccle"], emb["z_tcga"]], axis=0)
    pooled_lineage = np.concatenate([emb["y_ccle"], emb["y_tcga"]])
    n_ccle = len(emb["y_ccle"])
    pooled_domain = np.array([0] * n_ccle + [1] * len(emb["y_tcga"]))

    silhouette = cross_domain_silhouette(pooled_z, pooled_lineage, pooled_domain)
    overall = knn["overall_accuracy"]
    return {
        "knn1": float(knn["k_table"].get(1, overall)),
        "knn5": float(overall),
        "silhouette": float(silhouette),
        "tfs": float(translational_fidelity_score(overall, silhouette)),
        "n_ccle_test": int(n_ccle),
        "n_tcga_test": int(len(emb["y_tcga"])),
    }


def run_one_seed(ccle_df, tcga_df, model_cfg, train_cfg, seed, val_frac, test_frac, k, ckpt_path):
    """Full pipeline for one split seed -> per-seed test metrics dict."""
    splitter = DataSplitter()
    splits = splitter.stratified_split(
        ccle_df, tcga_df, val_frac=val_frac, test_frac=test_frac, seed=seed
    )
    scalers = splitter.fit_scalers(
        ccle_df.loc[splits["ccle"]["train"]], tcga_df.loc[splits["tcga"]["train"]]
    )

    def scaled(df, ids):
        return splitter.apply_scalers(df.loc[ids], scalers)

    ccle_train = CCLEDataset(scaled(ccle_df, splits["ccle"]["train"]))
    ccle_val = CCLEDataset(scaled(ccle_df, splits["ccle"]["val"]))
    ccle_test = CCLEDataset(scaled(ccle_df, splits["ccle"]["test"]))
    tcga_train = TCGADataset(scaled(tcga_df, splits["tcga"]["train"]))
    tcga_val = TCGADataset(scaled(tcga_df, splits["tcga"]["val"]))
    tcga_test = TCGADataset(scaled(tcga_df, splits["tcga"]["test"]))

    sampler = StratifiedContrastiveBatchSampler(
        ccle_train, tcga_train, batch_size=train_cfg.get("batch_size", 48)
    )
    model, loss_fn = _build_model(model_cfg)

    # Per-seed temp checkpoint so the Phase-1 best_model.pt is never overwritten.
    run_cfg = dict(train_cfg)
    run_cfg["checkpoint_path"] = str(ckpt_path)
    trainer = ContrastiveTrainer(
        model, loss_fn, sampler, ccle_val, tcga_val, run_cfg, mlflow_run_name=None
    )
    result = trainer.train(n_epochs=run_cfg.get("n_epochs", 30))
    # Score the best checkpoint (by val kNN), not the final-epoch weights.
    trainer.load_checkpoint(ckpt_path)
    model.eval()

    metrics = _eval_test(model, ccle_test, tcga_test, k=k)
    metrics["seed"] = int(seed)
    metrics["best_epoch"] = int(result["best_epoch"])
    metrics["best_val_knn"] = float(result["best_val_knn_accuracy"])
    return metrics


@app.command()
def main(
    data_dir: str = "data/processed/",
    model_config: str = "configs/model.yaml",
    training_config: str = "configs/training.yaml",
    seed_start: int = 42,
    n_seeds: int = 10,
    k: int = 5,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    output: str = "reports/multiseed_results.json",
):
    """Re-run split->train->test-eval over ``n_seeds`` splits; report stability."""
    data_dir = Path(data_dir)
    model_cfg = _load_yaml(model_config)
    train_cfg = _load_yaml(training_config)

    ccle_df = pd.read_parquet(data_dir / "ccle_2k.parquet")
    tcga_df = pd.read_parquet(data_dir / "tcga_2k.parquet")
    seeds = list(range(seed_start, seed_start + n_seeds))
    typer.echo(f"Multi-seed reproducibility: seeds {seeds[0]}..{seeds[-1]} ({n_seeds} runs)")

    rows = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for seed in seeds:
            ckpt = Path(tmpdir) / f"seed_{seed}.pt"
            metrics = run_one_seed(
                ccle_df, tcga_df, model_cfg, train_cfg, seed,
                val_frac, test_frac, k, ckpt,
            )
            rows.append(metrics)
            typer.echo(
                f"  seed {seed}: kNN@5 {metrics['knn5'] * 100:5.1f}%  "
                f"kNN@1 {metrics['knn1'] * 100:5.1f}%  "
                f"sil {metrics['silhouette']:+.3f}  TFS {metrics['tfs']:.3f}  "
                f"(best epoch {metrics['best_epoch'] + 1}, n_test={metrics['n_ccle_test']})"
            )

    aggregate = {
        metric: aggregate_seeds([r[metric] for r in rows])
        for metric in ("knn5", "knn1", "silhouette", "tfs")
    }

    summary = {
        "seeds": seeds,
        "n_seeds": n_seeds,
        "k": k,
        "per_seed": rows,
        "aggregate": aggregate,
    }
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    bar = "=" * 54
    typer.echo("")
    typer.echo(bar)
    typer.echo("        MULTI-SEED REPRODUCIBILITY SUMMARY")
    typer.echo(bar)
    for name, key in (("kNN@5", "knn5"), ("kNN@1", "knn1"),
                      ("Silhouette", "silhouette"), ("TFS", "tfs")):
        a = aggregate[key]
        typer.echo(
            f"{name:11s} mean {a['mean']:.3f} +/- {a['sd']:.3f}  "
            f"CI [{a['ci_low']:.3f}, {a['ci_high']:.3f}]  "
            f"(min {a['min']:.3f}, max {a['max']:.3f})"
        )
    typer.echo(bar)
    typer.echo(f"Wrote {out_path}")
    return summary


if __name__ == "__main__":
    app()
