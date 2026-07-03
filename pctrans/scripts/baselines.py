"""``pctrans-baselines`` CLI: real batch-correction baselines (Day 17).

Runs PCA/Harmony/ComBat/Scanorama + the supervised CCLE->TCGA classifier
ceiling on the same processed test split `pctrans-evaluate` scores the
contrastive model on, and writes `reports/baselines.json`. If
`reports/eval_summary.json` already exists, its contrastive kNN@5 (+ Wilson CI)
is folded into the same table so the comparison is one place, not two.
"""

import json
import pickle
from pathlib import Path

import pandas as pd
import typer

from pctrans.data.dataset import CCLEDataset, TCGADataset
from pctrans.data.preprocessor import DataSplitter
from pctrans.evaluation.baselines import (
    RANDOM_BASELINE,
    combat_knn,
    harmony_knn,
    pca_knn,
    scanorama_knn,
    supervised_ceiling,
)

app = typer.Typer()

# Real baselines only (excludes random + the supervised ceiling, which are
# reference points rather than alignment methods the contrastive model competes
# against).
REAL_BASELINE_KEYS = ("pca_knn", "harmony_knn", "combat_knn", "scanorama_knn")


@app.command()
def main(
    data_dir: str = "data/processed/",
    ccle_file: str = "ccle_2k.parquet",
    tcga_file: str = "tcga_2k.parquet",
    splits_file: str = "splits.json",
    scalers_file: str = "scalers.pkl",
    eval_summary: str = "reports/eval_summary.json",
    k: int = 5,
    output: str = "reports/baselines.json",
):
    """Compute every real baseline + the supervised ceiling; write `output`."""
    data_dir = Path(data_dir)
    ccle_df = pd.read_parquet(data_dir / ccle_file)
    tcga_df = pd.read_parquet(data_dir / tcga_file)
    with open(data_dir / splits_file, encoding="utf-8") as f:
        splits = json.load(f)
    with open(data_dir / scalers_file, "rb") as f:
        scalers = pickle.load(f)

    splitter = DataSplitter()

    def scaled(df, ids):
        return splitter.apply_scalers(df.loc[ids], scalers)

    ccle_train = CCLEDataset(scaled(ccle_df, splits["ccle"]["train"]))
    ccle_test = CCLEDataset(scaled(ccle_df, splits["ccle"]["test"]))
    tcga_test = TCGADataset(scaled(tcga_df, splits["tcga"]["test"]))
    typer.echo(f"Test set: CCLE {len(ccle_test)} + TCGA {len(tcga_test)}")

    results = {
        "random": RANDOM_BASELINE,
        "pca_knn": pca_knn(ccle_test, tcga_test, k=k),
        "harmony_knn": harmony_knn(ccle_test, tcga_test, k=k),
        "combat_knn": combat_knn(ccle_test, tcga_test, k=k),
        "scanorama_knn": scanorama_knn(ccle_test, tcga_test, k=k),
        "supervised_ceiling": supervised_ceiling(ccle_train, tcga_test),
    }

    contrastive, contrastive_ci = None, None
    eval_path = Path(eval_summary)
    if eval_path.exists():
        with open(eval_path, encoding="utf-8") as f:
            summ = json.load(f)
        contrastive = summ.get("overall_knn_accuracy")
        contrastive_ci = summ.get("knn_wilson_ci")

    real_values = [results[key] for key in REAL_BASELINE_KEYS if results[key] is not None]
    best_real_baseline = max(real_values) if real_values else None
    beats_best_baseline_by = (
        contrastive - best_real_baseline
        if contrastive is not None and best_real_baseline is not None
        else None
    )
    matches_supervised_ceiling = (
        abs(contrastive - results["supervised_ceiling"]) < 0.02
        if contrastive is not None
        else None
    )

    bar = "=" * 50
    typer.echo("")
    typer.echo(bar)
    typer.echo("      REAL BASELINES + SUPERVISED CEILING")
    typer.echo(bar)
    typer.echo(f"Random               {RANDOM_BASELINE * 100:5.1f}%")
    typer.echo(f"PCA+kNN              {results['pca_knn'] * 100:5.1f}%")
    for label, key in (
        ("ComBat+kNN", "combat_knn"),
        ("Harmony+kNN", "harmony_knn"),
        ("Scanorama+kNN", "scanorama_knn"),
    ):
        val = results[key]
        shown = f"{val * 100:5.1f}%" if val is not None else "  n/a (dep not installed)"
        typer.echo(f"{label:20s} {shown}")
    typer.echo(
        f"Supervised ceiling   {results['supervised_ceiling'] * 100:5.1f}%  "
        "(CCLE train -> TCGA test, no alignment)"
    )
    if contrastive is not None:
        ci_str = ""
        if contrastive_ci:
            ci_str = f"  (Wilson {contrastive_ci['low'] * 100:.1f}-{contrastive_ci['high'] * 100:.1f}%)"
        typer.echo(f"Contrastive (ours)   {contrastive * 100:5.1f}%{ci_str}")
        if best_real_baseline is not None:
            typer.echo(f"  beats best real baseline by {beats_best_baseline_by * 100:+.1f} pts")
        typer.echo(f"  matches supervised ceiling: {matches_supervised_ceiling}")
    else:
        typer.echo(f"Contrastive (ours)   n/a (run pctrans-evaluate first for {eval_summary})")
    typer.echo(bar)

    summary = {
        **results,
        "contrastive_knn5": contrastive,
        "contrastive_wilson_ci": contrastive_ci,
        "best_real_baseline": best_real_baseline,
        "beats_best_baseline_by": beats_best_baseline_by,
        "matches_supervised_ceiling": matches_supervised_ceiling,
    }
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    typer.echo(f"Wrote {out_path}")
    return summary


if __name__ == "__main__":
    app()
