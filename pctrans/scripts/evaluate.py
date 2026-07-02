"""``pctrans-evaluate`` CLI: the Day 10 Gate 1 evaluation.

Loads ``models/best_model.pt`` and the processed test split, then computes the
held-out kNN@{1,3,5,10} retrieval accuracy, the cross-domain silhouette, the
composite TFS (overall + per cell line), and the required baselines (random +
PCA-then-kNN with no alignment). It prints the Gate 1 report, applies the
non-negotiable ≥70% DEPLOY threshold, and writes ``reports/eval_summary.json``.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import typer
import yaml
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

from pctrans.data.dataset import IDX_TO_LINEAGE, CCLEDataset, TCGADataset
from pctrans.data.preprocessor import DataSplitter
from pctrans.evaluation.knn import knn_accuracy_from_embeddings, knn_retrieval_accuracy
from pctrans.evaluation.silhouette import (
    cross_domain_silhouette,
    silhouette_contributions,
)
from pctrans.evaluation.stats import bootstrap_ci, bootstrap_metric_ci, wilson_ci
from pctrans.evaluation.tfs import per_cell_line_tfs, translational_fidelity_score
from pctrans.models.dual_tower import DualTowerModel
from pctrans.models.encoders import CCLEEncoder, TCGAEncoder

app = typer.Typer()

RANDOM_BASELINE = 1.0 / 3.0
HARMONY_BASELINE = 0.63  # literature reference (harmonypy not a project dependency)
DEPLOY_THRESHOLD = 0.70


def _load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_model(model_cfg):
    kwargs = {
        "input_dim": model_cfg.get("input_dim", 2000),
        "hidden_dims": tuple(model_cfg.get("hidden_dims", [1024, 512, 256, 128])),
        "embed_dim": model_cfg.get("embed_dim", 64),
        "dropout": model_cfg.get("dropout_high", 0.3),
        "dropout_low": model_cfg.get("dropout_low", 0.2),
    }
    return DualTowerModel(CCLEEncoder(**kwargs), TCGAEncoder(**kwargs))


def _decision(accuracy):
    """Map test kNN@5 accuracy onto the PLAN.md Gate 1 decision bands."""
    if accuracy >= DEPLOY_THRESHOLD:
        return "DEPLOY", "PASS (>=70% -> deploy path, Days 11-14)"
    if accuracy >= 0.60:
        return "DEBUG", "SOFT FAIL (60-70% -> one fix + 10-epoch rerun)"
    if accuracy >= 0.50:
        return "DEBUG", "HARD FAIL (50-60% -> batch-construction fix)"
    return "DEBUG", "ARCHITECTURE FAILURE (<50% -> pivot)"


def _pca_knn_baseline(ccle_ds, tcga_ds, n_components=50, k=5):
    """No-alignment baseline: PCA on pooled raw features, then cross-domain kNN."""
    x_ccle = ccle_ds.features.numpy()
    x_tcga = tcga_ds.features.numpy()
    pooled = np.concatenate([x_ccle, x_tcga], axis=0)
    n_comp = min(n_components, pooled.shape[0], pooled.shape[1])
    coords = PCA(n_components=n_comp, random_state=42).fit_transform(pooled)
    z_ccle, z_tcga = coords[: len(x_ccle)], coords[len(x_ccle):]
    res = knn_accuracy_from_embeddings(
        z_ccle, ccle_ds.labels, z_tcga, tcga_ds.labels, k=k
    )
    return res["overall_accuracy"]


@app.command()
def main(
    model: str = "models/best_model.pt",
    data_dir: str = "data/processed/",
    model_config: str = "configs/model.yaml",
    k: int = 5,
    output: str = "reports/eval_summary.json",
):
    """Evaluate the trained dual-tower model on the held-out test set (Gate 1)."""
    data_dir = Path(data_dir)
    model_cfg = _load_yaml(model_config)

    ccle_df = pd.read_parquet(data_dir / "ccle_2k.parquet")
    tcga_df = pd.read_parquet(data_dir / "tcga_2k.parquet")
    with open(data_dir / "splits.json", encoding="utf-8") as f:
        splits = json.load(f)
    with open(data_dir / "scalers.pkl", "rb") as f:
        scalers = pickle.load(f)

    splitter = DataSplitter()

    def scaled(df, ids):
        return splitter.apply_scalers(df.loc[ids], scalers)

    ccle_test = CCLEDataset(scaled(ccle_df, splits["ccle"]["test"]))
    tcga_test = TCGADataset(scaled(tcga_df, splits["tcga"]["test"]))
    typer.echo(f"Test set: CCLE {len(ccle_test)} + TCGA {len(tcga_test)}")

    net = _build_model(model_cfg)
    checkpoint = torch.load(model, map_location="cpu", weights_only=False)
    net.load_state_dict(checkpoint["model_state_dict"])
    net.eval()
    ckpt_epoch = checkpoint.get("epoch")
    ckpt_val_knn = checkpoint.get("val_knn_accuracy")

    ccle_loader = DataLoader(ccle_test, batch_size=256, shuffle=False)
    tcga_loader = DataLoader(tcga_test, batch_size=256, shuffle=False)

    knn = knn_retrieval_accuracy(net, ccle_loader, tcga_loader, k=k)
    overall = knn["overall_accuracy"]

    # Pool embeddings (CCLE first) for silhouette + per-cell-line TFS.
    emb = knn["embeddings"]
    pooled_z = np.concatenate([emb["z_ccle"], emb["z_tcga"]], axis=0)
    pooled_lineage = np.concatenate([emb["y_ccle"], emb["y_tcga"]])
    n_ccle = len(emb["y_ccle"])
    pooled_domain = np.array([0] * n_ccle + [1] * len(emb["y_tcga"]))

    silhouette = cross_domain_silhouette(pooled_z, pooled_lineage, pooled_domain)
    tfs_overall = float(translational_fidelity_score(overall, silhouette))

    sil_samples = silhouette_contributions(pooled_z, pooled_lineage)
    ccle_sil = sil_samples[:n_ccle]
    ccle_tfs = per_cell_line_tfs(knn["match_fraction"], ccle_sil)

    # Day 15: attach intervals to every point metric. Wilson is the analytic
    # binomial interval for the retrieval proportion (correct for the ~38-anchor
    # n); the bootstrap intervals resample anchors (kNN) / pooled samples
    # (silhouette) with replacement and make no distributional assumption.
    n_anchors = n_ccle
    successes = int(round(overall * n_anchors))
    knn_wilson = wilson_ci(successes, n_anchors)
    knn_boot = bootstrap_metric_ci(knn["match_fraction"])
    sil_boot = bootstrap_ci(sil_samples, np.mean)

    per_cell_line = [
        {
            "id": str(cid),
            "lineage": IDX_TO_LINEAGE[int(lbl)],
            "knn_match_fraction": float(mf),
            "silhouette": float(sc),
            "tfs": float(tfs),
        }
        for cid, lbl, mf, sc, tfs in zip(
            ccle_test.ids, emb["y_ccle"], knn["match_fraction"], ccle_sil, ccle_tfs
        )
    ]
    per_cell_line.sort(key=lambda r: r["tfs"], reverse=True)

    pca_baseline = _pca_knn_baseline(ccle_test, tcga_test, k=k)
    decision, band = _decision(overall)

    # -- Gate 1 report ---------------------------------------------------------
    bar = "=" * 46
    typer.echo("")
    typer.echo(bar)
    typer.echo("           GATE 1 EVALUATION REPORT")
    typer.echo(bar)
    typer.echo(
        f"Overall kNN@{knn['k']} Accuracy:  {overall * 100:5.1f}%   (threshold: 70%)"
    )
    typer.echo(
        f"  Wilson 95% CI:   {knn_wilson[0] * 100:5.1f}-{knn_wilson[1] * 100:5.1f}%  "
        f"(n={n_anchors})"
    )
    typer.echo(
        f"  Bootstrap 95% CI: {knn_boot['ci_low'] * 100:5.1f}-{knn_boot['ci_high'] * 100:5.1f}%"
    )
    typer.echo("Per-lineage kNN@{}:".format(knn["k"]))
    for lineage in knn["confusion_labels"]:
        val = knn["per_lineage"].get(lineage)
        shown = f"{val * 100:5.1f}%" if val is not None else "  n/a"
        typer.echo(f"  {lineage}:  {shown}")
    typer.echo("kNN@k table:  " + "  ".join(
        f"k={kk}:{acc * 100:.1f}%" for kk, acc in sorted(knn["k_table"].items())
    ))
    typer.echo(
        f"Silhouette Score:  {silhouette:+.2f}   (> 0 = good alignment)  "
        f"[boot 95% CI {sil_boot['ci_low']:+.2f}, {sil_boot['ci_high']:+.2f}]"
    )
    typer.echo(f"TFS (composite):   {tfs_overall:.2f}    (> 0.6 = deploy)")
    typer.echo(f"Random baseline:   {RANDOM_BASELINE * 100:.1f}%")
    typer.echo(f"PCA+kNN baseline:  {pca_baseline * 100:.1f}%")
    typer.echo(f"Harmony baseline:  ~{HARMONY_BASELINE * 100:.0f}%  (literature)")
    typer.echo(bar)
    typer.echo(f"DECISION: {decision}   [{band}]")
    typer.echo(bar)

    summary = {
        "model": str(model),
        "checkpoint_epoch": ckpt_epoch,
        "checkpoint_val_knn_accuracy": ckpt_val_knn,
        "test_sizes": {"ccle": len(ccle_test), "tcga": len(tcga_test)},
        "k": knn["k"],
        "overall_knn_accuracy": overall,
        "knn_wilson_ci": {"low": knn_wilson[0], "high": knn_wilson[1], "n": n_anchors},
        "knn_bootstrap_ci": {"low": knn_boot["ci_low"], "high": knn_boot["ci_high"]},
        "silhouette_bootstrap_ci": {"low": sil_boot["ci_low"], "high": sil_boot["ci_high"]},
        "per_lineage_knn": knn["per_lineage"],
        "knn_k_table": knn["k_table"],
        "confusion_matrix": knn["confusion_matrix"],
        "confusion_labels": knn["confusion_labels"],
        "silhouette_score": silhouette,
        "tfs_overall": tfs_overall,
        "baselines": {
            "random": RANDOM_BASELINE,
            "pca_knn": pca_baseline,
            "harmony_reference": HARMONY_BASELINE,
        },
        "decision": decision,
        "decision_band": band,
        "per_cell_line_tfs": per_cell_line,
    }
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    typer.echo(f"Wrote {out_path}")

    return summary


if __name__ == "__main__":
    app()
