"""kNN@k cross-domain retrieval — the Day 10 Gate 1 primary metric.

`knn_retrieval_accuracy` embeds the held-out test CCLE and TCGA samples with the
frozen dual-tower model, then for each CCLE cell line finds its k nearest TCGA
patients in the 64-dim L2 space and checks whether the majority-vote neighbour
lineage matches the cell line's true lineage. This is the clinical question made
quantitative: *given a drug tested on this cell line, are its nearest human
analogues the right patient population?*

The heavy lifting lives in `knn_accuracy_from_embeddings`, which operates on
plain arrays so it can be unit-tested on synthetic (random / perfect) embeddings
without a model. It also returns the per-CCLE-sample match fraction that the
per-cell-line TFS (see `pctrans.evaluation.tfs`) is built from.
"""

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import NearestNeighbors

from pctrans.data.dataset import IDX_TO_LINEAGE, LINEAGE_TO_IDX

LINEAGE_ORDER = ["LUAD", "BRCA", "SKCM"]
DEFAULT_K_VALUES = (1, 3, 5, 10)


def embed_loader(encode_fn, loader):
    """Run a frozen encoder over a loader, returning ``(embeddings, labels)``."""
    embeddings, labels = [], []
    with torch.no_grad():
        for features, label in loader:
            embeddings.append(encode_fn(features).cpu().numpy())
            labels.append(np.asarray(label))
    return np.concatenate(embeddings), np.concatenate(labels)


def _row_majority(labels_2d):
    """Per-row majority label of an ``(n, k)`` integer array (ties -> lowest label)."""
    preds = np.empty(labels_2d.shape[0], dtype=labels_2d.dtype)
    for i, row in enumerate(labels_2d):
        values, counts = np.unique(row, return_counts=True)
        preds[i] = values[counts.argmax()]
    return preds


def knn_accuracy_from_embeddings(
    z_ccle, y_ccle, z_tcga, y_tcga, k=5, k_values=DEFAULT_K_VALUES
):
    """kNN retrieval accuracy of CCLE anchors against a TCGA gallery.

    Returns a dict with the overall accuracy at ``k``, per-lineage accuracy, a
    confusion matrix (rows = true CCLE lineage, cols = predicted), a supplementary
    ``k_table`` at ``k_values``, and the per-CCLE-sample neighbour match fraction
    at ``k`` (used for per-cell-line TFS).
    """
    z_ccle = np.asarray(z_ccle, dtype=np.float64)
    z_tcga = np.asarray(z_tcga, dtype=np.float64)
    y_ccle = np.asarray(y_ccle)
    y_tcga = np.asarray(y_tcga)

    n_ccle = len(y_ccle)
    # k can never exceed the TCGA gallery size.
    k_max = min(max(max(k_values), k), len(z_tcga))
    neigh_idx = (
        NearestNeighbors(n_neighbors=k_max, metric="euclidean")
        .fit(z_tcga)
        .kneighbors(z_ccle, return_distance=False)
    )
    neigh_labels = y_tcga[neigh_idx]  # (n_ccle, k_max)

    k_eff = min(k, k_max)
    # Fraction of the k neighbours whose lineage matches the anchor (per sample).
    match_fraction = (neigh_labels[:, :k_eff] == y_ccle[:, None]).mean(axis=1)
    preds = _row_majority(neigh_labels[:, :k_eff])
    correct = preds == y_ccle

    k_table = {}
    for kk in sorted(set(k_values) | {k}):
        ke = min(kk, k_max)
        p = _row_majority(neigh_labels[:, :ke])
        k_table[kk] = float((p == y_ccle).mean()) if n_ccle else 0.0

    per_lineage = {}
    for idx, lineage in IDX_TO_LINEAGE.items():
        mask = y_ccle == idx
        if mask.any():
            per_lineage[lineage] = float(correct[mask].mean())

    label_ids = [LINEAGE_TO_IDX[lineage] for lineage in LINEAGE_ORDER]
    cm = confusion_matrix(y_ccle, preds, labels=label_ids).tolist()

    return {
        "k": k_eff,
        "overall_accuracy": float(correct.mean()) if n_ccle else 0.0,
        "per_lineage": per_lineage,
        "confusion_matrix": cm,
        "confusion_labels": LINEAGE_ORDER,
        "k_table": k_table,
        "match_fraction": match_fraction,
        "ccle_labels": y_ccle,
    }


def knn_retrieval_accuracy(
    model, ccle_test_loader, tcga_test_loader, k=5, k_values=DEFAULT_K_VALUES
):
    """Embed the frozen model's test sets and score kNN@k retrieval.

    The returned dict is that of `knn_accuracy_from_embeddings` plus an
    ``embeddings`` entry (pooled arrays) so callers can reuse the embeddings for
    the silhouette score without a second forward pass.
    """
    was_training = model.training
    model.eval()
    z_ccle, y_ccle = embed_loader(model.encode_ccle, ccle_test_loader)
    z_tcga, y_tcga = embed_loader(model.encode_tcga, tcga_test_loader)
    if was_training:
        model.train()

    result = knn_accuracy_from_embeddings(
        z_ccle, y_ccle, z_tcga, y_tcga, k=k, k_values=k_values
    )
    result["embeddings"] = {
        "z_ccle": z_ccle,
        "y_ccle": y_ccle,
        "z_tcga": z_tcga,
        "y_tcga": y_tcga,
    }
    return result
