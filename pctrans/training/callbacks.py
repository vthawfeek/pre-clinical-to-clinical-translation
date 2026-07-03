"""Validation callbacks for the contrastive trainer.

`KNNValidationCallback` measures cross-domain retrieval quality: it embeds all
validation CCLE and TCGA samples with the frozen model, then for each CCLE
sample finds its k nearest TCGA neighbours in the 64-dim L2 space and checks
whether the majority-vote neighbour lineage matches the CCLE lineage. This is
the metric the trainer early-stops and checkpoints on, and a proxy for the
Day 10 Gate 1 test-set kNN@5.
"""

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

from pctrans.data.dataset import IDX_TO_LINEAGE


def _row_majority(labels_2d):
    """Per-row majority label of an (n, k) integer array.

    Ties resolve to the lowest label: ``np.unique`` returns values ascending and
    ``argmax`` picks the first maximal count, so the result is deterministic.
    """
    preds = np.empty(labels_2d.shape[0], dtype=labels_2d.dtype)
    for i, row in enumerate(labels_2d):
        values, counts = np.unique(row, return_counts=True)
        preds[i] = values[counts.argmax()]
    return preds


class KNNValidationCallback:
    """kNN@k cross-domain retrieval accuracy on validation embeddings."""

    def __init__(self, k=5, idx_to_lineage=None):
        self.k = k
        self.idx_to_lineage = idx_to_lineage if idx_to_lineage is not None else IDX_TO_LINEAGE

    @staticmethod
    def _embed(encode_fn, loader):
        """Run the frozen encoder over a loader, returning (embeddings, labels)."""
        embeddings, labels = [], []
        for features, label in loader:
            embeddings.append(encode_fn(features).cpu().numpy())
            labels.append(np.asarray(label))
        return np.concatenate(embeddings), np.concatenate(labels)

    def __call__(self, model, val_ccle_loader, val_tcga_loader, k=None):
        k = self.k if k is None else k
        was_training = model.training
        model.eval()
        with torch.no_grad():
            z_ccle, y_ccle = self._embed(model.encode_ccle, val_ccle_loader)
            z_tcga, y_tcga = self._embed(model.encode_tcga, val_tcga_loader)
        if was_training:
            model.train()

        # k can never exceed the size of the TCGA gallery.
        k_eff = min(k, len(z_tcga))
        neighbours = (
            NearestNeighbors(n_neighbors=k_eff, metric="euclidean")
            .fit(z_tcga)
            .kneighbors(z_ccle, return_distance=False)
        )
        preds = _row_majority(y_tcga[neighbours])
        correct = preds == y_ccle

        per_lineage = {}
        for idx, lineage in self.idx_to_lineage.items():
            mask = y_ccle == idx
            if mask.any():
                per_lineage[lineage] = float(correct[mask].mean())

        return {
            "val_knn_accuracy": float(correct.mean()) if correct.size else 0.0,
            "per_lineage": per_lineage,
            "k": k_eff,
        }
