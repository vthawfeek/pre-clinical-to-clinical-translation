"""PyTorch datasets for the two domains.

Both `CCLEDataset` and `TCGADataset` wrap a processed expression frame
(genes as float columns + a trailing lineage column) and expose the same
interface: `__getitem__` returns `(features_tensor, lineage_label_int)`.
Lineage strings are encoded via a `{lineage: idx}` map, defaulting to the
module-level `LINEAGE_TO_IDX` (see `build_lineage_maps` for the config-driven
15-lineage variant introduced on Day 18).
"""

import numpy as np
import torch
from torch.utils.data import Dataset


def build_lineage_maps(lineages) -> tuple[dict, dict]:
    """Build `{lineage: idx}` / `{idx: lineage}` maps from an ordered lineage list.

    Index assignment follows the list order given (no re-sorting), so the
    Phase-1 default below reproduces the original hardcoded
    ``{"LUAD": 0, "BRCA": 1, "SKCM": 2}`` byte-for-byte -- existing checkpoints
    and committed embeddings (``ccle_embeddings.npz``, ``embeddings_test.npz``)
    depend on that exact label-id convention.
    """
    lineage_to_idx = {lineage: idx for idx, lineage in enumerate(lineages)}
    idx_to_lineage = {idx: lineage for lineage, idx in lineage_to_idx.items()}
    return lineage_to_idx, idx_to_lineage


LINEAGE_TO_IDX, IDX_TO_LINEAGE = build_lineage_maps(["LUAD", "BRCA", "SKCM"])


class _ExpressionDataset(Dataset):
    """Shared implementation for both domains (identical feature interface)."""

    def __init__(self, expr_df, lineage_col="lineage", lineage_to_idx=None):
        if lineage_col not in expr_df.columns:
            raise ValueError(f"expr_df has no lineage column {lineage_col!r}")
        if lineage_to_idx is None:
            lineage_to_idx = LINEAGE_TO_IDX

        self.lineage_to_idx = lineage_to_idx
        self.feature_names = [c for c in expr_df.columns if c != lineage_col]
        self.features = torch.tensor(
            expr_df[self.feature_names].to_numpy(dtype=np.float32)
        )
        self.labels = expr_df[lineage_col].map(lineage_to_idx).to_numpy()
        if np.isnan(self.labels.astype(float)).any():
            unknown = sorted(set(expr_df[lineage_col]) - set(lineage_to_idx))
            raise ValueError(f"unknown lineage label(s): {unknown}")
        self.labels = self.labels.astype(np.int64)
        self.ids = expr_df.index.to_numpy()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], int(self.labels[idx])


class CCLEDataset(_ExpressionDataset):
    """CCLE cell-line expression (small N, oversampled by the batch sampler)."""


class TCGADataset(_ExpressionDataset):
    """TCGA patient expression (large N, undersampled by the batch sampler)."""
