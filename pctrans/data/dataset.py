"""PyTorch datasets for the two domains.

Both `CCLEDataset` and `TCGADataset` wrap a processed expression frame
(genes as float columns + a trailing lineage column) and expose the same
interface: `__getitem__` returns `(features_tensor, lineage_label_int)`.
Lineage strings are encoded via the module-level `LINEAGE_TO_IDX` map.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

LINEAGE_TO_IDX = {"LUAD": 0, "BRCA": 1, "SKCM": 2}
IDX_TO_LINEAGE = {idx: lineage for lineage, idx in LINEAGE_TO_IDX.items()}


class _ExpressionDataset(Dataset):
    """Shared implementation for both domains (identical feature interface)."""

    def __init__(self, expr_df, lineage_col="lineage"):
        if lineage_col not in expr_df.columns:
            raise ValueError(f"expr_df has no lineage column {lineage_col!r}")

        self.feature_names = [c for c in expr_df.columns if c != lineage_col]
        self.features = torch.tensor(
            expr_df[self.feature_names].to_numpy(dtype=np.float32)
        )
        self.labels = expr_df[lineage_col].map(LINEAGE_TO_IDX).to_numpy()
        if np.isnan(self.labels.astype(float)).any():
            unknown = sorted(set(expr_df[lineage_col]) - set(LINEAGE_TO_IDX))
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
