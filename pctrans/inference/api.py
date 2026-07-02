"""Cell-line -> patient inference API (Day 13).

`TranslationEmbedder` wraps a trained dual-tower checkpoint plus the processed
CCLE catalogue so a caller can, given a CCLE ``ModelID``:

  * `embed_cell_line` — return the cell line's 64-d aligned embedding (the same
    z-scored-input, no-grad forward the evaluation code uses), and
  * `query_patients` — retrieve its k nearest TCGA patients in that shared space.

This is the programmatic core the Streamlit app and the ``pctrans-query`` CLI sit
on top of. It reuses the *train-fit* scalers (``scalers.pkl``) so a query is
z-scored exactly as training was — no leakage, no re-fitting at serve time.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.neighbors import NearestNeighbors

from pctrans.data.dataset import IDX_TO_LINEAGE
from pctrans.data.preprocessor import DataSplitter
from pctrans.models.dual_tower import DualTowerModel
from pctrans.models.encoders import CCLEEncoder, TCGAEncoder


def _load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_model(model_cfg):
    """Reconstruct the dual-tower model from a model-config dict (weights untrained)."""
    kwargs = {
        "input_dim": model_cfg.get("input_dim", 2000),
        "hidden_dims": tuple(model_cfg.get("hidden_dims", [1024, 512, 256, 128])),
        "embed_dim": model_cfg.get("embed_dim", 64),
        "dropout": model_cfg.get("dropout_high", 0.3),
        "dropout_low": model_cfg.get("dropout_low", 0.2),
    }
    return DualTowerModel(CCLEEncoder(**kwargs), TCGAEncoder(**kwargs))


class TranslationEmbedder:
    """Frozen dual-tower inference over the processed CCLE catalogue.

    Parameters
    ----------
    checkpoint_path : str | Path
        A checkpoint written by `ContrastiveTrainer` (has ``model_state_dict``).
    data_dir : str | Path
        Processed-data directory holding ``ccle_2k.parquet`` and ``scalers.pkl``
        (and, for `query_patients`, the ``embeddings_test.npz`` TCGA gallery).
    model_config : str | Path
        YAML with the encoder dimensions the checkpoint was trained with.
    """

    def __init__(
        self,
        checkpoint_path,
        data_dir="data/processed/",
        model_config="configs/model.yaml",
    ):
        self.data_dir = Path(data_dir)
        self.model = _build_model(_load_yaml(model_config))
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.checkpoint_epoch = checkpoint.get("epoch")
        self.checkpoint_val_knn = checkpoint.get("val_knn_accuracy")

        self._ccle_df = pd.read_parquet(self.data_dir / "ccle_2k.parquet")
        with open(self.data_dir / "scalers.pkl", "rb") as f:
            self._scalers = pickle.load(f)
        self._splitter = DataSplitter()
        self._feature_cols = self._scalers["feature_cols"]

    @property
    def cell_line_ids(self):
        """All CCLE ModelIDs available to embed (the processed catalogue index)."""
        return [str(i) for i in self._ccle_df.index]

    def embed_cell_line(self, cell_line_id):
        """Embed one CCLE cell line -> ``(1, embed_dim)`` L2-normalised array."""
        if cell_line_id not in self._ccle_df.index:
            raise KeyError(f"unknown cell line id: {cell_line_id!r}")
        row = self._ccle_df.loc[[cell_line_id]]
        scaled = self._splitter.apply_scalers(row, self._scalers)
        feats = torch.tensor(scaled[self._feature_cols].to_numpy(dtype=np.float32))
        with torch.no_grad():
            z = self.model.encode_ccle(feats).cpu().numpy()
        return z

    def _tcga_gallery(self):
        """Load the precomputed TCGA test embeddings (z, lineage, ids)."""
        with np.load(self.data_dir / "embeddings_test.npz", allow_pickle=True) as d:
            return (
                np.asarray(d["z_tcga"], dtype=np.float64),
                np.asarray(d["y_tcga"]),
                np.array([str(s) for s in d["ids_tcga"]]),
            )

    def query_patients(self, cell_line_id, k=5):
        """Return the ``k`` nearest TCGA patients to ``cell_line_id``.

        Each entry is ``{"patient_id", "lineage", "distance"}``, ordered nearest
        first. ``k`` is clamped to the gallery size.
        """
        z = self.embed_cell_line(cell_line_id).astype(np.float64)
        z_tcga, y_tcga, ids_tcga = self._tcga_gallery()
        k = min(k, len(z_tcga))
        neigh = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(z_tcga)
        dist, idx = neigh.kneighbors(z)
        return [
            {
                "patient_id": str(ids_tcga[i]),
                "lineage": IDX_TO_LINEAGE[int(y_tcga[i])],
                "distance": float(dist[0][rank]),
            }
            for rank, i in enumerate(idx[0])
        ]
