"""Day 13 inference-API tests: TranslationEmbedder over a trained checkpoint."""

import numpy as np
import pytest

from pctrans.data.dataset import IDX_TO_LINEAGE
from pctrans.inference.api import TranslationEmbedder


def _embedder(pipeline):
    return TranslationEmbedder(
        pipeline["model"],
        data_dir=pipeline["data_dir"],
        model_config=pipeline["model_config"],
    )


def test_translation_embedder_loads_checkpoint(pipeline):
    embedder = _embedder(pipeline)
    assert embedder.checkpoint_epoch is not None
    assert len(embedder.cell_line_ids) == pipeline["n_ccle"]


def test_embed_cell_line_returns_shape_1_64(pipeline):
    embedder = _embedder(pipeline)
    z = embedder.embed_cell_line(pipeline["any_ccle_id"])
    assert z.shape == (1, 64)
    # Embeddings live on the unit hypersphere (L2-normalised by DualTowerModel).
    assert np.isclose(np.linalg.norm(z), 1.0, atol=1e-5)


def test_embed_cell_line_unknown_id_raises(pipeline):
    embedder = _embedder(pipeline)
    with pytest.raises(KeyError, match="unknown cell line"):
        embedder.embed_cell_line("ACH-NOT-A-REAL-ID")


def test_query_patients_returns_k_neighbours(pipeline):
    embedder = _embedder(pipeline)
    neighbours = embedder.query_patients(pipeline["any_ccle_id"], k=5)
    assert len(neighbours) == 5
    valid_lineages = set(IDX_TO_LINEAGE.values())
    for neighbour in neighbours:
        assert set(neighbour) == {"patient_id", "lineage", "distance"}
        assert neighbour["lineage"] in valid_lineages
        assert neighbour["distance"] >= 0.0
    # Distances are returned nearest-first.
    dists = [n["distance"] for n in neighbours]
    assert dists == sorted(dists)


def test_query_patients_clamps_k_to_gallery(pipeline):
    embedder = _embedder(pipeline)
    neighbours = embedder.query_patients(pipeline["any_ccle_id"], k=10_000)
    assert 0 < len(neighbours) <= 10_000
