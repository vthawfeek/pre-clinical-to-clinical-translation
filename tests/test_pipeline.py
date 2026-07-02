"""Day 13 end-to-end CLI tests: run the train -> evaluate -> visualize -> query
pipeline against the synthetic ``pipeline`` fixture (tiny model, real weights).

These drive the actual ``main()`` bodies of the Typer scripts (not just ``--help``)
so the CLI code paths are covered. The UMAP-bearing visualize test is marked
``slow`` because ``umap_projection`` triggers the numba JIT.
"""

from pathlib import Path

import numpy as np
import pytest

from pctrans.scripts.evaluate import main as evaluate_main
from pctrans.scripts.precompute import main as precompute_main
from pctrans.scripts.query import main as query_main
from pctrans.scripts.visualize import main as visualize_main


def _run_evaluate(pipeline):
    return evaluate_main(
        model=pipeline["model"],
        data_dir=pipeline["data_dir"],
        model_config=pipeline["model_config"],
        k=5,
        output=pipeline["eval_summary"],
    )


def test_evaluate_cli_writes_summary_and_decides(pipeline):
    summary = _run_evaluate(pipeline)
    assert summary["decision"] in {"DEPLOY", "DEBUG"}
    assert 0.0 <= summary["overall_knn_accuracy"] <= 1.0
    assert set(summary["per_lineage_knn"]) <= {"LUAD", "BRCA", "SKCM"}
    assert Path(pipeline["eval_summary"]).exists()
    # Per-cell-line TFS is ranked highest-first.
    tfs = [r["tfs"] for r in summary["per_cell_line_tfs"]]
    assert tfs == sorted(tfs, reverse=True)


def test_precompute_cli_writes_catalogue(pipeline):
    result = precompute_main(
        model=pipeline["model"],
        data_dir=pipeline["data_dir"],
        raw_dir=str(Path(pipeline["data_dir"]) / "no_raw"),  # absent -> degrades gracefully
        model_config=pipeline["model_config"],
    )
    assert result["n_ccle"] == pipeline["n_ccle"]
    emb_path = Path(pipeline["data_dir"]) / "ccle_embeddings.npz"
    assert emb_path.exists()
    with np.load(emb_path, allow_pickle=True) as d:
        assert d["z"].shape[0] == pipeline["n_ccle"]
        assert d["z"].shape[1] == 64
    assert (Path(pipeline["data_dir"]) / "app_meta.json").exists()


def test_query_cli_prints_neighbours(pipeline):
    result = query_main(
        cell_line=pipeline["any_ccle_id"],
        model=pipeline["model"],
        data_dir=pipeline["data_dir"],
        model_config=pipeline["model_config"],
        k=3,
    )
    assert result["embedding_dim"] == 64
    assert len(result["neighbours"]) == 3


@pytest.mark.slow
def test_visualize_cli_renders_figures(pipeline):
    # eval_summary drives the TFS-ranking branch, so evaluate must run first.
    _run_evaluate(pipeline)
    result = visualize_main(
        model=pipeline["model"],
        data_dir=pipeline["data_dir"],
        model_config=pipeline["model_config"],
        eval_summary=pipeline["eval_summary"],
        reports_dir=pipeline["reports_dir"],
        seed=42,
    )
    assert result["coords_shape"][1] == 2
    reports = Path(pipeline["reports_dir"])
    assert (reports / "umap_test_set.html").exists()
    assert (reports / "umap_before_after.png").exists()
    assert (Path(pipeline["data_dir"]) / "embeddings_test.npz").exists()
