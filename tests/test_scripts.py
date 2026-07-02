"""Smoke tests for the Typer CLI entry points (``--help`` exits 0)."""

import pytest
from typer.testing import CliRunner

from pctrans.scripts.download import app as download_app
from pctrans.scripts.evaluate import app as evaluate_app
from pctrans.scripts.multiseed import app as multiseed_app
from pctrans.scripts.precompute import app as precompute_app
from pctrans.scripts.preprocess import app as preprocess_app
from pctrans.scripts.query import app as query_app
from pctrans.scripts.train import app as train_app
from pctrans.scripts.visualize import app as visualize_app

runner = CliRunner()


@pytest.mark.parametrize(
    "app",
    [
        download_app,
        preprocess_app,
        train_app,
        evaluate_app,
        multiseed_app,
        visualize_app,
        precompute_app,
        query_app,
    ],
)
def test_cli_help_exits_zero(app):
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
