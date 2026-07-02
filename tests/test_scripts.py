"""Smoke tests for the Typer CLI entry points."""

from typer.testing import CliRunner

from pctrans.scripts.evaluate import app as evaluate_app
from pctrans.scripts.visualize import app as visualize_app

runner = CliRunner()


def test_evaluate_help_exits_zero():
    result = runner.invoke(evaluate_app, ["--help"])
    assert result.exit_code == 0


def test_visualize_help_exits_zero():
    result = runner.invoke(visualize_app, ["--help"])
    assert result.exit_code == 0
