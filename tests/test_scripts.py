"""Smoke tests for the Typer CLI entry points."""

from typer.testing import CliRunner

from pctrans.scripts.evaluate import app as evaluate_app

runner = CliRunner()


def test_evaluate_help_exits_zero():
    result = runner.invoke(evaluate_app, ["--help"])
    assert result.exit_code == 0
