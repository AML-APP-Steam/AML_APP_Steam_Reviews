"""Test analysis command parsing, dispatch and failure reporting."""

import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from steam_reviews import cli
from steam_reviews.analysis import cli as analysis_cli


def test_analyse_report_arguments_parse() -> None:
    """Parse the report subcommand with its game, cutoff and limit options."""
    args = cli.build_parser().parse_args(
        ["analyse", "report", "--game", "Alpha", "--cutoff", "2024-01-01", "--top", "5"]
    )

    assert args.command == "analyse"
    assert args.analysis_command == "report"
    assert args.game == "Alpha"
    assert args.cutoff == "2024-01-01"
    assert args.top == 5
    assert args.min_word_freq == 2


def test_analyse_preset_supplies_default_cutoff() -> None:
    """Use the preset boundary when no cutoff is supplied."""
    args = cli.build_parser().parse_args(
        ["analyse", "report", "--game", "730", "--preset", "Counter_Strike"]
    )

    assert analysis_cli._cutoff(args) == "2023-09-29"


def test_analyse_requires_cutoff_without_preset() -> None:
    """Fail with a clear exit code when neither cutoff nor preset is given."""
    assert cli.main(["analyse", "report", "--game", "Alpha"]) == 1


def test_analyse_rejects_nonpositive_limits() -> None:
    """Reject zero or negative word-frequency limits as usage errors."""
    with pytest.raises(SystemExit) as caught:
        cli.main(["analyse", "report", "--game", "Alpha", "--cutoff", "2024-01-01", "--top", "0"])
    assert caught.value.code == 2


def test_analyse_report_dispatches_to_workflow(monkeypatch: pytest.MonkeyPatch) -> None:
    """Forward command-line options to the public report workflow."""
    report = SimpleNamespace(
        game="Alpha",
        output_dir=Path("out"),
        prepared=SimpleNamespace(reviews=[1, 2]),
        tables={"summary": Path("summary.json")},
        figures={},
    )
    workflow = Mock(return_value=report)
    monkeypatch.setattr(analysis_cli, "run_analysis", workflow)

    assert (
        cli.main(
            [
                "analyse",
                "report",
                "--game",
                "Alpha",
                "--cutoff",
                "2024-01-01",
                "--preset",
                "PUBG",
                "--top",
                "7",
                "--min-word-freq",
                "3",
            ]
        )
        == 0
    )
    assert workflow.call_args.args == ("Alpha", "2024-01-01")
    assert workflow.call_args.kwargs["top_n"] == 7
    assert workflow.call_args.kwargs["min_word_freq"] == 3
    assert workflow.call_args.kwargs["config"] is not None


def test_analyse_report_failure_returns_failure_code(monkeypatch: pytest.MonkeyPatch) -> None:
    """Translate an expected workflow error into a failure exit code."""
    monkeypatch.setattr(
        analysis_cli,
        "run_analysis",
        Mock(side_effect=ValueError("missing game")),
    )

    assert cli.main(["analyse", "report", "--game", "Missing", "--cutoff", "2024-01-01"]) == 1


def test_analyse_games_lists_processed_games(
    monkeypatch: pytest.MonkeyPatch,
    sample_frame: pd.DataFrame,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Log each processed game with its application ID and review count."""
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(analysis_cli, "read_processed", Mock(return_value=sample_frame))

    assert cli.main(["analyse", "games"]) == 0
    assert "Alpha (app 1): 5 reviews" in caplog.text
    assert "Beta (app 2): 2 reviews" in caplog.text


def test_analyse_games_fails_without_processed_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """Report failure when the processed dataset contains no reviews."""
    monkeypatch.setattr(analysis_cli, "read_processed", Mock(return_value=pd.DataFrame()))

    assert cli.main(["analyse", "games"]) == 1
