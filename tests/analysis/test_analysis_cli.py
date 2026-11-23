"""Test analysis command parsing, dispatch and failure reporting."""

import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from steam_review_analysis import cli
from steam_review_analysis.analysis import cli as analysis_cli

pytestmark = pytest.mark.usefixtures("no_nltk_download")


def test_analyse_report_arguments_parse() -> None:
    """Parse the report subcommand with its game, cutoff and limit options."""
    # Act
    args = cli.build_parser().parse_args(
        ["analyse", "report", "--game", "Alpha", "--cutoff", "2024-01-01", "--top", "5"]
    )

    # Assert
    assert args.command == "analyse"
    assert args.analysis_command == "report"
    assert args.game == "Alpha"
    assert args.cutoff == "2024-01-01"
    assert args.top == 5
    assert args.min_word_freq == 2


PRESETS_TOML = """
[presets.Shooter]
app_id = 730
cutoff = "2023-09-29"
preserve_keywords = ["cs", "vac"]

[presets.Shooter.replacements]
cheater = "cheat"
"""


@pytest.fixture
def presets_file(tmp_path: Path) -> Path:
    """Write a one-preset TOML file.

    Returns:
        Path of the presets file.
    """
    path = tmp_path / "presets.toml"
    path.write_text(PRESETS_TOML, encoding="utf-8")
    return path


def test_analyse_preset_supplies_default_cutoff(presets_file: Path) -> None:
    """Use the preset boundary when no cutoff is supplied."""
    # Arrange
    args = cli.build_parser().parse_args(
        [
            "analyse",
            "report",
            "--game",
            "730",
            "--preset",
            "Shooter",
            "--presets",
            str(presets_file),
        ]
    )

    # Act & Assert
    assert analysis_cli._cutoff(args) == "2023-09-29"


def test_analyse_unknown_preset_fails(
    presets_file: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Report a preset missing from the presets file as a workflow failure."""
    # Arrange
    workflow = Mock()
    monkeypatch.setattr(analysis_cli, "run_analysis", workflow)
    argv = ["analyse", "report", "--game", "1", "--preset", "Nope", "--presets", str(presets_file)]

    # Act & Assert
    assert cli.main(argv) == 1
    assert "available: Shooter" in caplog.text
    workflow.assert_not_called()


def test_analyse_preset_without_presets_file_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail cleanly when a preset is requested but no presets file exists."""
    # Arrange
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(analysis_cli, "run_analysis", Mock())

    # Act & Assert
    assert cli.main(["analyse", "report", "--game", "1", "--preset", "Shooter"]) == 1


def test_analyse_uses_data_dir_for_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve processed data and default artifact paths from `--data-dir`."""
    # Arrange
    loader = Mock(side_effect=ValueError("stop after resolving paths"))
    monkeypatch.setattr(analysis_cli, "load_reviews", loader)

    argv = ["--data-dir", str(tmp_path), "analyse", "sentiment", "--game", "1"]

    # Act & Assert
    assert cli.main(argv) == 1
    assert loader.call_args.args == ("1", tmp_path / "processed")


def test_analyse_requires_cutoff_without_preset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail with a clear exit code when neither cutoff nor preset is given."""
    # Arrange
    workflow = Mock()
    monkeypatch.setattr(analysis_cli, "run_analysis", workflow)

    # Act & Assert
    assert cli.main(["analyse", "report", "--game", "Alpha"]) == 1
    workflow.assert_not_called()


@pytest.mark.parametrize("command", ["frequencies", "wordcloud"])
def test_period_commands_validate_cutoff_before_loading(
    command: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a missing cutoff before reading or preprocessing reviews."""
    # Arrange
    loader = Mock()
    monkeypatch.setattr(analysis_cli, "load_reviews", loader)

    # Act & Assert
    assert cli.main(["analyse", command, "--game", "Alpha"]) == 1
    loader.assert_not_called()


def test_analyse_rejects_nonpositive_limits() -> None:
    """Reject zero or negative word-frequency limits as usage errors."""
    # Act
    with pytest.raises(SystemExit) as caught:
        cli.main(["analyse", "report", "--game", "Alpha", "--cutoff", "2024-01-01", "--top", "0"])

    # Assert
    assert caught.value.code == 2


def test_analyse_report_dispatches_to_workflow(
    monkeypatch: pytest.MonkeyPatch, presets_file: Path
) -> None:
    """Forward command-line options to the public report workflow."""
    # Arrange
    report = SimpleNamespace(
        game="Alpha",
        output_dir=Path("out"),
        prepared=SimpleNamespace(reviews=[1, 2]),
        tables={"summary": Path("summary.json")},
        figures={},
    )
    workflow = Mock(return_value=report)
    monkeypatch.setattr(analysis_cli, "run_analysis", workflow)

    # Act & Assert
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
                "Shooter",
                "--presets",
                str(presets_file),
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
    assert workflow.call_args.kwargs["config"].replacements == {"cheater": "cheat"}


def test_analyse_report_failure_returns_failure_code(monkeypatch: pytest.MonkeyPatch) -> None:
    """Translate an expected workflow error into a failure exit code."""
    # Arrange
    monkeypatch.setattr(
        analysis_cli,
        "run_analysis",
        Mock(side_effect=ValueError("missing game")),
    )

    # Act & Assert
    assert cli.main(["analyse", "report", "--game", "Missing", "--cutoff", "2024-01-01"]) == 1


def test_analyse_games_lists_processed_games(
    monkeypatch: pytest.MonkeyPatch,
    sample_frame: pd.DataFrame,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Log each processed game with its application ID and review count."""
    # Arrange
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(analysis_cli, "read_processed", Mock(return_value=sample_frame))

    # Act & Assert
    assert cli.main(["analyse", "games"]) == 0
    assert "Alpha (app 1): 5 reviews" in caplog.text
    assert "Beta (app 2): 2 reviews" in caplog.text


def test_analyse_games_fails_without_processed_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """Report failure when the processed dataset contains no reviews."""
    # Arrange
    monkeypatch.setattr(analysis_cli, "read_processed", Mock(return_value=pd.DataFrame()))

    # Act & Assert
    assert cli.main(["analyse", "games"]) == 1
