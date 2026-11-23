"""Test CLI argument handling, workflow dispatch, and reported failure exit codes."""

import subprocess
import sys
from datetime import date
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

import steam_review_analysis
from steam_review_analysis import cli
from steam_review_analysis.io import read_processed, write_raw_reviews
from steam_review_analysis.schema import RAW_COLUMNS, Review
from steam_review_analysis.scraping import service


def test_scrape_defaults() -> None:
    """Parse a game ID and headless mode while retaining the default review target."""
    # Act
    args = cli.build_parser().parse_args(["scrape", "105600", "--headless"])

    # Assert
    assert args.app_ids == [105600]
    assert args.headless is True
    assert args.target == 600


def test_command_is_required() -> None:
    """Reject invocation without a scrape or clean subcommand."""
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([])


def test_clean_fails_without_raw_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """Report failure when there are no raw reviews to clean."""
    # Arrange
    monkeypatch.setattr(cli, "read_raw_reviews", lambda _raw_dir: pd.DataFrame())

    # Act & Assert
    assert cli.main(["clean"]) == 1


@pytest.mark.parametrize("text,recommendation", [("Valid text", "Unknown"), ("  ", "Recommended")])
def test_clean_failure_does_not_publish_output(
    text: str,
    recommendation: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Avoid publishing output when cleaning rejects labels or removes every review."""
    # Arrange
    raw = pd.DataFrame(
        [
            [42, "2024-05-21", text, "12.5", recommendation, "0"],
        ],
        columns=list(RAW_COLUMNS),
    )
    writer = Mock()
    monkeypatch.setattr(cli, "read_raw_reviews", Mock(return_value=raw))
    monkeypatch.setattr(cli, "load_games", Mock(return_value={}))
    monkeypatch.setattr(cli, "write_processed", writer)

    # Act & Assert
    assert cli.main(["clean"]) == 1
    writer.assert_not_called()
    assert caplog.records


def test_clean_io_failure_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    """Convert an input-file read error into the cleaning failure exit code."""
    # Arrange
    monkeypatch.setattr(cli, "read_raw_reviews", Mock(side_effect=OSError("cannot read raw CSV")))

    # Act & Assert
    assert cli.main(["clean"]) == 1


@pytest.mark.parametrize(
    "option,value",
    [
        ("--target", "0"),
        ("--target", "-1"),
        ("--scroll-pause", "-1"),
        ("--scroll-pause", "nan"),
        ("--wait-timeout", "0"),
        ("--wait-timeout", "inf"),
        ("--page-load-timeout", "0"),
        ("--max-stalled-scrolls", "0"),
        ("--max-stale-retries", "0"),
        ("--language", ""),
        ("--filter", ""),
    ],
)
def test_invalid_parameters_are_usage_errors(
    option: str,
    value: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject invalid scraping options before invoking the workflow, using exit code 2."""
    # Arrange
    workflow = Mock()
    monkeypatch.setattr(service, "scrape_games", workflow)

    # Act
    with pytest.raises(SystemExit) as caught:
        cli.main(["scrape", "42", option, value])

    # Assert
    assert caught.value.code == 2
    workflow.assert_not_called()


def test_cli_passes_explicit_settings_to_public_workflow(monkeypatch: pytest.MonkeyPatch) -> None:
    """Forward user-supplied limits and flags to the public scraping workflow unchanged."""
    # Arrange
    workflow = Mock(return_value={})
    monkeypatch.setattr(service, "scrape_games", workflow)

    # Act & Assert
    assert (
        cli.main(
            [
                "scrape",
                "42",
                "--target",
                "12",
                "--headless",
                "--overwrite",
                "--scroll-pause",
                "0.5",
                "--wait-timeout",
                "7",
                "--page-load-timeout",
                "15",
                "--max-stalled-scrolls",
                "4",
                "--max-stale-retries",
                "2",
            ]
        )
        == 0
    )
    assert workflow.call_args.args == ([42],)
    assert workflow.call_args.kwargs["headless"] is True
    assert workflow.call_args.kwargs["overwrite"] is True
    settings = workflow.call_args.kwargs["settings"]
    assert settings.target_reviews == 12
    assert settings.scroll_pause_s == 0.5
    assert settings.wait_timeout_s == 7
    assert settings.page_load_timeout_s == 15
    assert settings.max_stalled_scrolls == 4
    assert settings.max_stale_retries == 2


def test_scrape_without_ids_or_registry_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Ask for app IDs or --games when no registry exists in the working directory."""
    # Arrange
    monkeypatch.chdir(tmp_path)
    workflow = Mock()
    monkeypatch.setattr(service, "scrape_games", workflow)

    # Act & Assert
    assert cli.main(["scrape"]) == 1
    assert "pass app IDs or --games" in caplog.text
    workflow.assert_not_called()


def test_scrape_reads_games_option_and_data_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Scrape every game in an explicit registry into `<data-dir>/raw`."""
    # Arrange
    games = tmp_path / "games.toml"
    games.write_text('[[games]]\napp_id = 7\nname = "Seven"\n', encoding="utf-8")
    workflow = Mock(return_value={})
    monkeypatch.setattr(service, "scrape_games", workflow)

    # Act & Assert
    assert cli.main(["--data-dir", str(tmp_path), "scrape", "--games", str(games)]) == 0
    assert workflow.call_args.args == ([7],)
    assert workflow.call_args.kwargs["raw_dir"] == tmp_path / "raw"


def test_clean_end_to_end_with_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Clean raw CSVs from `--data-dir` and name games from the --games registry."""
    # Arrange
    monkeypatch.chdir(tmp_path)
    review = Review(7, date(2024, 5, 21), "Fine", "1.0", "Recommended", "0")
    write_raw_reviews(tmp_path / "store" / "raw" / "7_reviews.csv", [review])
    games = tmp_path / "games.toml"
    games.write_text('[[games]]\napp_id = 7\nname = "Seven"\n', encoding="utf-8")

    # Act & Assert
    assert cli.main(["--data-dir", str(tmp_path / "store"), "clean", "--games", str(games)]) == 0
    cleaned = read_processed(tmp_path / "store" / "processed")
    assert cleaned["game"].tolist() == ["Seven"]
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize("argv", [["analyse", "games"], ["download-nltk"]])
def test_analysis_commands_explain_missing_extra(
    argv: list[str], monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Point to the `analysis` extra instead of crashing when it is not installed."""
    # Arrange
    monkeypatch.setattr(cli, "_load_analysis_cli", lambda: None)

    # Act & Assert
    assert cli.main(argv) == 1
    assert "steam-review-analysis[analysis]" in caplog.text


@pytest.mark.parametrize(
    "argv",
    [
        ["--data-dir", "/d", "--log-level", "DEBUG", "clean"],
        ["clean", "--data-dir", "/d", "--log-level", "DEBUG"],
        ["--log-level", "DEBUG", "clean", "--data-dir", "/d"],
        ["analyse", "games", "--data-dir", "/d", "--log-level", "DEBUG"],
        ["analyse", "--data-dir", "/d", "--log-level", "DEBUG", "games"],
    ],
)
def test_global_options_work_before_or_after_the_command(argv: list[str]) -> None:
    """Accept `--data-dir` and `--log-level` on either side of the command name."""
    # Act
    args = cli.build_parser().parse_args(argv)

    # Assert
    assert args.data_dir == Path("/d")
    assert args.log_level == "DEBUG"


def test_global_option_defaults_survive_subcommand_parsing() -> None:
    """Keep top-level defaults when a subcommand does not repeat the global options."""
    # Act
    args = cli.build_parser().parse_args(["scrape", "42"])

    # Assert
    assert args.data_dir is None
    assert args.log_level == "INFO"


def test_version_flag_prints_package_version(capsys: pytest.CaptureFixture[str]) -> None:
    """Print the installed package version and exit successfully."""
    # Act
    with pytest.raises(SystemExit) as caught:
        cli.main(["--version"])

    # Assert
    assert caught.value.code == 0
    assert (
        capsys.readouterr().out.strip()
        == f"steam-review-analysis {steam_review_analysis.__version__}"
    )


def test_ctrl_c_exits_with_interrupt_status(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Report Ctrl+C in one line with exit status 130 instead of a traceback."""
    # Arrange
    monkeypatch.setattr(cli, "read_raw_reviews", Mock(side_effect=KeyboardInterrupt))

    # Act & Assert
    assert cli.main(["clean"]) == cli.EXIT_INTERRUPTED
    assert "Interrupted by user" in caplog.text


def test_building_the_parser_skips_heavy_imports() -> None:
    """Keep `--help` fast: building the parser must not import the optional heavy libraries."""
    # Arrange
    code = (
        "import sys\n"
        "from steam_review_analysis import cli\n"
        "cli.build_parser()\n"
        "heavy = {'nltk', 'scipy', 'matplotlib', 'sklearn', 'wordcloud', 'selenium'}\n"
        "print(sorted(heavy & set(sys.modules)))\n"
    )

    # Act
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )

    # Assert
    assert result.stdout.strip() == "[]"


def test_analyse_help_lists_subcommands_without_analysis_extra(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Show real `analyse` help even when the analysis handlers cannot be imported."""
    # Arrange
    monkeypatch.setattr(cli, "_load_analysis_cli", lambda: None)

    # Act
    with pytest.raises(SystemExit):
        cli.main(["analyse", "--help"])

    # Assert
    assert "transformer-sentiment" in capsys.readouterr().out
