"""Test CLI argument handling, workflow dispatch, and reported failure exit codes."""

from unittest.mock import Mock

import pandas as pd
import pytest

from steam_reviews import cli
from steam_reviews.schema import RAW_COLUMNS


def test_scrape_defaults() -> None:
    """Parse a game ID and headless mode while retaining the default review target."""
    args = cli.build_parser().parse_args(["scrape", "105600", "--headless"])

    assert args.app_ids == [105600]
    assert args.headless is True
    assert args.target == 600


def test_command_is_required() -> None:
    """Reject invocation without a scrape or clean subcommand."""
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([])


def test_clean_fails_without_raw_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """Report failure when there are no raw reviews to clean."""
    monkeypatch.setattr(cli, "read_raw_reviews", lambda: pd.DataFrame())

    assert cli.main(["clean"]) == 1


@pytest.mark.parametrize("text,recommendation", [("Valid text", "Unknown"), ("  ", "Recommended")])
def test_clean_failure_does_not_publish_output(
    text: str,
    recommendation: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Avoid publishing output when cleaning rejects labels or removes every review."""
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
    assert cli.main(["clean"]) == 1
    writer.assert_not_called()
    assert caplog.records


def test_clean_io_failure_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    """Convert an input-file read error into the cleaning failure exit code."""
    monkeypatch.setattr(cli, "read_raw_reviews", Mock(side_effect=OSError("cannot read raw CSV")))
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
    workflow = Mock()
    monkeypatch.setattr(cli, "scrape_games", workflow)
    with pytest.raises(SystemExit) as caught:
        cli.main(["scrape", "42", option, value])
    assert caught.value.code == 2
    workflow.assert_not_called()


def test_cli_passes_explicit_settings_to_public_workflow(monkeypatch: pytest.MonkeyPatch) -> None:
    """Forward user-supplied limits and flags to the public scraping workflow unchanged."""
    workflow = Mock(return_value={})
    monkeypatch.setattr(cli, "scrape_games", workflow)
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
