import pandas as pd
import pytest

from steam_reviews import cli


def test_scrape_defaults() -> None:
    args = cli.build_parser().parse_args(["scrape", "105600", "--headless"])

    assert args.app_ids == [105600]
    assert args.headless is True
    assert args.target == 600


def test_command_is_required() -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([])


def test_clean_fails_without_raw_data(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "read_raw_reviews", lambda: pd.DataFrame())

    assert cli.main(["clean"]) == 1
