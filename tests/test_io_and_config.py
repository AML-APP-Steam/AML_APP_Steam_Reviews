"""Test raw CSV serialization and loading the configured game registry."""

from datetime import date
from pathlib import Path

import pytest

from steam_review_analysis import config
from steam_review_analysis.config import Game, load_games
from steam_review_analysis.io import raw_path, read_raw_reviews, write_raw_reviews
from steam_review_analysis.schema import RAW_COLUMNS, Review


def test_raw_roundtrip(tmp_path: Path) -> None:
    """Preserve raw column order, quoted review text, and ISO dates through CSV I/O."""
    # Arrange
    review = Review(1, date(2024, 5, 21), "Text, with a comma", "12.5", "Recommended", "3 people")

    # Act
    write_raw_reviews(raw_path(1, tmp_path), [review])

    df = read_raw_reviews(tmp_path)

    # Assert
    assert list(df.columns) == list(RAW_COLUMNS)
    assert df.loc[0, "Review"] == "Text, with a comma"
    assert df.loc[0, "Date"] == "2024-05-21"


def test_read_raw_reviews_empty_dir(tmp_path: Path) -> None:
    """Return an empty frame when a directory contains no raw review files."""
    assert read_raw_reviews(tmp_path).empty


def test_load_games(tmp_path: Path) -> None:
    """Read a TOML game entry into a registry keyed by its application ID."""
    # Arrange
    path = tmp_path / "games.toml"
    path.write_text('[[games]]\napp_id = 7\nname = "Seven"\n', encoding="utf-8")

    # Act & Assert
    assert load_games(path) == {7: Game(7, "Seven")}


def test_repository_games_file_is_valid() -> None:
    """Ensure the repository game registry loads and includes the Terraria sample ID."""
    # Arrange
    repository_root = Path(__file__).resolve().parents[1]

    # Act & Assert
    assert 105600 in load_games(repository_root / config.DEFAULT_GAMES_FILE)


def test_data_dir_prefers_argument_then_environment_then_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resolve data locations at call time, never relative to the installed package."""
    # Arrange
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(config.DATA_DIR_ENV, raising=False)

    # Act & Assert
    assert config.data_dir() == tmp_path / "data"
    assert config.raw_dir() == tmp_path / "data" / "raw"

    # Arrange
    monkeypatch.setenv(config.DATA_DIR_ENV, str(tmp_path / "env"))

    # Act & Assert
    assert config.processed_dir() == tmp_path / "env" / "processed"
    assert config.analysis_dir(tmp_path / "arg") == tmp_path / "arg" / "processed" / "analysis"


def test_io_defaults_follow_configured_data_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Write and read raw data under the configured directory when no path is passed."""
    # Arrange
    monkeypatch.setenv(config.DATA_DIR_ENV, str(tmp_path))
    review = Review(7, date(2024, 5, 21), "Fine", "1.0", "Recommended", "0")

    # Act
    write_raw_reviews(raw_path(7), [review])

    # Assert
    assert (tmp_path / "raw" / "7_reviews.csv").is_file()
    assert read_raw_reviews()["Game"].tolist() == [7]
