"""Test raw CSV serialization and loading the configured game registry."""

from datetime import date
from pathlib import Path

from steam_reviews.config import GAMES_FILE, Game, load_games
from steam_reviews.io import raw_path, read_raw_reviews, write_raw_reviews
from steam_reviews.schema import RAW_COLUMNS, Review


def test_raw_roundtrip(tmp_path: Path) -> None:
    """Preserve raw column order, quoted review text, and ISO dates through CSV I/O."""
    review = Review(1, date(2024, 5, 21), "Text, with a comma", "12.5", "Recommended", "3 people")
    write_raw_reviews(raw_path(1, tmp_path), [review])

    df = read_raw_reviews(tmp_path)

    assert list(df.columns) == list(RAW_COLUMNS)
    assert df.loc[0, "Review"] == "Text, with a comma"
    assert df.loc[0, "Date"] == "2024-05-21"


def test_read_raw_reviews_empty_dir(tmp_path: Path) -> None:
    """Return an empty frame when a directory contains no raw review files."""
    assert read_raw_reviews(tmp_path).empty


def test_load_games(tmp_path: Path) -> None:
    """Read a TOML game entry into a registry keyed by its application ID."""
    path = tmp_path / "games.toml"
    path.write_text('[[games]]\napp_id = 7\nname = "Seven"\n', encoding="utf-8")
    assert load_games(path) == {7: Game(7, "Seven")}


def test_shipped_games_file_is_valid() -> None:
    """Ensure the repository game registry loads and includes the Terraria sample ID."""
    assert 105600 in load_games(GAMES_FILE)
