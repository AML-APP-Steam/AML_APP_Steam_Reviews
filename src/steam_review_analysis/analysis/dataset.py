"""Load processed reviews for one game and split them into analysis periods."""

import re
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from steam_review_analysis import config
from steam_review_analysis.io import read_processed

_NON_ALPHANUMERIC = re.compile(r"[^a-z0-9]+")


def _normalize_name(value: str) -> str:
    """Reduce a game name to lowercase alphanumeric words.

    Args:
        value: Display name or selector such as `Counter_Strike`.

    Returns:
        Lowercase words separated by single spaces.
    """
    return _NON_ALPHANUMERIC.sub(" ", value.casefold()).strip()


def _available_games(reviews: pd.DataFrame) -> str:
    """Describe the games present in a processed frame.

    Args:
        reviews: Processed reviews with `app_id` and optional `game` columns.

    Returns:
        Comma-separated `name (app_id)` labels for error messages.
    """
    if reviews.empty:
        return "none"
    if "game" not in reviews.columns:
        return ", ".join(str(value) for value in sorted(reviews["app_id"].unique()))
    grouped = reviews.groupby(["app_id", "game"], dropna=False).size().reset_index()
    return ", ".join(f"{row.game} ({row.app_id})" for row in grouped.itertuples())


def resolve_game(reviews: pd.DataFrame, game: int | str) -> int:
    """Resolve a selector to the Steam application ID of one game.

    Args:
        reviews: Processed reviews with `app_id` and optional `game` columns.
        game: Steam application ID, numeric string, or case-insensitive display name.

    Returns:
        Application ID represented in the frame.

    Raises:
        ValueError: The selector matches no game or the frame is empty.
    """
    if reviews.empty:
        raise ValueError("No processed reviews are available")
    if isinstance(game, str) and game.strip().isdigit():
        game = int(game)
    if isinstance(game, int):
        matches = reviews.loc[reviews["app_id"] == game, "app_id"].unique()
    else:
        selector = _normalize_name(game)
        if "game" not in reviews.columns:
            raise ValueError("Cannot select by name because processed data has no game column")
        names = reviews["game"].fillna("").astype(str).map(_normalize_name)
        matches = reviews.loc[names == selector, "app_id"].unique()
    if len(matches) == 0:
        raise ValueError(
            f"Game {game!r} is not in the processed data; available games: "
            f"{_available_games(reviews)}"
        )
    if len(matches) > 1:
        raise ValueError(f"Game selector {game!r} matches more than one application ID")
    return int(matches[0])


def load_reviews(game: int | str, processed_dir: Path | None = None) -> pd.DataFrame:
    """Load every processed review for one game.

    Args:
        game: Steam application ID, numeric string, or display name.
        processed_dir: Directory containing `reviews.csv`; defaults to
            `steam_review_analysis.config.processed_dir()`.

    Returns:
        Game reviews sorted by date with a reset index.

    Raises:
        ValueError: The game is not present or the processed file is empty.
        OSError: The processed CSV cannot be read.
    """
    reviews = read_processed(processed_dir)
    app_id = resolve_game(reviews, game)
    selected = reviews.loc[reviews["app_id"] == app_id].copy()
    return selected.sort_values("date").reset_index(drop=True)


def game_name(reviews: pd.DataFrame) -> str:
    """Return a display name for the reviews' game.

    Args:
        reviews: Nonempty processed review frame.

    Returns:
        Game name when available, otherwise the application ID as text.

    Raises:
        ValueError: The frame is empty.
    """
    if reviews.empty:
        raise ValueError("Cannot name an empty review frame")
    if "game" in reviews.columns and pd.notna(reviews.iloc[0]["game"]):
        return str(reviews.iloc[0]["game"])
    return str(reviews.iloc[0]["app_id"])


def slugify(value: str) -> str:
    """Convert a game name into a filesystem-safe directory name.

    Args:
        value: Display name such as `PUBG: BATTLEGROUNDS`.

    Returns:
        Lowercase alphanumeric name joined by underscores.
    """
    slug = _NON_ALPHANUMERIC.sub("_", value.casefold()).strip("_")
    return slug or "reviews"


def default_output_dir(name: str, root: Path | None = None) -> Path:
    """Build the default artifact directory for one game.

    Args:
        name: Game display name.
        root: Parent directory for analysis artifacts; defaults to
            `steam_review_analysis.config.analysis_dir()`.

    Returns:
        Path such as `data/processed/analysis/terraria`.
    """
    return (root or config.analysis_dir()) / slugify(name)


def parse_cutoff(cutoff: str | pd.Timestamp) -> pd.Timestamp:
    """Parse and validate a before/after boundary date.

    Args:
        cutoff: ISO date, datetime-like value, or timestamp.

    Returns:
        Timestamp used to split reviews.

    Raises:
        ValueError: The cutoff is missing or cannot be parsed.
    """
    try:
        boundary = pd.Timestamp(cutoff)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid cutoff date: {cutoff!r}") from exc
    if pd.isna(boundary):
        raise ValueError(f"Invalid cutoff date: {cutoff!r}")
    return boundary


def split_by_cutoff(
    reviews: pd.DataFrame,
    cutoff: str | pd.Timestamp,
    *,
    date_column: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split reviews into before and on-or-after groups.

    Args:
        reviews: Reviews with a datetime column.
        cutoff: Boundary date; rows on the boundary belong to the later group.
        date_column: Name of the datetime column.

    Returns:
        `(before, after)` frames with their original indices.

    Raises:
        KeyError: The date column does not exist.
        ValueError: The cutoff cannot be parsed.
    """
    if date_column not in reviews.columns:
        raise KeyError(f"Missing date column: {date_column}")
    boundary = parse_cutoff(cutoff)
    return reviews.loc[reviews[date_column] < boundary], reviews.loc[
        reviews[date_column] >= boundary
    ]


def add_year(reviews: pd.DataFrame, *, date_column: str = "date") -> pd.DataFrame:
    """Add the calendar year of each review date.

    Args:
        reviews: Reviews with a datetime column.
        date_column: Name of the datetime column.

    Returns:
        Copy of the frame with an integer `year` column inserted after the date.

    Raises:
        KeyError: The date column does not exist.
        ValueError: A date is missing.
    """
    if date_column not in reviews.columns:
        raise KeyError(f"Missing date column: {date_column}")
    if reviews[date_column].isna().any():
        raise ValueError("Cannot derive a year from missing review dates")
    result = reviews.copy()
    years = result[date_column].dt.year.astype("int64")
    position = list(result.columns).index(date_column) + 1
    result.insert(position, "year", years)
    return result


def add_period(
    reviews: pd.DataFrame,
    cutoff: str | pd.Timestamp,
    *,
    date_column: str = "date",
) -> pd.DataFrame:
    """Label each review as before or on-or-after a cutoff.

    Args:
        reviews: Reviews with a datetime column.
        cutoff: Boundary date; rows on the boundary are labelled `After`.
        date_column: Name of the datetime column.

    Returns:
        Copy of the frame with a `period` column containing `Before` or `After`.

    Raises:
        KeyError: The date column does not exist.
        ValueError: The cutoff cannot be parsed.
    """
    before, _ = split_by_cutoff(reviews, cutoff, date_column=date_column)
    result = reviews.copy()
    result["period"] = "After"
    result.loc[before.index, "period"] = "Before"
    return result


def period_frames(
    reviews: pd.DataFrame,
    cutoff: str | pd.Timestamp,
) -> Mapping[str, pd.DataFrame]:
    """Return named before/after frames for reporting code.

    Args:
        reviews: Reviews with a datetime `date` column.
        cutoff: Boundary date.

    Returns:
        Mapping with `before` and `after` keys.
    """
    before, after = split_by_cutoff(reviews, cutoff)
    return {"before": before, "after": after}
