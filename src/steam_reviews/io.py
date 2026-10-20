"""Reading and writing the raw and processed review CSVs."""

import csv
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from steam_reviews.config import PROCESSED_DIR, RAW_DIR
from steam_reviews.schema import RAW_COLUMNS, Review

PROCESSED_FILE = "reviews.csv"


def raw_path(app_id: int, raw_dir: Path = RAW_DIR) -> Path:
    """Build the raw CSV path for one game.

    Args:
        app_id: Steam app id of the game.
        raw_dir: Directory holding the raw CSVs.

    Returns:
        Path such as `data/raw/105600_reviews.csv`.
    """
    return raw_dir / f"{app_id}_reviews.csv"


def write_raw_reviews(path: Path, reviews: Iterable[Review]) -> None:
    """Write reviews to a raw CSV, creating parent directories.

    Args:
        path: Destination file; overwritten if it exists.
        reviews: Reviews to write, one row each, after a header row.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(RAW_COLUMNS)
        writer.writerows(review.as_row() for review in reviews)


def read_raw_reviews(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load and concatenate every raw review CSV.

    Args:
        raw_dir: Directory searched for `*_reviews.csv` files.

    Returns:
        One frame with the raw column names; empty if no files exist.
    """
    paths = sorted(raw_dir.glob("*_reviews.csv"))
    if not paths:
        return pd.DataFrame(columns=list(RAW_COLUMNS))
    return pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)


def write_processed(df: pd.DataFrame, processed_dir: Path = PROCESSED_DIR) -> Path:
    """Save the cleaned reviews as `reviews.csv`.

    Args:
        df: Cleaned reviews, as returned by `steam_reviews.processing.cleaning.clean_reviews`.
        processed_dir: Output directory, created if missing.

    Returns:
        Path of the written file.
    """
    path = processed_dir / PROCESSED_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def read_processed(processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Load the cleaned reviews written by `write_processed`.

    Args:
        processed_dir: Directory containing `reviews.csv`.

    Returns:
        Cleaned reviews with the `date` column parsed as datetimes.
    """
    return pd.read_csv(processed_dir / PROCESSED_FILE, parse_dates=["date"])
