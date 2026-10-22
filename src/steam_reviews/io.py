"""Reading and writing the raw and processed review CSVs."""

import csv
from collections.abc import Iterable
from pathlib import Path
from tempfile import NamedTemporaryFile

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


def write_raw_reviews(path: Path, reviews: Iterable[Review], *, overwrite: bool = False) -> None:
    """Publish a complete, nonempty raw CSV without truncating an existing file.

    Args:
        path: Destination file; parent directories are created when needed.
        reviews: Nonempty iterable of records, written after the shared CSV header.
        overwrite: Explicitly allow atomic replacement of an existing destination.

    Raises:
        ValueError: The iterable is empty; no file is created or changed.
        FileExistsError: The destination exists and overwrite is False.
        OSError: Writing or publishing failed. Existing data remains intact.
    """
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; enable overwrite to replace it")
    iterator = iter(reviews)
    try:
        first = next(iterator)
    except StopIteration:
        raise ValueError("Refusing to write an empty raw review dataset") from None
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary = Path(file.name)
            writer = csv.writer(file)
            writer.writerow(RAW_COLUMNS)
            writer.writerow(first.as_row())
            writer.writerows(review.as_row() for review in iterator)
        if overwrite:
            temporary.replace(path)
        else:
            path.hardlink_to(temporary)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_raw_reviews(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load and concatenate every raw review CSV.

    Args:
        raw_dir: Directory searched for `*_reviews.csv` files.

    Returns:
        One frame with the raw column names; empty if no files exist. Review text stays
        literal, including strings such as `NA`, `NULL`, and leading-zero numbers.
    """
    paths = sorted(raw_dir.glob("*_reviews.csv"))
    if not paths:
        return pd.DataFrame(columns=list(RAW_COLUMNS))
    return pd.concat(
        (pd.read_csv(path, dtype={"Review": str}, keep_default_na=False) for path in paths),
        ignore_index=True,
    )


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
        Cleaned reviews with parsed dates and literal review text. Empty optional hours
        and game names are read as missing values.
    """
    return pd.read_csv(
        processed_dir / PROCESSED_FILE,
        parse_dates=["date"],
        dtype={"text": str},
        keep_default_na=False,
        na_values={"hours_played": [""], "game": [""]},
    )
