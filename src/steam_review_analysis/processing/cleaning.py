"""Normalize raw review records without changing their sentiment-bearing text."""

import logging
import re
from collections.abc import Mapping

import pandas as pd

from steam_review_analysis.config import Game
from steam_review_analysis.schema import RAW_TO_CLEAN_COLUMNS

_logger = logging.getLogger(__name__)

_FIRST_NUMBER = re.compile(r"-?\d[\d,]*")
_ISO_DATE_PREFIX = r"^\d{4}-\d{2}-\d{2}(?:$|[ T])"
_NO_VOTES = 0
_RECOMMENDATIONS = {
    "recommended": "Recommended",
    "not recommended": "Not Recommended",
}


def extract_first_int(text: object) -> int:
    """Extract the first nonnegative vote count from a raw label.

    Args:
        text: Label such as `1,234 people found this review helpful`, a number, or a
            missing value. Commas within the matched count are thousands separators.

    Returns:
        First integer count, or zero if no number is present or the first count is negative.
        Missing labels and `No helpful votes` retain the existing zero-vote convention.
    """
    match = _FIRST_NUMBER.search(str(text))
    if match is None:
        return _NO_VOTES
    return max(_NO_VOTES, int(match.group(0).replace(",", "")))


def _select_raw_columns(raw: pd.DataFrame) -> pd.DataFrame:
    """Validate the input schema and copy only the shared raw review columns.

    Args:
        raw: Caller-owned data frame, potentially with additional metadata columns.

    Returns:
        Independent frame with the canonical snake_case column names and order.

    Raises:
        ValueError: Required columns are missing or column names are duplicated.
    """
    if not raw.columns.is_unique:
        raise ValueError("Raw review column names must be unique")
    missing = [name for name in RAW_TO_CLEAN_COLUMNS if name not in raw.columns]
    if missing:
        raise ValueError(f"Missing raw review columns: {', '.join(missing)}")
    return raw.loc[:, list(RAW_TO_CLEAN_COLUMNS)].rename(columns=RAW_TO_CLEAN_COLUMNS).copy()


def _parse_app_ids(values: pd.Series) -> pd.Series:
    """Convert IDs without silently truncating fractional numbers.

    Args:
        values: Raw application IDs from rows with nonempty review text.

    Returns:
        Series of positive, signed 64-bit integer IDs with the original index.

    Raises:
        ValueError: IDs are missing, nonnumeric, boolean, fractional, nonpositive,
            or outside the signed 64-bit integer range.
    """
    message = "Game/app_id must contain positive whole numbers within the int64 range"
    if values.map(lambda value: isinstance(value, bool)).any():
        raise ValueError(message)
    try:
        parsed = pd.to_numeric(values, errors="coerce").astype("Int64")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if parsed.isna().any() or parsed.le(0).any():
        raise ValueError(message)
    return parsed.astype("int64")


def _parse_dates(values: pd.Series) -> pd.Series:
    """Parse current ISO dates and timestamps from older scraper exports.

    Args:
        values: ISO date labels, ISO timestamps without a timezone, or datetime values.

    Returns:
        Datetime series with nanosecond resolution and the original index.

    Raises:
        ValueError: A date is missing, invalid, timezone-aware, or out of range.
    """
    message = "Date must contain valid ISO dates or timestamps without a timezone"
    text = values.astype("string").str.strip()
    if not text.str.match(_ISO_DATE_PREFIX, na=False).all():
        raise ValueError(message)
    try:
        parsed = pd.to_datetime(text, format="ISO8601", errors="coerce").astype("datetime64[ns]")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if parsed.isna().any():
        raise ValueError(message)
    return parsed


def _parse_hours(values: pd.Series) -> pd.Series:
    """Normalize optional playtime without inventing missing or negative hours.

    Args:
        values: Numeric values or English number strings with optional thousands commas.

    Returns:
        Float64 series; missing, unparseable, negative, and infinite values become NaN.
    """
    text = values.astype("string").str.replace(",", "", regex=False).str.strip()
    hours = pd.to_numeric(text, errors="coerce").astype("float64")
    return hours.where(hours.ge(0) & hours.lt(float("inf")))


def _normalize_recommendations(values: pd.Series) -> pd.Series:
    """Validate recommendation labels before deriving a binary target.

    Args:
        values: Raw recommendation labels, with optional outer whitespace or case differences.

    Returns:
        Series containing only `Recommended` and `Not Recommended`.

    Raises:
        ValueError: Any label is missing or unknown; unknown labels are never mapped to False.
    """
    labels = values.fillna("").astype(str).str.strip().str.casefold().map(_RECOMMENDATIONS)
    invalid_count = int(labels.isna().sum())
    if invalid_count:
        raise ValueError(
            f"Recommendation has {invalid_count} missing or unknown labels; "
            "expected Recommended or Not Recommended"
        )
    return labels


def clean_reviews(raw: pd.DataFrame, games: Mapping[int, Game] | None = None) -> pd.DataFrame:
    """Prepare a typed, deduplicated dataset through one public function.

    Args:
        raw: Frame containing every raw column from `schema.RAW_COLUMNS`. Extra columns
            are ignored. The input is never modified.
        games: Optional mapping from app IDs to game records. Passing any mapping,
            including an empty one, adds a `game` column; unknown names remain missing.

    Returns:
        New frame with `app_id` (int64), `date` (datetime64[ns]), `text` (stripped text),
        `hours_played` (float64, NaN if unknown), `recommendation` (canonical label),
        `helpful_votes` (integer count), and `is_recommended` (bool). If requested, `game`
        follows `app_id`. Empty-text rows and exact normalized duplicates are removed;
        text case, punctuation, and internal whitespace are preserved. Removal counts
        are logged. A valid empty input returns an empty frame with the same schema.

    Raises:
        ValueError: The input schema is invalid, or a retained row has an invalid ID,
            date, or recommendation. Bad essential fields are not silently repaired.
    """
    df = _select_raw_columns(raw)
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    empty_text = df["text"].eq("")
    empty_count = int(empty_text.sum())
    df = df.loc[~empty_text].copy()

    df["app_id"] = _parse_app_ids(df["app_id"])
    df["date"] = _parse_dates(df["date"])
    df["hours_played"] = _parse_hours(df["hours_played"])
    df["recommendation"] = _normalize_recommendations(df["recommendation"])
    df["helpful_votes"] = df["helpful_votes"].map(extract_first_int).astype("int64")
    df["is_recommended"] = df["recommendation"].eq("Recommended").astype(bool)

    duplicate_count = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    if games is not None:
        names = df["app_id"].map({app_id: game.name for app_id, game in games.items()})
        df.insert(1, "game", names.astype("string"))
        unknown_count = int(names.isna().sum())
        if unknown_count:
            _logger.warning("Game names are missing for %d reviews", unknown_count)
    _logger.info(
        "Cleaned reviews: input=%d, kept=%d, empty_text=%d, duplicates=%d, unknown_hours=%d",
        len(raw),
        len(df),
        empty_count,
        duplicate_count,
        int(df["hours_played"].isna().sum()),
    )
    return df
