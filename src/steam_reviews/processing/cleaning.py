"""Turn raw scraped reviews into typed, deduplicated, analysis-ready data."""

import re
from collections.abc import Mapping

import pandas as pd

from steam_reviews.config import Game
from steam_reviews.schema import RAW_TO_CLEAN_COLUMNS

_FIRST_NUMBER = re.compile(r"[\d,]+")


def extract_first_int(text: object) -> int:
    """Extract the first integer from a text.

    Args:
        text: Text such as `"1,234 people found this review helpful"`; commas are allowed.

    Returns:
        The first integer found (1234 for the example above), or 0 if there is none.
    """
    match = _FIRST_NUMBER.search(str(text))
    return int(match.group(0).replace(",", "")) if match else 0


def clean_reviews(raw: pd.DataFrame, games: Mapping[int, Game] | None = None) -> pd.DataFrame:
    """Turn raw scraped reviews into a typed, deduplicated, analysis-ready frame.

    Args:
        raw: Frame with the raw CSV columns (see `steam_reviews.schema.RAW_COLUMNS`).
        games: If given, a `game` name column is added by looking up each app id.

    Returns:
        New frame (the input is not modified) with snake_case columns: `app_id`, `date`
        (datetime), `text` (stripped), `hours_played` (float, NaN if unknown), `helpful_votes`
        (int), `recommendation` and a boolean `is_recommended`. Rows with empty text and
        exact duplicates are dropped.
    """
    df = raw.rename(columns=RAW_TO_CLEAN_COLUMNS)
    df["app_id"] = df["app_id"].astype("int64")
    df["date"] = pd.to_datetime(df["date"])
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df["hours_played"] = pd.to_numeric(
        df["hours_played"].astype(str).str.replace(",", "", regex=False), errors="coerce"
    )
    df["helpful_votes"] = df["helpful_votes"].map(extract_first_int)
    df["is_recommended"] = df["recommendation"].eq("Recommended")

    df = df[df["text"].ne("")].drop_duplicates().reset_index(drop=True)

    if games:
        df.insert(1, "game", df["app_id"].map({app_id: g.name for app_id, g in games.items()}))
    return df
