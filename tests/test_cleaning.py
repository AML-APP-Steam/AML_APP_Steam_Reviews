"""Test data validation, normalization, and CSV roundtrips for review cleaning."""

import logging
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from steam_review_analysis.config import Game
from steam_review_analysis.io import (
    read_processed,
    read_raw_reviews,
    write_processed,
    write_raw_reviews,
)
from steam_review_analysis.processing.cleaning import clean_reviews, extract_first_int
from steam_review_analysis.schema import RAW_COLUMNS, Review


@pytest.fixture
def raw_reviews() -> pd.DataFrame:
    """Build dirty rows covering duplicates, missing hours, and empty review text.

    Returns:
        Raw review frame containing two distinct reviews, one duplicate, and two empty rows.
    """
    return pd.DataFrame(
        [
            [42, "2024-05-21", " Great GAME! ", "1,234.5", " Recommended ", "1,234 people"],
            [42, "2024-05-21 00:00:00", "Great GAME!", "1234.5", "recommended", "1,234 people"],
            [43, "2024-05-22", "Not\nfor me.", "Unknown Hours", "NOT RECOMMENDED", "No votes, yet"],
            [None, None, "  ", None, None, None],
            [None, None, None, None, None, None],
        ],
        columns=list(RAW_COLUMNS),
    )


def test_cleaning_normalizes_values_without_mutating_input(
    raw_reviews: pd.DataFrame,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Normalize and deduplicate dirty rows while preserving input data and logging removals."""
    original = raw_reviews.copy(deep=True)

    with caplog.at_level(logging.INFO):
        result = clean_reviews(raw_reviews, {42: Game(42, "First game")})

    assert_frame_equal(raw_reviews, original)
    assert len(result) == 2
    assert list(result.index) == [0, 1]
    assert result["app_id"].tolist() == [42, 43]
    assert result["text"].tolist() == ["Great GAME!", "Not\nfor me."]
    assert result["recommendation"].tolist() == ["Recommended", "Not Recommended"]
    assert result["is_recommended"].tolist() == [True, False]
    assert result["helpful_votes"].tolist() == [1234, 0]
    assert result.loc[0, "hours_played"] == 1234.5
    assert pd.isna(result.loc[1, "hours_played"])
    assert result.loc[0, "date"] == pd.Timestamp("2024-05-21")
    assert result.loc[0, "game"] == "First game"
    assert pd.isna(result.loc[1, "game"])
    assert str(result["app_id"].dtype) == "int64"
    assert str(result["date"].dtype) == "datetime64[ns]"
    assert str(result["hours_played"].dtype) == "float64"
    assert str(result["helpful_votes"].dtype) == "int64"
    assert str(result["is_recommended"].dtype) == "bool"
    assert "empty_text=2" in caplog.text
    assert "duplicates=1" in caplog.text
    assert "missing for 1 reviews" in caplog.text


@pytest.mark.parametrize(
    "label,expected",
    [
        ("1,234 people found this review helpful", 1234),
        ("3 people", 3),
        ("No helpful votes", 0),
        ("No votes, yet", 0),
        (",,,", 0),
        ("-3 people", 0),
        (None, 0),
        (float("nan"), 0),
        (5, 5),
    ],
)
def test_vote_labels_are_safe(label: object, expected: int) -> None:
    """Parse thousands-separated votes and return zero for absent or negative counts."""
    assert extract_first_int(label) == expected


@pytest.mark.parametrize("column", ["Game", "Recommendation", "Review"])
def test_missing_columns_have_a_clear_error(raw_reviews: pd.DataFrame, column: str) -> None:
    """Name the missing raw column in the validation error."""
    with pytest.raises(ValueError, match=f"Missing raw review columns: {column}"):
        clean_reviews(raw_reviews.drop(columns=column))


def test_duplicate_column_names_are_rejected(raw_reviews: pd.DataFrame) -> None:
    """Reject ambiguous input with repeated column names before transforming any rows."""
    duplicated = pd.concat([raw_reviews, raw_reviews[["Game"]]], axis=1)

    with pytest.raises(ValueError, match="column names must be unique"):
        clean_reviews(duplicated)


@pytest.mark.parametrize("app_id", [0, -1, 1.5, "not an ID", None, True, 2**63])
def test_invalid_ids_are_not_silently_repaired(raw_reviews: pd.DataFrame, app_id: object) -> None:
    """Reject invalid IDs, including fractions that an integer cast could silently truncate."""
    raw = raw_reviews.iloc[:1].assign(Game=app_id)

    with pytest.raises(ValueError, match="Game/app_id"):
        clean_reviews(raw)


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        "2024-02-30",
        "21/05/2024",
        "2024",
        20240521,
        "2024-05-21T00:00:00+02:00",
    ],
)
def test_invalid_dates_are_not_invented(raw_reviews: pd.DataFrame, value: object) -> None:
    """Reject invalid, partial, numeric, and timezone-aware dates instead of inventing dates."""
    raw = raw_reviews.iloc[:1].assign(Date=value)

    with pytest.raises(ValueError, match="Date must contain valid ISO"):
        clean_reviews(raw)


@pytest.mark.parametrize("label", [None, "", "Unknown", "Positive"])
def test_unknown_recommendations_never_become_negative(
    raw_reviews: pd.DataFrame,
    label: object,
) -> None:
    """Reject missing or unknown labels instead of creating negative training examples."""
    raw = raw_reviews.iloc[:1].assign(Recommendation=label)

    with pytest.raises(ValueError, match="missing or unknown labels"):
        clean_reviews(raw)


@pytest.mark.parametrize("hours", [None, "", "Unknown Hours", "-5", "inf", float("-inf")])
def test_invalid_hours_are_missing(raw_reviews: pd.DataFrame, hours: object) -> None:
    """Represent absent, invalid, negative, and infinite playtime as float NaN."""
    raw = raw_reviews.iloc[:1].assign(**{"Hours Played": hours})

    result = clean_reviews(raw)

    assert pd.isna(result.loc[0, "hours_played"])
    assert str(result["hours_played"].dtype) == "float64"


def test_integer_hours_are_still_float(raw_reviews: pd.DataFrame) -> None:
    """Keep valid zero playtime as float64 even when every source value is an integer."""
    result = clean_reviews(raw_reviews.iloc[:1].assign(**{"Hours Played": "0"}))

    assert result.loc[0, "hours_played"] == 0.0
    assert str(result["hours_played"].dtype) == "float64"


def test_empty_mapping_still_adds_game_column(raw_reviews: pd.DataFrame) -> None:
    """Distinguish an empty game registry from omitting game-name enrichment entirely."""
    result = clean_reviews(raw_reviews.iloc[:1], {})

    assert list(result.columns[:2]) == ["app_id", "game"]
    assert result["game"].isna().all()
    assert "game" not in clean_reviews(raw_reviews.iloc[:1]).columns


@pytest.mark.parametrize("all_blank", [False, True])
def test_empty_result_keeps_typed_schema(raw_reviews: pd.DataFrame, all_blank: bool) -> None:
    """Preserve output columns and numeric types when no review survives cleaning."""
    raw = raw_reviews.iloc[3:] if all_blank else raw_reviews.iloc[:0]

    result = clean_reviews(raw, {})

    assert result.empty
    assert list(result.columns) == [
        "app_id",
        "game",
        "date",
        "text",
        "hours_played",
        "recommendation",
        "helpful_votes",
        "is_recommended",
    ]
    assert str(result["app_id"].dtype) == "int64"
    assert str(result["hours_played"].dtype) == "float64"
    assert str(result["is_recommended"].dtype) == "bool"


def test_extra_metadata_does_not_prevent_review_deduplication(raw_reviews: pd.DataFrame) -> None:
    """Deduplicate equivalent reviews even when unrelated source metadata differs."""
    raw = raw_reviews.iloc[:2].assign(source_file=["first.csv", "second.csv"])

    result = clean_reviews(raw)

    assert len(result) == 1
    assert "source_file" not in result.columns


def test_csv_workflow_preserves_literal_text_and_missing_hours(tmp_path: Path) -> None:
    """Preserve literal text and missing optional values through the complete CSV workflow."""
    reviews = [
        Review(42, date(2024, 5, 21), text, "Unknown Hours", "Recommended", "0")
        for text in ["NA", "NULL", "001", "Great, game!"]
    ]
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"

    write_raw_reviews(raw_dir / "42_reviews.csv", reviews)
    raw = read_raw_reviews(raw_dir)
    expected_text = [review.text for review in reviews]

    assert raw["Review"].tolist() == expected_text

    cleaned = clean_reviews(raw, {})
    write_processed(cleaned, processed_dir)
    restored = read_processed(processed_dir)

    assert restored["text"].tolist() == expected_text
    assert restored["hours_played"].isna().all()
    assert restored["game"].isna().all()
    assert restored["is_recommended"].tolist() == [True] * len(reviews)
    assert pd.api.types.is_datetime64_any_dtype(restored["date"])


def test_numeric_only_review_keeps_leading_zeros_when_loading(tmp_path: Path) -> None:
    """Keep a numeric-looking review as text rather than losing its leading zeros."""
    review = Review(42, date(2024, 5, 21), "001", "1", "Recommended", "0")
    write_raw_reviews(tmp_path / "42_reviews.csv", [review])

    assert clean_reviews(read_raw_reviews(tmp_path)).loc[0, "text"] == "001"
