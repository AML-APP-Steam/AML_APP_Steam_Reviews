"""Test period correlations, yearly aggregates and engagement splits."""

import pandas as pd
import pytest

from steam_review_analysis.analysis.dataset import add_year
from steam_review_analysis.analysis.statistics import (
    engagement_split,
    pearson_correlations,
    yearly_mean,
)


def _scored_frame(sample_frame: pd.DataFrame) -> pd.DataFrame:
    """Attach a date column and deterministic sentiment scores to sample reviews.

    Args:
        sample_frame: In-memory review fixture.

    Returns:
        Frame ready for statistics helpers.
    """
    scores = [-0.2, 0.3, 0.5, -0.6, 0.8, 0.4, -0.7]
    return sample_frame.assign(
        date=pd.to_datetime(sample_frame["date"]),
        sentiment_score=scores,
    )


def test_pearson_correlations_returns_before_and_after_rows(sample_frame: pd.DataFrame) -> None:
    """Measure each requested pair before and after the cutoff."""
    reviews = _scored_frame(sample_frame)

    result = pearson_correlations(reviews, "2023-09-29")

    assert result["comparison"].tolist() == [
        "sentiment_score vs hours_played",
        "sentiment_score vs helpful_votes",
    ]
    assert set(result.columns) == {"comparison", "before", "after"}
    assert result[["before", "after"]].notna().all().all()


def test_pearson_correlations_rejects_missing_column(sample_frame: pd.DataFrame) -> None:
    """Report a requested column that is absent from the frame."""
    reviews = _scored_frame(sample_frame).drop(columns=["helpful_votes"])
    with pytest.raises(KeyError):
        pearson_correlations(reviews, "2023-09-29")


def test_yearly_mean_aggregates_and_sorts(sample_frame: pd.DataFrame) -> None:
    """Average sentiment per year and sort ascending."""
    reviews = add_year(_scored_frame(sample_frame))

    result = yearly_mean(reviews)

    assert result["year"].tolist() == sorted(result["year"].tolist())
    assert result.loc[result["year"] == 2024, "sentiment_score"].iloc[0] == pytest.approx(0.1)


def test_yearly_mean_rejects_missing_column(sample_frame: pd.DataFrame) -> None:
    """Report a missing year column."""
    with pytest.raises(KeyError, match="year"):
        yearly_mean(_scored_frame(sample_frame))


def test_engagement_split_uses_median_hours(sample_frame: pd.DataFrame) -> None:
    """Flag reviews whose playtime exceeds the median."""
    reviews = _scored_frame(sample_frame)

    result = engagement_split(reviews)

    assert result["engaged"].dtype == bool
    assert result.loc[result["hours_played"] > reviews["hours_played"].median(), "engaged"].all()


def test_engagement_split_rejects_missing_hours(sample_frame: pd.DataFrame) -> None:
    """Report a missing hours column."""
    with pytest.raises(KeyError, match="hours_played"):
        engagement_split(_scored_frame(sample_frame).drop(columns=["hours_played"]))
