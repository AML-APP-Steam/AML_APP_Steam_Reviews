"""Test VADER scoring, sentiment categories and distribution fitting."""

import math

import pandas as pd
import pytest

from steam_review_analysis.analysis.sentiment import (
    add_sentiment,
    fit_sentiment_distributions,
    sentiment_category,
    sentiment_scores,
    sentiment_summary,
)


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.5, "Positive"),
        (0.05, "Positive"),
        (0.0, "Neutral"),
        (-0.05, "Negative"),
        (-1.0, "Negative"),
    ],
)
def test_sentiment_category_uses_vader_thresholds(score: float, expected: str) -> None:
    """Classify scores at, above and below the VADER thresholds."""
    assert sentiment_category(score) == expected


def test_sentiment_scores_follow_text_polarity() -> None:
    """Assign a positive compound score to praise and a negative score to criticism."""
    positive, negative = sentiment_scores(["I love this great game", "I hate this bad game"])

    assert positive > 0
    assert negative < 0


def test_add_sentiment_preserves_input_and_adds_columns(sample_frame: pd.DataFrame) -> None:
    """Score processed reviews without changing the caller's frame."""
    original = sample_frame.copy(deep=True)
    processed = sample_frame.assign(processed_review=sample_frame["text"])
    original_processed = processed.copy(deep=True)

    result = add_sentiment(processed)

    pd.testing.assert_frame_equal(processed, original_processed)
    assert sample_frame.equals(original)
    assert {"sentiment_score", "sentiment_category"}.issubset(result.columns)
    assert result["sentiment_score"].between(-1.0, 1.0).all()
    assert set(result["sentiment_category"]).issubset({"Positive", "Neutral", "Negative"})


def test_add_sentiment_rejects_missing_column(sample_frame: pd.DataFrame) -> None:
    """Report a missing text column with a clear error."""
    with pytest.raises(KeyError, match="processed_review"):
        add_sentiment(sample_frame)


def test_fit_sentiment_distributions_returns_finite_parameters() -> None:
    """Fit all three candidate distributions to varied finite scores."""
    fits = fit_sentiment_distributions([-0.8, -0.3, -0.1, 0.0, 0.2, 0.5, 0.9])

    assert all(math.isfinite(value) for value in (*fits.normal, *fits.laplace, *fits.beta))
    assert fits.beta[0] > 0
    assert fits.beta[1] > 0


def test_fit_sentiment_distributions_rejects_too_few_values() -> None:
    """Require at least two finite scores for fitting."""
    with pytest.raises(ValueError, match="two finite"):
        fit_sentiment_distributions([0.5, float("nan")])


def test_sentiment_summary_reports_counts_and_shares() -> None:
    """Summarize count, mean score and category shares."""
    reviews = pd.DataFrame(
        {
            "sentiment_score": [0.5, 0.0, -0.8, 0.2],
            "sentiment_category": ["Positive", "Neutral", "Negative", "Positive"],
        }
    )

    summary = sentiment_summary(reviews)

    assert summary["count"] == 4
    assert summary["mean_score"] == pytest.approx(-0.025)
    assert summary["positive_share"] == 0.5
    assert summary["neutral_share"] == 0.25
    assert summary["negative_share"] == 0.25


def test_sentiment_summary_rejects_missing_columns() -> None:
    """Report missing sentiment columns with a clear error."""
    with pytest.raises(KeyError, match="sentiment_score"):
        sentiment_summary(pd.DataFrame({"sentiment_category": ["Positive"]}))
