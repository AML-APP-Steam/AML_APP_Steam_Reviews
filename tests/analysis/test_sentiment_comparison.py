"""Test sentiment comparison metrics against recommendation labels."""

import pandas as pd
import pytest

from steam_review_analysis.analysis.sentiment_comparison import (
    compare_sentiment_methods,
    confusion_matrix,
    evaluate_sentiment,
    recommendation_truth,
)


def test_recommendation_truth_maps_steam_labels() -> None:
    """Map recommendations to binary sentiment and leave unknowns missing."""
    # Arrange
    reviews = pd.DataFrame({"recommendation": ["Recommended", "Not Recommended", "Unknown"]})

    # Act
    truth = recommendation_truth(reviews)

    # Assert
    assert truth.tolist()[:2] == ["Positive", "Negative"]
    assert pd.isna(truth.iloc[2])


def test_recommendation_truth_rejects_missing_column() -> None:
    """Report a missing recommendation column."""
    with pytest.raises(KeyError, match="recommendation"):
        recommendation_truth(pd.DataFrame({"text": ["review"]}))


def test_confusion_matrix_counts_labels_in_fixed_order() -> None:
    """Include every label on both axes even when it never occurs."""
    # Arrange
    truth = pd.Series(["Positive", "Negative", "Positive"])
    predicted = pd.Series(["Positive", "Neutral", "Negative"])

    # Act
    matrix = confusion_matrix(truth, predicted)

    # Assert
    assert list(matrix.index) == ["Positive", "Neutral", "Negative"]
    assert list(matrix.columns) == ["Positive", "Neutral", "Negative"]
    assert matrix.loc["Positive", "Positive"] == 1
    assert matrix.loc["Positive", "Negative"] == 1
    assert matrix.loc["Negative", "Neutral"] == 1


def test_evaluate_sentiment_scores_binary_predictions() -> None:
    """Score accuracy and binary agreement while counting neutral abstentions."""
    # Arrange
    truth = pd.Series(["Positive", "Positive", "Negative", "Negative"])
    predicted = pd.Series(["Positive", "Neutral", "Negative", "Negative"])

    # Act
    metrics = evaluate_sentiment(truth, predicted)

    # Assert
    assert metrics["reviews"] == 4
    assert metrics["coverage"] == pytest.approx(0.75)
    assert metrics["neutral_rate"] == pytest.approx(0.25)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["f1_macro"] == pytest.approx(1.0)
    assert metrics["kappa"] == pytest.approx(1.0)
    assert metrics["mcc"] == pytest.approx(1.0)


def test_evaluate_sentiment_reports_missing_metrics_without_coverage() -> None:
    """Return NaN metrics when a method only predicts neutral."""
    # Arrange
    truth = pd.Series(["Positive", "Negative"])
    predicted = pd.Series(["Neutral", "Neutral"])

    # Act
    metrics = evaluate_sentiment(truth, predicted)

    # Assert
    assert metrics["coverage"] == 0.0
    assert metrics["neutral_rate"] == 1.0
    assert metrics["accuracy"] != metrics["accuracy"]


def test_compare_sentiment_methods_returns_one_row_per_method() -> None:
    """Compare several prediction columns against one truth column."""
    # Arrange
    scored = pd.DataFrame(
        {
            "recommendation_label": ["Positive", "Positive", "Negative", "Negative"],
            "vader": ["Positive", "Neutral", "Negative", "Negative"],
            "transformer": ["Positive", "Positive", "Negative", "Negative"],
        }
    )

    # Act
    table = compare_sentiment_methods(scored, ["vader", "transformer"])

    # Assert
    assert table["method"].tolist() == ["vader", "transformer"]
    assert table.loc[0, "coverage"] == pytest.approx(0.75)
    assert table.loc[1, "accuracy"] == pytest.approx(1.0)


def test_compare_sentiment_methods_rejects_missing_columns() -> None:
    """Report missing truth or prediction columns."""
    # Arrange
    scored = pd.DataFrame({"recommendation_label": ["Positive"]})

    # Act & Assert
    with pytest.raises(KeyError, match="prediction"):
        compare_sentiment_methods(scored, ["vader"])
    with pytest.raises(KeyError, match="truth"):
        compare_sentiment_methods(scored, ["vader"], truth_column="missing")
