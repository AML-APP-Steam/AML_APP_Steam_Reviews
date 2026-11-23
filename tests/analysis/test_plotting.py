"""Test plotting helpers with the non-interactive Agg backend."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from scipy import stats

from steam_review_analysis.analysis.plotting import (
    plot_boxplot,
    plot_distribution_fits,
    plot_engagement_boxplot,
    plot_scatter,
    plot_sentiment_categories,
    plot_sentiment_distribution,
    plot_sentiment_vs_recommendation,
    plot_word_frequencies,
    plot_wordcloud,
    plot_yearly_sentiment,
    save_figure,
)
from steam_review_analysis.analysis.sentiment import fit_sentiment_distributions

plt.switch_backend("Agg")


@pytest.fixture
def scored_frame(sample_frame: pd.DataFrame) -> pd.DataFrame:
    """Attach deterministic sentiment columns to the sample reviews.

    Args:
        sample_frame: In-memory review fixture.

    Returns:
        Frame ready for plotting helpers.
    """
    scores = [-0.2, 0.3, 0.5, -0.6, 0.8, 0.4, -0.7]
    categories = [
        "Negative",
        "Positive",
        "Positive",
        "Negative",
        "Positive",
        "Positive",
        "Negative",
    ]
    return sample_frame.assign(
        sentiment_score=scores,
        sentiment_category=categories,
        engaged=[False, False, True, False, True, False, False],
    )


def test_plot_word_frequencies_returns_figure() -> None:
    """Draw a bar chart from a frequency table."""
    # Arrange
    table = pd.DataFrame({"word": ["game", "play"], "frequency": [5, 3]})

    # Act
    figure = plot_word_frequencies(table, title="Frequencies")

    # Assert
    assert isinstance(figure, Figure)
    plt.close(figure)


def test_plot_word_frequencies_rejects_empty_table() -> None:
    """Reject a frequency table without rows."""
    with pytest.raises(ValueError, match="empty"):
        plot_word_frequencies(pd.DataFrame({"word": [], "frequency": []}), title="Frequencies")


def test_plot_wordcloud_returns_figure() -> None:
    """Draw a word cloud from nonempty text."""
    # Act
    figure = plot_wordcloud("game play fun", title="Cloud")

    # Assert
    assert isinstance(figure, Figure)
    plt.close(figure)


def test_plot_wordcloud_rejects_empty_text() -> None:
    """Reject text that contains no words."""
    with pytest.raises(ValueError, match="empty"):
        plot_wordcloud("   ", title="Cloud")


def test_plot_sentiment_distribution_returns_figure() -> None:
    """Overlay score histograms for two periods."""
    # Act
    figure = plot_sentiment_distribution(
        pd.Series([-0.5, 0.1, 0.4]),
        pd.Series([0.2, 0.8]),
        cutoff="2024-01-01",
    )

    # Assert
    assert isinstance(figure, Figure)
    plt.close(figure)


def test_plot_sentiment_distribution_rejects_empty_periods() -> None:
    """Reject plotting when both periods are empty."""
    with pytest.raises(ValueError, match="empty periods"):
        plot_sentiment_distribution(pd.Series(dtype=float), pd.Series(dtype=float), cutoff="x")


def test_plot_sentiment_categories_returns_two_panel_figure(scored_frame: pd.DataFrame) -> None:
    """Draw category counts for before and after periods."""
    # Act
    figure = plot_sentiment_categories(
        scored_frame.iloc[:3],
        scored_frame.iloc[3:],
        cutoff="2024-01-01",
    )

    # Assert
    assert isinstance(figure, Figure)
    assert len(figure.axes) == 2
    plt.close(figure)


def test_plot_yearly_sentiment_returns_figure() -> None:
    """Draw average sentiment per year."""
    # Arrange
    yearly = pd.DataFrame({"year": [2022, 2023, 2024], "sentiment_score": [0.1, 0.3, -0.2]})

    # Act
    figure = plot_yearly_sentiment(yearly)

    # Assert
    assert isinstance(figure, Figure)
    plt.close(figure)


def test_plot_distribution_fits_returns_figure() -> None:
    """Overlay fitted distributions on a score histogram."""
    # Act
    figure = plot_distribution_fits([-0.8, -0.2, 0.0, 0.4, 0.9], title="Fit")

    # Assert
    assert isinstance(figure, Figure)
    plt.close(figure)


def test_plot_distribution_fits_scales_rescaled_beta_density() -> None:
    """Divide the beta density by two after mapping its support onto the score range."""
    # Arrange
    scores = [-0.8, -0.2, 0.0, 0.4, 0.9]

    # Act
    figure = plot_distribution_fits(scores, title="Fit")

    # Assert
    beta_line = next(
        line for line in figure.axes[0].get_lines() if line.get_label() == "Beta (rescaled)"
    )
    beta_x = np.linspace(1e-6, 1.0 - 1e-6, 300)
    fits = fit_sentiment_distributions(scores)

    np.testing.assert_allclose(
        beta_line.get_ydata(),
        stats.beta.pdf(beta_x, *fits.beta) / 2.0,
    )
    np.testing.assert_allclose(beta_line.get_xdata(), beta_x * 2.0 - 1.0)
    plt.close(figure)


def test_plot_sentiment_vs_recommendation_returns_figure(scored_frame: pd.DataFrame) -> None:
    """Draw grouped sentiment counts by recommendation."""
    # Act
    figure = plot_sentiment_vs_recommendation(scored_frame)

    # Assert
    assert isinstance(figure, Figure)
    plt.close(figure)


def test_plot_boxplot_returns_figure(scored_frame: pd.DataFrame) -> None:
    """Draw a sentiment box plot by recommendation."""
    # Act
    figure = plot_boxplot(
        scored_frame,
        category_column="recommendation",
        value_column="sentiment_score",
        title="By recommendation",
    )

    # Assert
    assert isinstance(figure, Figure)
    plt.close(figure)


def test_plot_engagement_boxplot_returns_figure(scored_frame: pd.DataFrame) -> None:
    """Draw sentiment by recommendation and engagement."""
    # Act
    figure = plot_engagement_boxplot(scored_frame)

    # Assert
    assert isinstance(figure, Figure)
    plt.close(figure)


def test_plot_scatter_returns_figure(scored_frame: pd.DataFrame) -> None:
    """Draw sentiment against playtime on a logarithmic axis."""
    # Act
    figure = plot_scatter(
        scored_frame,
        x_column="hours_played",
        title="Sentiment vs hours",
    )

    # Assert
    assert isinstance(figure, Figure)
    plt.close(figure)


def test_save_figure_writes_png(tmp_path: Path) -> None:
    """Write a figure to the requested destination."""
    # Arrange
    figure = plot_yearly_sentiment(
        pd.DataFrame({"year": [2024], "sentiment_score": [0.5]}),
    )

    # Act
    path = save_figure(figure, tmp_path / "nested" / "plot.png")

    # Assert
    assert path.exists()
    assert path.stat().st_size > 0
