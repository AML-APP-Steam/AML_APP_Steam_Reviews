"""Plotting helpers shared by the analysis notebooks and CLI."""

from collections.abc import Sequence
from pathlib import Path
from typing import cast

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy import stats
from wordcloud import WordCloud

from steam_reviews.analysis.sentiment import (
    SENTIMENT_CATEGORIES,
    fit_sentiment_distributions,
)

RECOMMENDATION_COLORS = {"Recommended": "steelblue", "Not Recommended": "orange"}


def _figure_and_axes(ax: Axes | None, figsize: tuple[float, float]) -> tuple[Figure, Axes]:
    """Return a supplied axes or create a new figure and axes.

    Args:
        ax: Existing axes to draw on, if any.
        figsize: Size used only when a new figure is created.

    Returns:
        Figure and axes ready for drawing.
    """
    if ax is not None:
        return cast(Figure, ax.figure), ax
    figure, axes = plt.subplots(figsize=figsize)
    return figure, axes


def plot_wordcloud(text: str, *, title: str, ax: Axes | None = None) -> Figure:
    """Draw a word cloud for one corpus of text.

    Args:
        text: Space-separated words; punctuation is ignored by the cloud generator.
        title: Figure title.
        ax: Optional axes to draw on.

    Returns:
        Figure containing the word cloud.

    Raises:
        ValueError: The text contains no words.
    """
    if not text.strip():
        raise ValueError("Cannot draw a word cloud from empty text")
    cloud = WordCloud(
        width=800,
        height=400,
        background_color="white",
        max_words=200,
        collocations=False,
    ).generate(text)
    figure, axes = _figure_and_axes(ax, (10, 5))
    axes.imshow(cloud, interpolation="bilinear")
    axes.axis("off")
    axes.set_title(title)
    return figure


def plot_word_frequencies(
    frequencies: pd.DataFrame, *, title: str, ax: Axes | None = None
) -> Figure:
    """Draw a horizontal bar chart of the most frequent words.

    Args:
        frequencies: Frame with `word` and `frequency` columns.
        title: Figure title.
        ax: Optional axes to draw on.

    Returns:
        Figure containing the bar chart.

    Raises:
        ValueError: The frame is empty or lacks required columns.
    """
    required = {"word", "frequency"}
    if not required.issubset(frequencies.columns):
        raise ValueError("frequencies must contain word and frequency columns")
    if frequencies.empty:
        raise ValueError("Cannot plot an empty frequency table")
    data = frequencies.iloc[::-1]
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(data)))
    figure, axes = _figure_and_axes(ax, (12, 6))
    axes.barh(data["word"].astype(str), data["frequency"].astype(float), color=colors)
    axes.set_title(title)
    axes.set_xlabel("Frequency")
    axes.set_ylabel("Word")
    figure.tight_layout()
    return figure


def plot_sentiment_distribution(
    before: pd.Series,
    after: pd.Series,
    *,
    cutoff: str,
    ax: Axes | None = None,
) -> Figure:
    """Compare sentiment score densities before and after a cutoff.

    Args:
        before: Compound scores from the earlier period.
        after: Compound scores from the later period.
        cutoff: Cutoff label used in the title.
        ax: Optional axes to draw on.

    Returns:
        Figure containing the overlaid histograms.

    Raises:
        ValueError: Both periods have no scores.
    """
    if before.empty and after.empty:
        raise ValueError("Cannot plot sentiment distributions for empty periods")
    figure, axes = _figure_and_axes(ax, (12, 5))
    axes.hist(before, bins=30, density=True, alpha=0.5, color="blue", label="Before")
    axes.hist(after, bins=30, density=True, alpha=0.5, color="green", label="After")
    axes.set_title(f"Sentiment Score Distribution: Before vs After {cutoff}")
    axes.set_xlabel("Compound Sentiment Score")
    axes.set_ylabel("Density")
    axes.legend()
    figure.tight_layout()
    return figure


def _category_counts(reviews: pd.DataFrame) -> list[int]:
    """Count sentiment categories in their canonical order.

    Args:
        reviews: Frame with a `sentiment_category` column.

    Returns:
        Counts ordered as positive, neutral, negative.
    """
    counts = reviews["sentiment_category"].value_counts()
    return [int(counts.get(category, 0)) for category in SENTIMENT_CATEGORIES]


def plot_sentiment_categories(
    before: pd.DataFrame,
    after: pd.DataFrame,
    *,
    cutoff: str,
) -> Figure:
    """Compare sentiment category counts across two periods.

    Args:
        before: Reviews from the earlier period.
        after: Reviews from the later period.
        cutoff: Cutoff label used in the titles.

    Returns:
        Figure with one panel per period.

    Raises:
        ValueError: A frame lacks the sentiment category column.
    """
    for frame in (before, after):
        if "sentiment_category" not in frame.columns:
            raise ValueError("Reviews must contain a sentiment_category column")
    figure = plt.figure(figsize=(14, 5))
    positions = np.arange(len(SENTIMENT_CATEGORIES))
    panels = (
        (figure.add_subplot(1, 2, 1), before, f"Sentiment Categories (Before {cutoff})", "Blues"),
        (figure.add_subplot(1, 2, 2), after, f"Sentiment Categories (On/After {cutoff})", "Greens"),
    )
    for axes, reviews, title, palette in panels:
        colors = matplotlib.colormaps[palette](np.linspace(0.35, 0.85, len(SENTIMENT_CATEGORIES)))
        axes.bar(positions, _category_counts(reviews), color=colors)
        axes.set_xticks(positions, SENTIMENT_CATEGORIES)
        axes.set_title(title)
        axes.set_xlabel("Sentiment")
        axes.set_ylabel("Count")
    figure.tight_layout()
    return figure


def plot_yearly_sentiment(yearly: pd.DataFrame, *, ax: Axes | None = None) -> Figure:
    """Draw mean sentiment by year.

    Args:
        yearly: Frame from `statistics.yearly_mean` with `year` and `sentiment_score`.
        ax: Optional axes to draw on.

    Returns:
        Figure containing the line plot.

    Raises:
        ValueError: Required columns are missing or the table is empty.
    """
    required = {"year", "sentiment_score"}
    if not required.issubset(yearly.columns):
        raise ValueError("yearly must contain year and sentiment_score columns")
    if yearly.empty:
        raise ValueError("Cannot plot an empty yearly table")
    figure, axes = _figure_and_axes(ax, (10, 6))
    axes.plot(yearly["year"], yearly["sentiment_score"], marker="o")
    axes.set_title("Average Sentiment Score Over the Years")
    axes.set_xlabel("Year")
    axes.set_ylabel("Average Sentiment Score")
    axes.grid(True)
    figure.tight_layout()
    return figure


def plot_distribution_fits(
    scores: Sequence[float] | pd.Series,
    *,
    title: str,
    ax: Axes | None = None,
) -> Figure:
    """Overlay normal, Laplace and beta fits on a sentiment score histogram.

    Args:
        scores: Compound sentiment scores for one period.
        title: Figure title.
        ax: Optional axes to draw on.

    Returns:
        Figure containing the histogram and fitted densities.

    Raises:
        ValueError: Fewer than two finite scores are supplied.
    """
    values = np.asarray([float(score) for score in scores], dtype=float)
    values = values[np.isfinite(values)]
    fits = fit_sentiment_distributions(values)
    figure, axes = _figure_and_axes(ax, (12, 6))
    axes.hist(
        values, bins=30, density=True, color="lightgray", edgecolor="black", label="Empirical"
    )
    x_values = np.linspace(values.min(), values.max(), 300)
    axes.plot(x_values, stats.norm.pdf(x_values, *fits.normal), "--", label="Normal")
    axes.plot(x_values, stats.laplace.pdf(x_values, *fits.laplace), "-.", label="Laplace")
    beta_x = np.linspace(1e-6, 1.0 - 1e-6, 300)
    axes.plot(beta_x * 2.0 - 1.0, stats.beta.pdf(beta_x, *fits.beta), label="Beta (rescaled)")
    axes.set_title(f"Distribution Fit - {title}")
    axes.set_xlabel("Sentiment Score")
    axes.set_ylabel("Density")
    axes.legend()
    axes.grid(True)
    figure.tight_layout()
    return figure


def plot_sentiment_vs_recommendation(
    reviews: pd.DataFrame,
    *,
    ax: Axes | None = None,
) -> Figure:
    """Compare sentiment categories across recommendation labels.

    Args:
        reviews: Reviews with `sentiment_category` and `recommendation` columns.
        ax: Optional axes to draw on.

    Returns:
        Figure containing grouped bars.

    Raises:
        ValueError: Required columns are missing.
    """
    required = {"sentiment_category", "recommendation"}
    if not required.issubset(reviews.columns):
        raise ValueError("Reviews must contain sentiment_category and recommendation columns")
    counts = (
        pd.crosstab(reviews["sentiment_category"], reviews["recommendation"])
        .reindex(SENTIMENT_CATEGORIES)
        .fillna(0)
    )
    figure, axes = _figure_and_axes(ax, (10, 6))
    positions = np.arange(len(SENTIMENT_CATEGORIES))
    width = 0.4
    for offset, (column, color) in zip(
        (-width / 2, width / 2),
        RECOMMENDATION_COLORS.items(),
        strict=True,
    ):
        values = counts[column] if column in counts.columns else np.zeros(len(counts))
        axes.bar(positions + offset, values, width, label=column, color=color)
    axes.set_xticks(positions, SENTIMENT_CATEGORIES)
    axes.set_title("Sentiment vs User Recommendation")
    axes.set_xlabel("Sentiment")
    axes.set_ylabel("Count")
    axes.legend()
    figure.tight_layout()
    return figure


def plot_boxplot(
    reviews: pd.DataFrame,
    *,
    category_column: str,
    value_column: str,
    title: str,
    ax: Axes | None = None,
) -> Figure:
    """Draw a box plot of a numeric value grouped by a category.

    Args:
        reviews: Reviews containing both columns.
        category_column: Column defining the groups.
        value_column: Numeric column to summarize.
        title: Figure title.
        ax: Optional axes to draw on.

    Returns:
        Figure containing the box plot.

    Raises:
        ValueError: A required column is missing or every group is empty.
    """
    for column in (category_column, value_column):
        if column not in reviews.columns:
            raise ValueError(f"Missing column: {column}")
    categories = [
        value
        for value in ("Recommended", "Not Recommended")
        if value in set(reviews[category_column])
    ]
    if not categories:
        categories = sorted(str(value) for value in reviews[category_column].dropna().unique())
    groups = [
        reviews.loc[reviews[category_column] == category, value_column].dropna().to_numpy()
        for category in categories
    ]
    if not any(group.size for group in groups):
        raise ValueError("Cannot plot a box plot without values")
    figure, axes = _figure_and_axes(ax, (10, 6))
    axes.boxplot(groups, tick_labels=categories)
    axes.set_title(title)
    axes.set_xlabel(category_column.replace("_", " ").title())
    axes.set_ylabel(value_column.replace("_", " ").title())
    axes.grid(True)
    figure.tight_layout()
    return figure


def plot_engagement_boxplot(reviews: pd.DataFrame, *, ax: Axes | None = None) -> Figure:
    """Compare sentiment by recommendation and median playtime engagement.

    Args:
        reviews: Reviews with `recommendation`, `engaged` and `sentiment_score` columns.
        ax: Optional axes to draw on.

    Returns:
        Figure containing the grouped box plot.

    Raises:
        ValueError: Required columns are missing or every group is empty.
    """
    for column in ("recommendation", "engaged", "sentiment_score"):
        if column not in reviews.columns:
            raise ValueError(f"Missing column: {column}")
    labels: list[str] = []
    groups: list[np.ndarray] = []
    for recommendation in ("Recommended", "Not Recommended"):
        for engaged, suffix in ((False, "Low"), (True, "High")):
            mask = (reviews["recommendation"] == recommendation) & (reviews["engaged"] == engaged)
            values = reviews.loc[mask, "sentiment_score"].dropna().to_numpy()
            labels.append(f"{recommendation}\n{suffix} engagement")
            groups.append(values)
    if not any(group.size for group in groups):
        raise ValueError("Cannot plot a box plot without values")
    figure, axes = _figure_and_axes(ax, (10, 6))
    axes.boxplot(groups, tick_labels=labels)
    axes.set_title("Sentiment Score by Recommendation (Engagement Split)")
    axes.set_ylabel("Sentiment Score")
    axes.grid(True)
    figure.tight_layout()
    return figure


def plot_scatter(
    reviews: pd.DataFrame,
    *,
    x_column: str,
    title: str,
    log_x: bool = True,
    ax: Axes | None = None,
) -> Figure:
    """Scatter sentiment against a numeric metadata column by recommendation.

    Args:
        reviews: Reviews with `sentiment_score`, `recommendation` and the x column.
        x_column: Numeric metadata column plotted on the horizontal axis.
        title: Figure title.
        log_x: Use a logarithmic horizontal axis and drop nonpositive values.
        ax: Optional axes to draw on.

    Returns:
        Figure containing the scatter plot.

    Raises:
        ValueError: Required columns are missing or no finite points remain.
    """
    for column in ("sentiment_score", "recommendation", x_column):
        if column not in reviews.columns:
            raise ValueError(f"Missing column: {column}")
    figure, axes = _figure_and_axes(ax, (10, 6))
    for recommendation, color in RECOMMENDATION_COLORS.items():
        subset = reviews.loc[reviews["recommendation"] == recommendation]
        if log_x:
            subset = subset.loc[subset[x_column] > 0]
        axes.scatter(
            subset[x_column],
            subset["sentiment_score"],
            alpha=0.6,
            label=recommendation,
            color=color,
        )
    if log_x:
        axes.set_xscale("log")
    axes.set_title(title)
    axes.set_xlabel(x_column.replace("_", " ").title())
    axes.set_ylabel("Sentiment Score")
    axes.grid(True)
    axes.legend()
    figure.tight_layout()
    return figure


def save_figure(figure: Figure, path: Path) -> Path:
    """Write a figure to disk and release its memory.

    Args:
        figure: Figure to save.
        path: Destination image path; parent directories are created.

    Returns:
        The destination path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return path
