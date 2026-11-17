"""Compare metadata and sentiment across review periods."""

from collections.abc import Sequence

import pandas as pd

from steam_review_analysis.analysis.dataset import split_by_cutoff

DEFAULT_CORRELATION_PAIRS = (
    ("sentiment_score", "hours_played"),
    ("sentiment_score", "helpful_votes"),
)


def _correlation(frame: pd.DataFrame, left: str, right: str) -> float:
    """Measure Pearson correlation when a period has enough observations.

    Args:
        frame: Reviews from one period.
        left: First numeric column.
        right: Second numeric column.

    Returns:
        Correlation, or NaN when fewer than two observations are available.
    """
    if len(frame) < 2:
        return float("nan")
    return float(frame[left].corr(frame[right]))


def pearson_correlations(
    reviews: pd.DataFrame,
    cutoff: str | pd.Timestamp,
    *,
    pairs: Sequence[tuple[str, str]] = DEFAULT_CORRELATION_PAIRS,
    date_column: str = "date",
) -> pd.DataFrame:
    """Compare Pearson correlations before and after a cutoff.

    Args:
        reviews: Reviews with datetime, sentiment and metadata columns.
        cutoff: Boundary date; rows on the boundary belong to the later group.
        pairs: Column pairs whose Pearson correlation is measured.
        date_column: Name of the datetime column.

    Returns:
        Frame with `comparison`, `before` and `after` columns, one row per pair.

    Raises:
        KeyError: A requested column does not exist.
        ValueError: The cutoff cannot be parsed.
    """
    before, after = split_by_cutoff(reviews, cutoff, date_column=date_column)
    rows: list[dict[str, object]] = []
    for left, right in pairs:
        rows.append(
            {
                "comparison": f"{left} vs {right}",
                "before": _correlation(before, left, right),
                "after": _correlation(after, left, right),
            }
        )
    return pd.DataFrame(rows, columns=["comparison", "before", "after"])


def yearly_mean(reviews: pd.DataFrame, *, value_column: str = "sentiment_score") -> pd.DataFrame:
    """Aggregate the mean value per review year.

    Args:
        reviews: Reviews with `year` and the requested value column.
        value_column: Numeric column to average.

    Returns:
        Frame with `year` and the mean value, sorted by year.

    Raises:
        KeyError: A requested column does not exist.
    """
    for column in ("year", value_column):
        if column not in reviews.columns:
            raise KeyError(f"Missing column: {column}")
    grouped = reviews.groupby("year")[value_column].mean().reset_index()
    return grouped.sort_values("year").reset_index(drop=True)


def engagement_split(
    reviews: pd.DataFrame,
    *,
    hours_column: str = "hours_played",
) -> pd.DataFrame:
    """Label reviews as high or low engagement relative to the median playtime.

    Args:
        reviews: Reviews with a numeric hours column.
        hours_column: Numeric playtime column used for the median split.

    Returns:
        Copy of the frame with an `engaged` boolean column.

    Raises:
        KeyError: The hours column does not exist.
    """
    if hours_column not in reviews.columns:
        raise KeyError(f"Missing column: {hours_column}")
    result = reviews.copy()
    median = result[hours_column].median()
    result["engaged"] = result[hours_column] > median
    return result
