"""VADER sentiment scoring and distribution fitting for review text."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from scipy import stats

from steam_reviews.analysis.nltk_resources import ensure_nltk_resources

POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05
SENTIMENT_CATEGORIES = ("Positive", "Neutral", "Negative")


@dataclass(frozen=True, slots=True)
class DistributionFits:
    """Store fitted location and scale parameters for three candidate distributions.

    Attributes:
        normal: Mean and standard deviation of the normal fit.
        laplace: Location and scale of the Laplace fit.
        beta: Two shape parameters of the beta fit on scores rescaled to `(0, 1)`.
    """

    normal: tuple[float, float]
    laplace: tuple[float, float]
    beta: tuple[float, float]


def sentiment_category(score: float) -> str:
    """Classify a VADER compound score with the library's standard thresholds.

    Args:
        score: Compound sentiment score in `[-1, 1]`.

    Returns:
        `Positive` for scores at least 0.05, `Negative` for scores at most -0.05,
        otherwise `Neutral`.
    """
    if score >= POSITIVE_THRESHOLD:
        return "Positive"
    if score <= NEGATIVE_THRESHOLD:
        return "Negative"
    return "Neutral"


def sentiment_scores(texts: Iterable[str]) -> list[float]:
    """Compute VADER compound scores for review texts.

    Args:
        texts: Review texts.

    Returns:
        Compound score for every input, in the same order.
    """
    ensure_nltk_resources()
    analyzer = SentimentIntensityAnalyzer()
    return [float(analyzer.polarity_scores(str(text))["compound"]) for text in texts]


def add_sentiment(
    reviews: pd.DataFrame,
    *,
    text_column: str = "processed_review",
) -> pd.DataFrame:
    """Attach compound scores and sentiment categories to reviews.

    Args:
        reviews: Reviews containing the text column to score.
        text_column: Column with processed review text.

    Returns:
        Copy of the frame with `sentiment_score` and `sentiment_category` columns.

    Raises:
        KeyError: The requested text column does not exist.
    """
    if text_column not in reviews.columns:
        raise KeyError(f"Missing review text column: {text_column}")
    result = reviews.copy()
    scores = sentiment_scores(result[text_column])
    result["sentiment_score"] = pd.Series(scores, index=result.index, dtype="float64")
    categories = [sentiment_category(score) for score in scores]
    result["sentiment_category"] = pd.Series(categories, index=result.index, dtype=object)
    return result


def fit_sentiment_distributions(scores: Iterable[float]) -> DistributionFits:
    """Fit normal, Laplace and rescaled beta distributions to sentiment scores.

    Args:
        scores: Finite or non-finite compound scores; non-finite values are dropped.

    Returns:
        Fitted parameters for all three candidate distributions.

    Raises:
        ValueError: Fewer than two finite scores are supplied.
    """
    values = np.asarray([float(score) for score in scores], dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 2:
        raise ValueError("At least two finite sentiment scores are required")
    samples = [float(value) for value in values]
    normal_fit = cast(Callable[..., tuple[float, float]], stats.norm.fit)
    laplace_fit = cast(Callable[..., tuple[float, float]], stats.laplace.fit)
    normal_mean, normal_std = normal_fit(samples)
    laplace_location, laplace_scale = laplace_fit(samples)
    rescaled = np.clip((values + 1.0) / 2.0, 1e-6, 1.0 - 1e-6)
    beta_a, beta_b, _, _ = stats.beta.fit([float(value) for value in rescaled], floc=0, fscale=1)
    return DistributionFits(
        normal=(float(normal_mean), float(normal_std)),
        laplace=(float(laplace_location), float(laplace_scale)),
        beta=(float(beta_a), float(beta_b)),
    )


def sentiment_summary(reviews: pd.DataFrame) -> dict[str, float]:
    """Summarize scores and category shares for one review frame.

    Args:
        reviews: Frame containing `sentiment_score` and `sentiment_category`.

    Returns:
        Count, mean score and the fraction of each sentiment category.

    Raises:
        KeyError: The sentiment columns are missing.
    """
    for column in ("sentiment_score", "sentiment_category"):
        if column not in reviews.columns:
            raise KeyError(f"Missing sentiment column: {column}")
    total = len(reviews)
    summary = {
        "count": float(total),
        "mean_score": float(reviews["sentiment_score"].mean()) if total else float("nan"),
    }
    for category in SENTIMENT_CATEGORIES:
        count = int((reviews["sentiment_category"] == category).sum())
        summary[f"{category.lower()}_share"] = count / total if total else float("nan")
    return summary
