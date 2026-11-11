"""Compare sentiment methods against Steam recommendation labels."""

from collections.abc import Sequence

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

SENTIMENT_LABELS = ("Positive", "Neutral", "Negative")
BINARY_LABELS = ("Positive", "Negative")
RECOMMENDATION_LABELS = {
    "Recommended": "Positive",
    "Not Recommended": "Negative",
}


def recommendation_truth(
    reviews: pd.DataFrame,
    *,
    column: str = "recommendation",
) -> pd.Series:
    """Map Steam recommendation labels to binary sentiment labels.

    Args:
        reviews: Frame containing the recommendation column.
        column: Name of the recommendation column.

    Returns:
        `Positive` for recommended reviews, `Negative` for not recommended reviews and
        missing for unknown labels.

    Raises:
        KeyError: The recommendation column does not exist.
    """
    if column not in reviews.columns:
        raise KeyError(f"Missing recommendation column: {column}")
    return reviews[column].astype("string").str.strip().map(RECOMMENDATION_LABELS)


def confusion_matrix(
    truth: pd.Series,
    predicted: pd.Series,
    labels: Sequence[str] = SENTIMENT_LABELS,
) -> pd.DataFrame:
    """Build an actual-versus-predicted count table.

    Args:
        truth: Actual sentiment labels.
        predicted: Predicted sentiment labels.
        labels: Label order for both axes.

    Returns:
        Count table with actual labels as rows and predicted labels as columns.
    """
    matrix = pd.crosstab(truth, predicted).reindex(
        index=list(labels),
        columns=list(labels),
        fill_value=0,
    )
    matrix.index.name = "actual"
    matrix.columns.name = "predicted"
    return matrix


def evaluate_sentiment(truth: pd.Series, predicted: pd.Series) -> dict[str, float]:
    """Score one method on binary recommendation truth.

    Neutral predictions are excluded from accuracy and F1 but counted in `coverage`
    and `neutral_rate`, so a method that abstains often is not rewarded.

    Args:
        truth: Actual binary sentiment labels.
        predicted: Predicted sentiment labels.

    Returns:
        Review count, coverage, neutral rate, accuracy, macro precision/recall/F1,
        weighted F1, Cohen's kappa and Matthews correlation coefficient.
    """
    truth = truth.astype("string")
    predicted = predicted.astype("string")
    valid = truth.isin(BINARY_LABELS) & predicted.notna()
    covered = valid & predicted.isin(BINARY_LABELS)
    valid_count = int(valid.sum())
    covered_count = int(covered.sum())
    result = {
        "reviews": float(valid_count),
        "coverage": covered_count / valid_count if valid_count else float("nan"),
        "neutral_rate": (
            int((valid & predicted.eq("Neutral")).sum()) / valid_count
            if valid_count
            else float("nan")
        ),
    }
    if not covered_count:
        result.update(
            accuracy=float("nan"),
            precision_macro=float("nan"),
            recall_macro=float("nan"),
            f1_macro=float("nan"),
            f1_weighted=float("nan"),
            kappa=float("nan"),
            mcc=float("nan"),
        )
        return result
    y_true = truth[covered]
    y_pred = predicted[covered]
    labels = sorted(set(y_true).union(y_pred))
    result.update(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision_macro=float(
            precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        recall_macro=float(
            recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        f1_macro=float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        f1_weighted=float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
        kappa=(
            float(cohen_kappa_score(y_true, y_pred))
            if y_true.nunique() > 1 and y_pred.nunique() > 1
            else float("nan")
        ),
        mcc=(
            float(matthews_corrcoef(y_true, y_pred))
            if y_true.nunique() > 1 and y_pred.nunique() > 1
            else float("nan")
        ),
    )
    return result


def compare_sentiment_methods(
    scored: pd.DataFrame,
    prediction_columns: Sequence[str],
    *,
    truth_column: str = "recommendation_label",
) -> pd.DataFrame:
    """Score several prediction columns against one truth column.

    Args:
        scored: Frame with the truth column and every prediction column.
        prediction_columns: Column names to compare, one per method.
        truth_column: Column with actual binary sentiment labels.

    Returns:
        One metrics row per prediction column.

    Raises:
        KeyError: The truth column or a prediction column is missing.
    """
    if truth_column not in scored.columns:
        raise KeyError(f"Missing truth column: {truth_column}")
    rows = []
    for method in prediction_columns:
        if method not in scored.columns:
            raise KeyError(f"Missing prediction column: {method}")
        rows.append(
            {
                "method": method,
                **evaluate_sentiment(scored[truth_column], scored[method]),
            }
        )
    return pd.DataFrame(rows)
