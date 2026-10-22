"""Shared fixtures and sample datasets for analysis tests."""

from pathlib import Path

import pandas as pd
import pytest

from steam_reviews.io import write_processed


def sample_reviews() -> pd.DataFrame:
    """Build two small game datasets spanning before and after the default cutoff.

    Returns:
        Processed review frame with text, metadata and recommendation labels.
    """
    return pd.DataFrame(
        [
            [1, "Alpha", "2023-09-28", "I love this game, great fun!", 10.0, "Recommended", 5],
            [
                1,
                "Alpha",
                "2023-09-29",
                "Not recommended, terrible cheat.",
                2.0,
                "Not Recommended",
                1,
            ],
            [
                1,
                "Alpha",
                "2023-10-01",
                "Great game with friends, love it.",
                100.0,
                "Recommended",
                10,
            ],
            [1, "Alpha", "2024-01-01", "Cheater and bug, bad game.", 1.0, "Not Recommended", 0],
            [1, "Alpha", "2024-02-01", "Amazing fun game, love playing.", 50.0, "Recommended", 3],
            [2, "Beta", "2023-01-01", "Excellent game.", 5.0, "Recommended", 2],
            [2, "Beta", "2023-06-01", "Boring and bad.", 0.5, "Not Recommended", 0],
        ],
        columns=[
            "app_id",
            "game",
            "date",
            "text",
            "hours_played",
            "recommendation",
            "helpful_votes",
        ],
    ).assign(is_recommended=lambda frame: frame["recommendation"] == "Recommended")


@pytest.fixture
def sample_frame() -> pd.DataFrame:
    """Return the in-memory sample review frame.

    Returns:
        Processed review frame used by unit tests.
    """
    return sample_reviews()


@pytest.fixture
def processed_dir(tmp_path: Path) -> Path:
    """Write the sample reviews as a processed dataset in a temporary directory.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        Directory containing `reviews.csv`.
    """
    write_processed(sample_reviews(), tmp_path)
    return tmp_path
