"""Scrape, clean and analyse Steam community reviews.

The core install (pandas only) provides reading, writing and cleaning of review data.
Optional extras add the heavier parts:

- `[scrape]`: `steam_review_analysis.scraping` (Selenium; needs Google Chrome);
- `[analysis]`: `steam_review_analysis.analysis` (NLTK, SciPy, scikit-learn, plots);
- `[sentiment]`: transformer sentiment models (PyTorch, Hugging Face).

Install them as `pip install "steam-review-analysis[analysis]"`.

Data locations are resolved by `steam_review_analysis.config`; see `config.data_dir`.
"""

from importlib.metadata import PackageNotFoundError, version

from steam_review_analysis.config import Game, data_dir, load_games, processed_dir, raw_dir
from steam_review_analysis.io import (
    read_processed,
    read_raw_reviews,
    write_processed,
    write_raw_reviews,
)
from steam_review_analysis.processing.cleaning import clean_reviews
from steam_review_analysis.schema import Review

try:
    __version__ = version("steam-review-analysis")
except PackageNotFoundError:  # pragma: no cover - only when running from an uninstalled tree
    __version__ = "0.0.0"

__all__ = [
    "Game",
    "Review",
    "__version__",
    "clean_reviews",
    "data_dir",
    "load_games",
    "processed_dir",
    "raw_dir",
    "read_processed",
    "read_raw_reviews",
    "write_processed",
    "write_raw_reviews",
]
