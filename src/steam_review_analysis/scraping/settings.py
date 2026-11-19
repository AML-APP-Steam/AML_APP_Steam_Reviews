"""Scraper parameters, validation and errors, importable without Selenium installed."""

import math
from dataclasses import dataclass


class ScrapingError(RuntimeError):
    """Signal an unsuccessful scrape whose partial results must not be saved."""


@dataclass(frozen=True, slots=True)
class ScraperSettings:
    """Define and validate scraping parameters in one place.

    Attributes:
        target_reviews: Maximum unique reviews per game; 600 is the default sample size.
        browse_filter: Steam review ordering, defaulting to `toprated`.
        language: Review language filter, defaulting to English. UI labels stay English.
        scroll_pause_s: Wait after scrolling, in seconds; 2 allows dynamic cards to load.
        wait_timeout_s: Maximum wait for the page body, in seconds.
        page_load_timeout_s: Maximum navigation time, in seconds, before failing the scrape.
        max_stalled_scrolls: Consecutive scrolls without new valid reviews before returning
            a smaller sample. The default 5 tolerates several delayed page updates.
        max_stale_retries: Consecutive stale DOM failures before raising an error.
            A successful read resets this counter; the default is 3 attempts.
    """

    target_reviews: int = 600
    browse_filter: str = "toprated"
    language: str = "english"
    scroll_pause_s: float = 2.0
    wait_timeout_s: float = 5.0
    page_load_timeout_s: float = 30.0
    max_stalled_scrolls: int = 5
    max_stale_retries: int = 3

    def __post_init__(self) -> None:
        """Reject invalid limits before starting a browser.

        Raises:
            ValueError: A count is not a positive integer, a time is invalid, or a filter
                is empty. Scroll pauses may be zero; timeouts must be positive.
        """
        for name in ("target_reviews", "max_stalled_scrolls", "max_stale_retries"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("scroll_pause_s", "wait_timeout_s", "page_load_timeout_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0 or (name != "scroll_pause_s" and value == 0):
                requirement = "non-negative" if name == "scroll_pause_s" else "positive"
                raise ValueError(f"{name} must be finite and {requirement}")
        if not self.browse_filter.strip() or not self.language.strip():
            raise ValueError("browse_filter and language must not be empty")


def validate_app_id(app_id: int) -> None:
    """Validate a Steam application identifier before navigation or output planning.

    Args:
        app_id: Application identifier supplied by the CLI or Python caller.

    Raises:
        ValueError: The identifier is not a positive integer; booleans are also rejected.
    """
    if isinstance(app_id, bool) or not isinstance(app_id, int) or app_id <= 0:
        raise ValueError("app_id must be a positive integer")
