"""Collect Steam reviews with bounded scrolling and explicit failure reporting."""

import logging
import math
import time
from dataclasses import dataclass
from urllib.parse import urlencode

from selenium.common.exceptions import (
    NoSuchElementException,
    StaleElementReferenceException,
    WebDriverException,
)
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support import expected_conditions as ec
from selenium.webdriver.support.ui import WebDriverWait

from steam_reviews.schema import NO_HELPFUL_VOTES, Review
from steam_reviews.scraping.parsers import parse_review_date, strip_hours_label, strip_review_date

logger = logging.getLogger(__name__)

CARD_SELECTOR = ".apphub_Card"
TEXT_SELECTOR = ".apphub_CardTextContent"
DATE_SELECTOR = ".date_posted"
HOURS_SELECTOR = ".hours"
RECOMMENDATION_SELECTOR = ".title"
HELPFUL_SELECTOR = ".found_helpful"


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


def review_url(app_id: int, browse_filter: str, language: str) -> str:
    """Build a community review URL with English interface labels.

    Args:
        app_id: Positive Steam application identifier.
        browse_filter: Steam ordering value, e.g. `toprated` or `mostrecent`.
        language: Review content language; independent of the English UI language.

    Returns:
        URL with escaped query parameters. `l=english` keeps dates and labels parseable.

    Raises:
        ValueError: The application identifier is not a positive integer.
    """
    validate_app_id(app_id)
    query = urlencode({"browsefilter": browse_filter, "filterLanguage": language, "l": "english"})
    return f"https://steamcommunity.com/app/{app_id}/reviews/?{query}"


def parse_card(card: WebElement, app_id: int) -> Review | None:
    """Extract one review, keeping browser-dependent parsing in the scraping layer.

    Args:
        card: Steam `.apphub_Card` element using English interface labels.
        app_id: Application identifier to include in the raw review record.

    Returns:
        Review record, or None if required fields are missing, empty, or unparseable.
        Missing helpful-vote labels become `NO_HELPFUL_VOTES`.

    Raises:
        StaleElementReferenceException: The card changed during extraction.
        WebDriverException: The browser could not read the card.
    """
    try:
        date_label = card.find_element(By.CSS_SELECTOR, DATE_SELECTOR).text
        text = strip_review_date(card.find_element(By.CSS_SELECTOR, TEXT_SELECTOR).text, date_label)
        review_date = parse_review_date(date_label)
        hours = strip_hours_label(card.find_element(By.CSS_SELECTOR, HOURS_SELECTOR).text)
        recommendation = card.find_element(By.CSS_SELECTOR, RECOMMENDATION_SELECTOR).text.strip()
    except NoSuchElementException as exc:
        logger.debug("App %s: skipping incomplete review card: %s", app_id, exc.msg)
        return None
    if not text or review_date is None or not hours or not recommendation:
        return None
    try:
        helpful_votes = card.find_element(By.CSS_SELECTOR, HELPFUL_SELECTOR).text.strip()
    except NoSuchElementException:
        helpful_votes = NO_HELPFUL_VOTES
    return Review(app_id, review_date, text, hours, recommendation, helpful_votes)


class ReviewScraper:
    """Collect review records with an existing browser; perform no filesystem writes."""

    def __init__(self, driver: WebDriver, settings: ScraperSettings | None = None) -> None:
        """Initialize the collector.

        Args:
            driver: Open Selenium browser, whose lifecycle belongs to the caller.
            settings: Validated scraping parameters; defaults to `ScraperSettings()`.
        """
        self._driver = driver
        self._settings = settings or ScraperSettings()

    def scrape(self, app_id: int) -> list[Review]:
        """Collect up to the configured number of unique reviews.

        Args:
            app_id: Positive Steam application identifier.

        Returns:
            Review records in page order, deduplicated by all raw fields. A nonempty
            smaller sample is returned when scrolling stops yielding new valid reviews.

        Raises:
            ValueError: The application identifier is invalid.
            ScrapingError: Navigation or browser access failed, stale retries were
                exhausted, or no valid reviews were collected. Partial results are discarded.
        """
        url = review_url(app_id, self._settings.browse_filter, self._settings.language)
        try:
            self._driver.set_page_load_timeout(self._settings.page_load_timeout_s)
            self._driver.get(url)
            reviews = self._collect(app_id)
        except WebDriverException as exc:
            raise ScrapingError(f"App {app_id}: browser failed; no results saved") from exc
        if not reviews:
            raise ScrapingError(f"App {app_id}: no valid reviews found; no results saved")
        return reviews

    def _collect(self, app_id: int) -> list[Review]:
        """Run the bounded collection loop on an already opened review page.

        Args:
            app_id: Application identifier to attach to each extracted review.

        Returns:
            Unique records in page order, limited to the requested sample size.

        Raises:
            ScrapingError: Consecutive stale-element errors reached the retry limit.
            WebDriverException: A browser operation failed for another reason.
        """
        settings = self._settings
        reviews: dict[Review, None] = {}
        stalled = 0
        stale_attempts = 0
        while len(reviews) < settings.target_reviews and stalled < settings.max_stalled_scrolls:
            before = len(reviews)
            try:
                self._scroll_down()
                self._read_cards(app_id, reviews)
            except StaleElementReferenceException as exc:
                stale_attempts += 1
                logger.warning(
                    "App %s: stale page (%d/%d)", app_id, stale_attempts, settings.max_stale_retries
                )
                if stale_attempts >= settings.max_stale_retries:
                    raise ScrapingError(
                        f"App {app_id}: stale-page retry limit reached; no results saved"
                    ) from exc
            else:
                stale_attempts = 0
                stalled = stalled + 1 if len(reviews) == before else 0
            logger.info(
                "App %s: %d/%d reviews collected", app_id, len(reviews), settings.target_reviews
            )
        if len(reviews) < settings.target_reviews:
            logger.warning(
                "App %s: stopped at %d/%d reviews after %d scrolls without new valid reviews",
                app_id,
                len(reviews),
                settings.target_reviews,
                stalled,
            )
        return list(reviews)

    def _read_cards(self, app_id: int, reviews: dict[Review, None]) -> None:
        """Add loaded cards to the ordered collection until the target is reached.

        Args:
            app_id: Application identifier for parsing each card.
            reviews: Mutable ordered set of review records collected so far.

        Raises:
            WebDriverException: Reading the page or a card failed, including stale elements.
        """
        for card in self._driver.find_elements(By.CSS_SELECTOR, CARD_SELECTOR):
            review = parse_card(card, app_id)
            if review is not None:
                reviews.setdefault(review)
            if len(reviews) >= self._settings.target_reviews:
                break

    def _scroll_down(self) -> None:
        """Scroll one viewport and allow the next group of cards to load.

        Raises:
            WebDriverException: The body did not appear before the timeout or scrolling failed.
        """
        body = WebDriverWait(self._driver, self._settings.wait_timeout_s).until(
            ec.presence_of_element_located((By.TAG_NAME, "body"))
        )
        body.send_keys(Keys.PAGE_DOWN)
        time.sleep(self._settings.scroll_pause_s)
