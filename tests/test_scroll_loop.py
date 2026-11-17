"""Test review collection limits and retry behavior with a simulated scrolling page."""

from datetime import date
from typing import cast

import pytest
from selenium.common.exceptions import StaleElementReferenceException, WebDriverException
from selenium.webdriver.remote.webdriver import WebDriver

from steam_review_analysis.schema import Review
from steam_review_analysis.scraping import scraper
from steam_review_analysis.scraping.scraper import ReviewScraper, ScraperSettings, ScrapingError


class FakeBody:
    """Accept scroll keystrokes without interacting with an actual browser."""

    def send_keys(self, *_: str) -> None:
        """Accept and ignore keystrokes sent by the production scrolling code.

        Args:
            *_: Selenium key arguments, unused by this minimal body substitute.
        """
        pass


class FakeDriver:
    """Reveals one more review card per scroll, up to `available`."""

    def __init__(self, available: int) -> None:
        """Create a finite pool of cards and reset navigation and scroll counters.

        Args:
            available: Total number of distinct cards the simulated page can reveal.
        """
        self.cards = [object() for _ in range(available)]
        self.scrolls = 0
        self.visited: list[str] = []

    def get(self, url: str) -> None:
        """Record navigation without making a network request.

        Args:
            url: Requested review page URL to append to the visited-page history.
        """
        self.visited.append(url)

    def set_page_load_timeout(self, seconds: float) -> None:
        """Record the navigation timeout configured by the scraper.

        Args:
            seconds: Requested timeout duration in seconds.
        """
        self.page_load_timeout = seconds

    def find_element(self, *_: str) -> FakeBody:
        """Advance one simulated scroll when the scraper looks up the page body.

        Args:
            *_: Selenium locator arguments, accepted and otherwise ignored.

        Returns:
            Fake body that accepts the subsequent scroll keystroke.
        """
        self.scrolls += 1
        return FakeBody()

    def find_elements(self, *_: str) -> list[object]:
        """Return all cards revealed by the simulated scrolls so far.

        Args:
            *_: Selenium locator arguments, accepted and otherwise ignored.

        Returns:
            Visible card objects, bounded by the page's available card pool.
        """
        return self.cards[: self.scrolls]


def scrape(
    monkeypatch: pytest.MonkeyPatch, available: int, target: int
) -> tuple[list[Review], FakeDriver]:
    """Run the real collection loop against a finite fake page without sleep delays.

    Args:
        monkeypatch: Pytest fixture used to replace DOM parsing with deterministic records.
        available: Number of distinct reviews available from the simulated page.
        target: Maximum number of reviews requested from the scraper.

    Returns:
        Collected review records and the fake driver exposing its scroll/navigation history.

    Raises:
        ScrapingError: The simulated page yields no valid reviews.
    """

    def parse_card(card: object, app_id: int) -> Review:
        """Convert an opaque fake card into a distinct review record.

        Args:
            card: Object whose identity distinguishes the simulated review from other cards.
            app_id: Application ID attached to the resulting review.

        Returns:
            Deterministic record for this card with fixed date, playtime, and recommendation.
        """
        return Review(app_id, date(2024, 1, 1), f"review {id(card)}", "1", "Recommended", "0")

    monkeypatch.setattr(scraper, "parse_card", parse_card)
    driver = FakeDriver(available)
    settings = ScraperSettings(target_reviews=target, scroll_pause_s=0, max_stalled_scrolls=3)
    reviews = ReviewScraper(cast(WebDriver, driver), settings).scrape(app_id=42)
    return reviews, driver


def test_stops_at_target_without_duplicates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop at the requested sample size while repeated visible cards remain deduplicated."""
    reviews, driver = scrape(monkeypatch, available=100, target=5)

    assert len(reviews) == 5
    assert len(set(reviews)) == 5
    assert "/app/42/reviews/" in driver.visited[0]


def test_gives_up_when_page_stops_loading_reviews(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return a smaller sample after the configured consecutive no-progress scroll limit."""
    reviews, driver = scrape(monkeypatch, available=4, target=600)

    assert len(reviews) == 4
    assert driver.scrolls == 4 + 3  # four productive scrolls, then three stalled ones


def test_empty_page_is_a_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Raise a scraping error when bounded scrolling finds no reviews."""
    with pytest.raises(ScrapingError, match="no valid reviews"):
        scrape(monkeypatch, available=0, target=5)


def test_repeated_stale_errors_have_a_finite_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop persistent stale-element failures at the configured retry limit."""

    def stale_cards(*_: object) -> list[object]:
        """Simulate a page whose card lookup is stale on every attempt.

        Args:
            *_: Unused Selenium locator arguments.

        Raises:
            StaleElementReferenceException: Always raised to exercise the retry limit.
        """
        raise StaleElementReferenceException("page changed")

    driver = FakeDriver(available=1)
    monkeypatch.setattr(driver, "find_elements", stale_cards)
    settings = ScraperSettings(scroll_pause_s=0, max_stale_retries=3)
    with pytest.raises(ScrapingError, match="retry limit"):
        ReviewScraper(cast(WebDriver, driver), settings).scrape(42)
    assert driver.scrolls == settings.max_stale_retries


def test_navigation_failure_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    """Raise a scraping error immediately when navigation fails, before any scrolling."""

    def fail_navigation(url: str) -> None:
        """Simulate a failed browser navigation request.

        Args:
            url: Requested page URL, accepted without contacting the network.

        Raises:
            WebDriverException: Always raised to simulate navigation failure.
        """
        raise WebDriverException("navigation failed")

    driver = FakeDriver(available=1)
    monkeypatch.setattr(driver, "get", fail_navigation)
    with pytest.raises(ScrapingError, match="browser failed"):
        ReviewScraper(cast(WebDriver, driver)).scrape(42)
    assert driver.scrolls == 0


def test_transient_stale_errors_reset_after_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset the consecutive stale counter after a successful page read."""
    calls = 0

    def intermittent_cards(*_: object) -> list[object]:
        """Alternate two stale reads with a successful read that reveals more cards.

        Args:
            *_: Unused Selenium locator arguments.

        Returns:
            One card on the first successful read and both cards on the next successful read.

        Raises:
            StaleElementReferenceException: Raised on the two reads preceding each success.
        """
        nonlocal calls
        calls += 1
        if calls in {1, 2, 4, 5}:
            raise StaleElementReferenceException("page changed")
        return driver.cards[:1] if calls == 3 else driver.cards

    def parse_card(card: object, app_id: int) -> Review:
        """Create stable review records so repeated cards are recognized as duplicates.

        Args:
            card: Object identity used as the simulated review's distinguishing text.
            app_id: Application ID attached to the record.

        Returns:
            Review with card-specific text and fixed metadata.
        """
        return Review(app_id, date(2024, 1, 1), f"review {id(card)}", "1", "Recommended", "0")

    driver = FakeDriver(available=2)
    monkeypatch.setattr(driver, "find_elements", intermittent_cards)
    monkeypatch.setattr(scraper, "parse_card", parse_card)
    settings = ScraperSettings(target_reviews=2, scroll_pause_s=0, max_stale_retries=3)
    assert len(ReviewScraper(cast(WebDriver, driver), settings).scrape(42)) == 2
    assert calls == 6
