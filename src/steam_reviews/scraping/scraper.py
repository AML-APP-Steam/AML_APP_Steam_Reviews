"""Scroll a game's Steam review page and extract reviews from the loaded cards."""

import logging
import time
from dataclasses import dataclass

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
from steam_reviews.scraping.parsers import parse_review_date, strip_hours_label

logger = logging.getLogger(__name__)

CARD_SELECTOR = ".apphub_Card"


@dataclass(frozen=True, slots=True)
class ScraperSettings:
    """Tunable scraping behaviour.

    Attributes:
        target_reviews: Stop once this many unique reviews are collected.
        browse_filter: Steam's `browsefilter` query value, e.g. `"toprated"`.
        language: Steam's `filterLanguage` query value.
        scroll_pause_s: Seconds to wait after each scroll for new reviews to load.
        wait_timeout_s: Seconds to wait for the page body to be present.
        max_stalled_scrolls: Give up after this many scrolls in a row with no new reviews.
    """

    target_reviews: int = 600
    browse_filter: str = "toprated"
    language: str = "english"
    scroll_pause_s: float = 2.0
    wait_timeout_s: float = 5.0
    max_stalled_scrolls: int = 5


def review_url(app_id: int, browse_filter: str, language: str) -> str:
    """Build the community reviews URL for a game.

    Args:
        app_id: Steam app id of the game.
        browse_filter: Steam's `browsefilter` value, e.g. `"toprated"`.
        language: Steam's `filterLanguage` value, e.g. `"english"`.

    Returns:
        URL of the game's reviews page.
    """
    return (
        f"https://steamcommunity.com/app/{app_id}/reviews/"
        f"?browsefilter={browse_filter}&filterLanguage={language}"
    )


def parse_card(card: WebElement, app_id: int) -> Review | None:
    """Extract a review from one card.

    Args:
        card: A `.apphub_Card` element on the reviews page.
        app_id: App id to store on the resulting review.

    Returns:
        The review, or None if a required field is missing or unparsable. Missing helpful
        votes are not an error; they become `NO_HELPFUL_VOTES`.
    """
    try:
        # The first line of the text block is the "Posted: ..." date, not part of the review.
        lines = card.find_element(By.CSS_SELECTOR, ".apphub_CardTextContent").text.split("\n")
        text = " ".join(lines[1:]).strip()
        review_date = parse_review_date(card.find_element(By.CSS_SELECTOR, ".date_posted").text)
        hours = strip_hours_label(card.find_element(By.CSS_SELECTOR, ".hours").text)
        recommendation = card.find_element(By.CSS_SELECTOR, ".title").text.strip()
    except NoSuchElementException as exc:
        logger.debug("Skipping a review card: %s", exc.msg)
        return None

    if not text or review_date is None:
        return None

    try:
        helpful_votes = card.find_element(By.CSS_SELECTOR, ".found_helpful").text.strip()
    except NoSuchElementException:
        helpful_votes = NO_HELPFUL_VOTES

    return Review(app_id, review_date, text, hours, recommendation, helpful_votes)


class ReviewScraper:
    """Collects reviews for a game by scrolling its Steam review page."""

    def __init__(self, driver: WebDriver, settings: ScraperSettings | None = None) -> None:
        """Create a scraper.

        Args:
            driver: An open Selenium driver, e.g. from `driver.chrome_driver`.
            settings: Scraping options; defaults to `ScraperSettings()`.
        """
        self._driver = driver
        self._settings = settings or ScraperSettings()

    def scrape(self, app_id: int) -> list[Review]:
        """Scrape up to `target_reviews` unique reviews for one game.

        Stops early if the page stops loading new reviews or the browser raises an error.

        Args:
            app_id: Steam app id of the game.

        Returns:
            Reviews in page order, without duplicates.
        """
        settings = self._settings
        self._driver.get(review_url(app_id, settings.browse_filter, settings.language))

        reviews: dict[Review, None] = {}  # insertion-ordered set: dedupes while keeping order
        stalled = 0
        while len(reviews) < settings.target_reviews and stalled < settings.max_stalled_scrolls:
            before = len(reviews)
            try:
                self._scroll_down()
                for card in self._driver.find_elements(By.CSS_SELECTOR, CARD_SELECTOR):
                    review = parse_card(card, app_id)
                    if review is not None:
                        reviews.setdefault(review)
            except StaleElementReferenceException:
                logger.warning("Stale element, retrying")
                time.sleep(settings.scroll_pause_s)
                continue
            except WebDriverException:
                logger.exception("Browser error while scraping app %s, stopping early", app_id)
                break

            stalled = stalled + 1 if len(reviews) == before else 0
            logger.info("App %s: %d reviews collected", app_id, len(reviews))

        if stalled >= settings.max_stalled_scrolls:
            logger.warning("App %s: no new reviews after %d scrolls", app_id, stalled)
        return list(reviews)[: settings.target_reviews]

    def _scroll_down(self) -> None:
        """Scroll the page down once and wait for new reviews to load."""
        body = WebDriverWait(self._driver, self._settings.wait_timeout_s).until(
            ec.presence_of_element_located((By.TAG_NAME, "body"))
        )
        body.send_keys(Keys.PAGE_DOWN)
        time.sleep(self._settings.scroll_pause_s)
