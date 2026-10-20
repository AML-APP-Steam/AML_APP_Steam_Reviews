from datetime import date
from typing import cast

import pytest
from selenium.webdriver.remote.webdriver import WebDriver

from steam_reviews.schema import Review
from steam_reviews.scraping import scraper
from steam_reviews.scraping.scraper import ReviewScraper, ScraperSettings


class FakeBody:
    def send_keys(self, *_: str) -> None:
        pass


class FakeDriver:
    """Reveals one more review card per scroll, up to `available`."""

    def __init__(self, available: int) -> None:
        self.cards = [object() for _ in range(available)]
        self.scrolls = 0
        self.visited: list[str] = []

    def get(self, url: str) -> None:
        self.visited.append(url)

    def find_element(self, *_: str) -> FakeBody:
        self.scrolls += 1
        return FakeBody()

    def find_elements(self, *_: str) -> list[object]:
        return self.cards[: self.scrolls]


def scrape(
    monkeypatch: pytest.MonkeyPatch, available: int, target: int
) -> tuple[list[Review], FakeDriver]:
    def parse_card(card: object, app_id: int) -> Review:
        return Review(app_id, date(2024, 1, 1), f"review {id(card)}", "1", "Recommended", "0")

    monkeypatch.setattr(scraper, "parse_card", parse_card)
    driver = FakeDriver(available)
    settings = ScraperSettings(target_reviews=target, scroll_pause_s=0, max_stalled_scrolls=3)
    reviews = ReviewScraper(cast(WebDriver, driver), settings).scrape(app_id=42)
    return reviews, driver


def test_stops_at_target_without_duplicates(monkeypatch: pytest.MonkeyPatch) -> None:
    reviews, driver = scrape(monkeypatch, available=100, target=5)

    assert len(reviews) == 5
    assert len(set(reviews)) == 5
    assert "/app/42/reviews/" in driver.visited[0]


def test_gives_up_when_page_stops_loading_reviews(monkeypatch: pytest.MonkeyPatch) -> None:
    reviews, driver = scrape(monkeypatch, available=4, target=600)

    assert len(reviews) == 4
    assert driver.scrolls == 4 + 3  # four productive scrolls, then three stalled ones
