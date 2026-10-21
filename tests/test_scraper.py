from datetime import date
from typing import cast

from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.remote.webelement import WebElement

from steam_reviews.schema import NO_HELPFUL_VOTES, Review
from steam_reviews.scraping.scraper import parse_card, review_url


class FakeElement:
    def __init__(self, text: str = "", children: dict[str, "FakeElement"] | None = None) -> None:
        self.text = text
        self._children = children or {}

    def find_element(self, by: str, value: str) -> "FakeElement":
        try:
            return self._children[value]
        except KeyError:
            raise NoSuchElementException(value) from None


def make_card(**overrides: str | None) -> WebElement:
    fields: dict[str, str | None] = {
        ".apphub_CardTextContent": "Posted: May 21, 2024\nGreat game.\nBuy it.",
        ".date_posted": "Posted: May 21, 2024",
        ".hours": "12.5 hrs on record",
        ".title": "Recommended",
        ".found_helpful": "3 people found this review helpful",
    }
    fields.update(overrides)
    children = {sel: FakeElement(text) for sel, text in fields.items() if text is not None}
    return cast(WebElement, FakeElement(children=children))


def test_review_url() -> None:
    assert review_url(105600, "toprated", "english") == (
        "https://steamcommunity.com/app/105600/reviews/?browsefilter=toprated&filterLanguage=english"
    )


def test_parse_card_extracts_all_fields() -> None:
    assert parse_card(make_card(), app_id=105600) == Review(
        app_id=105600,
        date=date(2024, 5, 21),
        text="Great game. Buy it.",
        hours_played="12.5",
        recommendation="Recommended",
        helpful_votes="3 people found this review helpful",
    )


def test_parse_card_without_helpful_votes_uses_placeholder() -> None:
    review = parse_card(make_card(**{".found_helpful": None}), app_id=1)
    assert review is not None
    assert review.helpful_votes == NO_HELPFUL_VOTES


def test_parse_card_skips_missing_required_field() -> None:
    assert parse_card(make_card(**{".hours": None}), app_id=1) is None


def test_parse_card_skips_unparsable_date() -> None:
    assert parse_card(make_card(**{".date_posted": "Posted: yesterday"}), app_id=1) is None


def test_parse_card_skips_empty_text() -> None:
    assert parse_card(make_card(**{".apphub_CardTextContent": "Posted: May 21, 2024"}), 1) is None
