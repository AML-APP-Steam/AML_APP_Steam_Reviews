"""Test review-card extraction with lightweight DOM substitutes and no browser."""

from datetime import date
from typing import cast

from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.remote.webelement import WebElement

from steam_review_analysis.schema import NO_HELPFUL_VOTES, Review
from steam_review_analysis.scraping.scraper import parse_card, review_url


class FakeElement:
    """Represent the text and selectable children needed by the card parser."""

    def __init__(self, text: str = "", children: dict[str, "FakeElement"] | None = None) -> None:
        """Create a minimal element with optional children indexed by CSS selector.

        Args:
            text: Displayed element text returned by the fake's text attribute.
            children: Mapping from selectors to fake child elements; defaults to no children.
        """
        self.text = text
        self._children = children or {}

    def find_element(self, by: str, value: str) -> "FakeElement":
        """Look up a child using Selenium-compatible lookup arguments.

        Args:
            by: Locator strategy; accepted for compatibility and otherwise ignored.
            value: CSS selector used as a key in the configured children mapping.

        Returns:
            Fake child element associated with the requested selector.

        Raises:
            NoSuchElementException: No child was configured for the selector.
        """
        try:
            return self._children[value]
        except KeyError:
            raise NoSuchElementException(value) from None


def make_card(**overrides: str | None) -> WebElement:
    """Construct a complete review card with optional field replacements or omissions.

    Args:
        **overrides: CSS selectors mapped to replacement text; None removes that field.

    Returns:
        Fake card cast to WebElement for the production parser's typed interface.
    """
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
    """Include the game ID and content filters while requesting English interface labels."""
    assert review_url(105600, "toprated", "english") == (
        "https://steamcommunity.com/app/105600/reviews/"
        "?browsefilter=toprated&filterLanguage=english&l=english"
    )


def test_parse_card_extracts_all_fields() -> None:
    """Extract every raw field and exclude the posting label from the review body."""
    assert parse_card(make_card(), app_id=105600) == Review(
        app_id=105600,
        date=date(2024, 5, 21),
        text="Great game. Buy it.",
        hours_played="12.5",
        recommendation="Recommended",
        helpful_votes="3 people found this review helpful",
    )


def test_parse_card_without_helpful_votes_uses_placeholder() -> None:
    """Keep a valid review when only its optional helpful-vote label is absent."""
    review = parse_card(make_card(**{".found_helpful": None}), app_id=1)
    assert review is not None
    assert review.helpful_votes == NO_HELPFUL_VOTES


def test_parse_card_skips_missing_required_field() -> None:
    """Skip a card whose required playtime element is missing."""
    assert parse_card(make_card(**{".hours": None}), app_id=1) is None


def test_parse_card_skips_unparsable_date() -> None:
    """Skip a card with a posting date the parser cannot interpret."""
    assert parse_card(make_card(**{".date_posted": "Posted: yesterday"}), app_id=1) is None


def test_parse_card_skips_empty_text() -> None:
    """Skip a card containing only a posting label and no review body."""
    assert parse_card(make_card(**{".apphub_CardTextContent": "Posted: May 21, 2024"}), 1) is None


def test_parse_card_preserves_first_line_without_date_in_body() -> None:
    """Keep the first review sentence when the text block contains no posting label."""
    review = parse_card(make_card(**{".apphub_CardTextContent": "Great game.\nBuy it."}), 1)
    assert review is not None
    assert review.text == "Great game. Buy it."


def test_parse_card_accepts_current_year_date() -> None:
    """Accept a yearless posting label and assign the current calendar year."""
    card = make_card(
        **{
            ".date_posted": "Posted: 21 May",
            ".apphub_CardTextContent": "Posted: 21 May\nGreat game.",
        }
    )
    review = parse_card(card, 1)
    assert review is not None
    assert review.date == date(date.today().year, 5, 21)
