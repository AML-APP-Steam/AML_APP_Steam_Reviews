from datetime import date

import pytest

from steam_reviews.scraping.parsers import parse_review_date, strip_hours_label


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Posted: July 15, 2023", date(2023, 7, 15)),
        ("December 25, 2018", date(2018, 12, 25)),
        # Regression: three-letter months used to be rejected by a length check.
        ("Posted: May 21, 2024", date(2024, 5, 21)),
    ],
)
def test_parse_review_date(raw: str, expected: date) -> None:
    assert parse_review_date(raw) == expected


@pytest.mark.parametrize("raw", ["", "Posted: yesterday", "21/05/2024"])
def test_parse_review_date_unsupported_format(raw: str) -> None:
    assert parse_review_date(raw) is None


def test_strip_hours_label() -> None:
    assert strip_hours_label("1,234.5 hrs on record") == "1,234.5"
