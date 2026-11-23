"""Test locale-independent date parsing and removal of Steam display labels."""

from datetime import date

import pytest

from steam_review_analysis.scraping.parsers import (
    parse_review_date,
    strip_hours_label,
    strip_review_date,
)


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
    """Parse supported English dates, including May which the original parser rejected."""
    assert parse_review_date(raw) == expected


@pytest.mark.parametrize("raw", ["", "Posted: yesterday", "21/05/2024"])
def test_parse_review_date_unsupported_format(raw: str) -> None:
    """Return None for unsupported date labels instead of guessing their meaning."""
    assert parse_review_date(raw) is None


def test_strip_hours_label() -> None:
    """Remove the hours label while preserving the decimal and thousands separators."""
    assert strip_hours_label("1,234.5 hrs on record") == "1,234.5"


@pytest.mark.parametrize("raw", ["Posted: October 20", "Posted: 20 October", "20 Oct", "Oct 20"])
def test_yearless_dates_use_reference_year(raw: str) -> None:
    """Use the supplied reference year for both English month orders and abbreviations."""
    assert parse_review_date(raw, reference_date=date(2026, 10, 21)) == date(2026, 10, 20)


@pytest.mark.parametrize("raw", ["Posted: 21 May, 2024", "May 21 2024", "21 May 2024"])
def test_alternative_english_date_orders(raw: str) -> None:
    """Accept full-year dates with either month order and an optional comma."""
    assert parse_review_date(raw) == date(2024, 5, 21)


@pytest.mark.parametrize(
    "raw", ["February 30, 2024", "May 0", "May 32", "Unknown 20", "May 21, 0000"]
)
def test_invalid_calendar_dates_are_rejected(raw: str) -> None:
    """Reject impossible calendar dates and unknown month names."""
    assert parse_review_date(raw, reference_date=date(2026, 1, 1)) is None


def test_yearless_leap_day_uses_supplied_year() -> None:
    """Accept February 29 only when the supplied reference year is a leap year."""
    assert parse_review_date("29 Feb", reference_date=date(2024, 3, 1)) == date(2024, 2, 29)
    assert parse_review_date("29 Feb", reference_date=date(2025, 3, 1)) is None


def test_posting_label_is_only_removed_from_first_line() -> None:
    """Remove only a leading date label and retain matching text inside the review body."""
    # Arrange
    label = "Posted: May 21, 2024"

    # Act & Assert
    assert (
        strip_review_date(f"{label}\n First line\nSecond line ", label) == "First line Second line"
    )
    assert strip_review_date("First line\nSecond line", label) == "First line Second line"
    assert strip_review_date(f"Story\n{label}", label) == f"Story {label}"
