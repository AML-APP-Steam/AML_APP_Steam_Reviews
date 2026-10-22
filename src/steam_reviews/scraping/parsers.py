"""Parse Steam review text without a browser or process-wide locale changes."""

import logging
import re
from datetime import date

logger = logging.getLogger(__name__)

_DATE_PATTERNS = (
    re.compile(r"(?P<month>[A-Za-z]+)\s+(?P<day>\d{1,2})(?:,?\s+(?P<year>\d{4}))?"),
    re.compile(r"(?P<day>\d{1,2})\s+(?P<month>[A-Za-z]+)(?:,?\s+(?P<year>\d{4}))?"),
)
_MONTH_NAMES = (
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
)

_MONTH_NUMBERS = {
    label: number for number, name in enumerate(_MONTH_NAMES, start=1) for label in (name, name[:3])
}


def parse_review_date(raw: str, *, reference_date: date | None = None) -> date | None:
    """Parse an English Steam posting date, including dates without a year.

    Args:
        raw: Posting label, e.g. `Posted: May 21, 2024` or `Posted: 21 May`.
        reference_date: Date whose year is used when Steam omits it; defaults to today.
            Supply this when parsing archived pages or writing reproducible tests.

    Returns:
        Parsed date, or None for an unsupported label or an invalid calendar date.
        Both full and three-letter English month names are accepted.
    """
    cleaned = raw.strip().removeprefix("Posted:").strip()
    for pattern in _DATE_PATTERNS:
        match = pattern.fullmatch(cleaned)
        if match is None:
            continue
        month = _MONTH_NUMBERS.get(match["month"].lower())
        if month is None:
            break
        year = int(match["year"]) if match["year"] else (reference_date or date.today()).year
        try:
            return date(year, month, int(match["day"]))
        except ValueError:
            break
    logger.debug("Unsupported review date: %r", raw)
    return None


def strip_review_date(raw: str, date_label: str) -> str:
    """Remove the posting label from the start of a review and collapse whitespace.

    Args:
        raw: Text of the complete review card's text block.
        date_label: Posting label obtained from the card's separate date element.

    Returns:
        Review body on one line. If the label is absent, the first review line is kept.
    """
    lines = raw.strip().splitlines()
    if lines and lines[0].strip() == date_label.strip():
        lines = lines[1:]
    return " ".join(" ".join(lines).split())


def strip_hours_label(raw: str) -> str:
    """Remove the English `hrs on record` suffix from the hours text.

    Args:
        raw: Hours text such as `1,234.5 hrs on record`.

    Returns:
        Number text with surrounding whitespace removed, preserving decimal separators.
    """
    return raw.strip().removesuffix("hrs on record").strip()
