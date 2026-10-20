"""Pure text parsers for strings found on a review card (no browser needed)."""

import logging
from datetime import date, datetime

logger = logging.getLogger(__name__)

_DATE_FORMAT = "%B %d, %Y"  # e.g. "May 21, 2024"


def parse_review_date(raw: str) -> date | None:
    """Parse the date shown on a review card.

    Args:
        raw: Date text such as `"Posted: May 21, 2024"`.

    Returns:
        The date, or None if the text is not in `Month D, YYYY` form (for example, Steam
        shows reviews from the current year without a year).
    """
    cleaned = raw.replace("Posted:", "").strip()
    try:
        return datetime.strptime(cleaned, _DATE_FORMAT).date()
    except ValueError:
        logger.debug("Unsupported date format: %r", raw)
        return None


def strip_hours_label(raw: str) -> str:
    """Remove the `hrs on record` suffix from the hours text.

    Args:
        raw: Hours text such as `"12.5 hrs on record"`.

    Returns:
        The number part only, e.g. `"12.5"`.
    """
    return raw.replace("hrs on record", "").strip()
