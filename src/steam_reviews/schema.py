"""Shared data contract: the `Review` record and the raw CSV column names."""

from dataclasses import dataclass
from datetime import date

# Column names of the raw CSVs. Kept identical to the files scraped before the refactor.
RAW_COLUMNS = ("Game", "Date", "Review", "Hours Played", "Recommendation", "Helpful Votes")

RAW_TO_CLEAN_COLUMNS = {
    "Game": "app_id",
    "Date": "date",
    "Review": "text",
    "Hours Played": "hours_played",
    "Recommendation": "recommendation",
    "Helpful Votes": "helpful_votes",
}

NO_HELPFUL_VOTES = "No helpful votes"


@dataclass(frozen=True, slots=True)
class Review:
    """One review exactly as shown on the page; numeric parsing happens in `processing`.

    Attributes:
        app_id: Steam app id of the reviewed game.
        date: Day the review was posted.
        text: Review body with line breaks collapsed to spaces.
        hours_played: Hours on record as shown, e.g. `"1,234.5"` (may contain a comma).
        recommendation: `"Recommended"` or `"Not Recommended"`.
        helpful_votes: Helpful-vote label as shown, or `NO_HELPFUL_VOTES` if absent.
    """

    app_id: int
    date: date
    text: str
    hours_played: str
    recommendation: str
    helpful_votes: str

    def as_row(self) -> tuple[int, str, str, str, str, str]:
        """Convert the review to a raw CSV row.

        Returns:
            Fields in `RAW_COLUMNS` order, with the date as an ISO string.
        """
        return (
            self.app_id,
            self.date.isoformat(),
            self.text,
            self.hours_played,
            self.recommendation,
            self.helpful_votes,
        )
