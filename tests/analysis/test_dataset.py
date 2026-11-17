"""Test game selection, processed-data loading and period boundaries."""

from pathlib import Path

import pandas as pd
import pytest

from steam_review_analysis.analysis.dataset import (
    add_period,
    add_year,
    default_output_dir,
    game_name,
    load_reviews,
    period_frames,
    resolve_game,
    slugify,
    split_by_cutoff,
)


def test_resolve_game_by_id_name_and_numeric_string(sample_frame: pd.DataFrame) -> None:
    """Resolve the same game from an ID, a numeric string and a case-insensitive name."""
    assert resolve_game(sample_frame, 1) == 1
    assert resolve_game(sample_frame, "1") == 1
    assert resolve_game(sample_frame, "alpha") == 1


def test_resolve_game_reports_available_games(sample_frame: pd.DataFrame) -> None:
    """Include available games when a selector does not match."""
    with pytest.raises(ValueError, match="Alpha \\(1\\)"):
        resolve_game(sample_frame, "Missing")


def test_resolve_game_rejects_empty_frame() -> None:
    """Reject selection when no processed reviews exist."""
    with pytest.raises(ValueError, match="No processed reviews"):
        resolve_game(pd.DataFrame(), "Alpha")


def test_load_reviews_filters_game_and_sorts_by_date(processed_dir: Path) -> None:
    """Load one game's reviews in ascending date order."""
    reviews = load_reviews(1, processed_dir)

    assert set(reviews["app_id"]) == {1}
    assert reviews["date"].is_monotonic_increasing
    assert list(reviews.index) == list(range(len(reviews)))


def test_load_reviews_rejects_unknown_game(processed_dir: Path) -> None:
    """Report an unknown game selector with the available games."""
    with pytest.raises(ValueError, match="available games"):
        load_reviews(999, processed_dir)


def test_game_name_prefers_display_name(sample_frame: pd.DataFrame) -> None:
    """Return the game's display name from the processed data."""
    assert game_name(sample_frame) == "Alpha"


def test_game_name_falls_back_to_app_id(sample_frame: pd.DataFrame) -> None:
    """Use the application ID when the game column is unavailable."""
    assert game_name(sample_frame.drop(columns=["game"])) == "1"


def test_game_name_rejects_empty_frame() -> None:
    """Reject naming an empty frame."""
    with pytest.raises(ValueError, match="empty"):
        game_name(pd.DataFrame())


def test_slugify_and_default_output_dir() -> None:
    """Build filesystem-safe analysis directory names."""
    assert slugify("PUBG: BATTLEGROUNDS") == "pubg_battlegrounds"
    assert default_output_dir("The Sims 4", Path("root")) == Path("root/the_sims_4")


def test_split_by_cutoff_includes_boundary_in_after(sample_frame: pd.DataFrame) -> None:
    """Place reviews on the cutoff date in the later period."""
    reviews = sample_frame.assign(date=pd.to_datetime(sample_frame["date"]))
    before, after = split_by_cutoff(reviews, "2023-09-29")

    assert "2023-09-29" not in before["date"].dt.strftime("%Y-%m-%d").tolist()
    assert "2023-09-29" in after["date"].dt.strftime("%Y-%m-%d").tolist()
    assert (after["date"] >= pd.Timestamp("2023-09-29")).all()
    assert (before["date"] < pd.Timestamp("2023-09-29")).all()


def test_split_by_cutoff_rejects_invalid_date(sample_frame: pd.DataFrame) -> None:
    """Reject an unparseable cutoff."""
    reviews = sample_frame.assign(date=pd.to_datetime(sample_frame["date"]))
    with pytest.raises(ValueError, match="Invalid cutoff"):
        split_by_cutoff(reviews, "not-a-date")


def test_add_year_inserts_integer_column(sample_frame: pd.DataFrame) -> None:
    """Derive the calendar year after the date column."""
    reviews = sample_frame.assign(date=pd.to_datetime(sample_frame["date"]))
    result = add_year(reviews)

    assert result["year"].tolist() == reviews["date"].dt.year.tolist()
    assert str(result["year"].dtype) == "int64"


def test_add_year_rejects_missing_dates(sample_frame: pd.DataFrame) -> None:
    """Reject frames whose dates contain missing values."""
    reviews = sample_frame.assign(date=pd.to_datetime(sample_frame["date"]))
    reviews.loc[0, "date"] = pd.NaT
    with pytest.raises(ValueError, match="missing"):
        add_year(reviews)


def test_add_period_labels_boundary_as_after(sample_frame: pd.DataFrame) -> None:
    """Label each review before or after the cutoff."""
    reviews = sample_frame.assign(date=pd.to_datetime(sample_frame["date"]))
    result = add_period(reviews, "2023-09-29")

    boundary_rows = result.loc[result["date"] == pd.Timestamp("2023-09-29"), "period"]
    assert boundary_rows.tolist() == ["After"]
    assert set(result["period"]) == {"Before", "After"}


def test_period_frames_returns_before_and_after(sample_frame: pd.DataFrame) -> None:
    """Return named before and after frames."""
    reviews = sample_frame.assign(date=pd.to_datetime(sample_frame["date"]))
    frames = period_frames(reviews, "2023-09-29")

    assert set(frames) == {"before", "after"}
    assert len(frames["before"]) + len(frames["after"]) == len(reviews)
