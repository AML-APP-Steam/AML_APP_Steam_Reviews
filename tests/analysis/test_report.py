"""Test the end-to-end analysis report workflow."""

from pathlib import Path

import pandas as pd
import pytest

from steam_reviews.analysis.report import run_analysis
from steam_reviews.io import write_processed


def test_run_analysis_writes_tables_without_plots(processed_dir: Path, tmp_path: Path) -> None:
    """Produce documents, sentiment, frequency and correlation artifacts."""
    output = tmp_path / "analysis"

    report = run_analysis(
        "Alpha",
        "2023-09-29",
        output_dir=output,
        save_plots=False,
        top_n=5,
        processed_dir=processed_dir,
    )

    assert report.game == "Alpha"
    assert report.app_id == 1
    assert len(report.prepared.reviews) == 5
    assert report.figures == {}
    assert set(report.frequencies) == {"overall", "before", "after"}
    assert len(report.frequencies["overall"]) <= 5
    assert {"sentiment_score", "sentiment_category", "year", "period"}.issubset(
        report.sentiment.columns
    )
    for path in report.tables.values():
        assert path.exists()
        assert path.stat().st_size > 0
    assert (output / "documents.csv").exists()
    assert (output / "metadata.csv").exists()
    assert (output / "summary.json").exists()


def test_run_analysis_writes_figures_when_requested(processed_dir: Path, tmp_path: Path) -> None:
    """Write word clouds, frequency charts and sentiment plots."""
    output = tmp_path / "figures"

    report = run_analysis(
        "Alpha",
        "2023-09-29",
        output_dir=output,
        save_plots=True,
        processed_dir=processed_dir,
    )

    assert "wordcloud_overall" in report.figures
    assert "word_frequencies_overall" in report.figures
    assert "sentiment_distribution" in report.figures
    assert "sentiment_categories" in report.figures
    assert "sentiment_yearly" in report.figures
    assert "sentiment_boxplot" in report.figures
    for path in report.figures.values():
        assert path.exists()
        assert path.stat().st_size > 0


def test_run_analysis_rejects_empty_preprocessed_corpus(tmp_path: Path) -> None:
    """Fail when every review becomes empty after preprocessing."""
    frame = pd.DataFrame(
        [
            [1, "Alpha", "2024-01-01", "the and of", 1.0, "Recommended", 0, True],
        ],
        columns=[
            "app_id",
            "game",
            "date",
            "text",
            "hours_played",
            "recommendation",
            "helpful_votes",
            "is_recommended",
        ],
    )
    write_processed(frame, tmp_path)

    with pytest.raises(ValueError, match="No reviews remain"):
        run_analysis("Alpha", "2024-01-01", processed_dir=tmp_path, save_plots=False)
