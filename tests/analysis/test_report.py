"""Test the end-to-end analysis report workflow."""

import json
from pathlib import Path

import pandas as pd
import pytest

from steam_review_analysis.analysis.report import run_analysis
from steam_review_analysis.io import write_processed

pytestmark = pytest.mark.usefixtures("nltk_corpora")


def test_run_analysis_writes_tables_without_plots(processed_dir: Path, tmp_path: Path) -> None:
    """Produce documents, sentiment, frequency and correlation artifacts."""
    # Arrange
    output = tmp_path / "analysis"

    # Act
    report = run_analysis(
        "Alpha",
        "2023-09-29",
        output_dir=output,
        save_plots=False,
        top_n=5,
        processed_dir=processed_dir,
    )

    # Assert
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
    # Arrange
    output = tmp_path / "figures"

    # Act
    report = run_analysis(
        "Alpha",
        "2023-09-29",
        output_dir=output,
        save_plots=True,
        processed_dir=processed_dir,
    )

    # Assert
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
    # Arrange
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

    # Act & Assert
    with pytest.raises(ValueError, match="No reviews remain"):
        run_analysis("Alpha", "2024-01-01", processed_dir=tmp_path, save_plots=False)


def test_run_analysis_defaults_output_beside_processed_data(
    processed_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Write artifacts under `<processed_dir>/analysis/<game>` when no output is given."""
    # Arrange
    monkeypatch.chdir(tmp_path)

    # Act
    report = run_analysis("Alpha", "2024-01-01", processed_dir=processed_dir, save_plots=False)

    # Assert
    assert report.output_dir == processed_dir / "analysis" / "alpha"
    assert (report.output_dir / "summary.json").is_file()


def test_run_analysis_writes_null_for_empty_period(processed_dir: Path, tmp_path: Path) -> None:
    """Write empty-period statistics as JSON null rather than the non-standard NaN."""
    # Act
    report = run_analysis(
        "Alpha", "2030-01-01", processed_dir=processed_dir, output_dir=tmp_path, save_plots=False
    )

    # Assert
    text = (report.output_dir / "summary.json").read_text(encoding="utf-8")
    assert "NaN" not in text
    summary = json.loads(text)
    assert summary["after"]["count"] == 0
    assert summary["after"]["mean_score"] is None
