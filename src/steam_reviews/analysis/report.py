"""Run the complete review analysis workflow and persist its artifacts."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

from steam_reviews.analysis.artifacts import save_documents, save_summary, save_table
from steam_reviews.analysis.dataset import (
    add_period,
    add_year,
    default_output_dir,
    game_name,
    load_reviews,
    parse_cutoff,
    split_by_cutoff,
)
from steam_reviews.analysis.plotting import (
    plot_boxplot,
    plot_distribution_fits,
    plot_engagement_boxplot,
    plot_scatter,
    plot_sentiment_categories,
    plot_sentiment_distribution,
    plot_sentiment_vs_recommendation,
    plot_word_frequencies,
    plot_wordcloud,
    plot_yearly_sentiment,
    save_figure,
)
from steam_reviews.analysis.sentiment import add_sentiment, sentiment_summary
from steam_reviews.analysis.statistics import engagement_split, pearson_correlations, yearly_mean
from steam_reviews.analysis.text import (
    PreparedDocuments,
    PreprocessingConfig,
    prepare_documents,
    word_frequencies,
)
from steam_reviews.config import PROCESSED_DIR

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AnalysisReport:
    """Collect the data, tables and figures produced for one game.

    Attributes:
        game: Display name of the analysed game.
        app_id: Steam application ID.
        cutoff: Before/after boundary used by the report.
        output_dir: Directory containing the written artifacts.
        prepared: Reviews with processed text and the shared vocabulary.
        sentiment: Prepared reviews enriched with scores, year and period.
        frequencies: Overall, before and after word-frequency tables.
        correlations: Pearson correlations before and after the cutoff.
        tables: Paths of every written CSV.
        figures: Paths of every written image.
        summary: Counts and aggregate statistics, also written as JSON.
    """

    game: str
    app_id: int
    cutoff: pd.Timestamp
    output_dir: Path
    prepared: PreparedDocuments
    sentiment: pd.DataFrame
    frequencies: dict[str, pd.DataFrame]
    correlations: pd.DataFrame
    tables: dict[str, Path]
    figures: dict[str, Path]
    summary: dict[str, object]


def _save_wordcloud(text: str, title: str, path: Path, figures: dict[str, Path]) -> None:
    """Save a word cloud when the corpus contains text.

    Args:
        text: Space-separated corpus.
        title: Figure title.
        path: Destination image path.
        figures: Mapping updated with the saved path under the image stem.
    """
    if not text.strip():
        logger.warning("Skipping word cloud %s because the corpus is empty", path.name)
        return
    figures[path.stem] = save_figure(plot_wordcloud(text, title=title), path)


def _save_period_figure(
    figure_factory: Callable[[], Figure],
    path: Path,
    figures: dict[str, Path],
) -> None:
    """Save a figure produced by a zero-argument factory, tolerating empty data.

    Args:
        figure_factory: Callable returning a matplotlib figure.
        path: Destination image path.
        figures: Mapping updated with the saved path under the image stem.
    """
    try:
        figure = figure_factory()
    except (ValueError, RuntimeError, FloatingPointError) as exc:
        logger.warning("Skipping figure %s: %s", path.name, exc)
        return
    figures[path.stem] = save_figure(figure, path)


def run_analysis(
    game: int | str,
    cutoff: str | pd.Timestamp,
    *,
    output_dir: Path | None = None,
    config: PreprocessingConfig | None = None,
    min_word_freq: int = 2,
    top_n: int = 20,
    save_plots: bool = True,
    processed_dir: Path = PROCESSED_DIR,
) -> AnalysisReport:
    """Preprocess, score, summarize and plot one game's processed reviews.

    Args:
        game: Steam application ID, numeric string, or display name.
        cutoff: Before/after boundary date; rows on the boundary count as after.
        output_dir: Artifact directory; defaults to `data/processed/analysis/<game>`.
        config: Domain keywords and replacements; defaults to no domain rules.
        min_word_freq: Smallest token frequency retained in documents.
        top_n: Number of most frequent words in each frequency table and chart.
        save_plots: Write PNG artifacts when true.
        processed_dir: Directory containing the processed `reviews.csv`.

    Returns:
        Report with in-memory tables and paths to every written artifact.

    Raises:
        ValueError: The game is missing, the cutoff is invalid, or no review remains.
        OSError: An artifact cannot be written.
    """
    reviews = load_reviews(game, processed_dir)
    name = game_name(reviews)
    app_id = int(reviews.iloc[0]["app_id"])
    boundary = parse_cutoff(cutoff)
    prepared = prepare_documents(reviews, config, min_word_freq=min_word_freq)
    if prepared.reviews.empty:
        raise ValueError(f"No reviews remain after preprocessing for {name}")
    target = output_dir or default_output_dir(name)
    target.mkdir(parents=True, exist_ok=True)

    tables: dict[str, Path] = {}
    documents_path, metadata_path = save_documents(prepared, target)
    tables["documents"] = documents_path
    tables["metadata"] = metadata_path

    before_documents, after_documents = split_by_cutoff(prepared.reviews, boundary)
    frequencies = {
        "overall": word_frequencies(prepared.reviews["document"], top_n=top_n),
        "before": word_frequencies(before_documents["document"], top_n=top_n),
        "after": word_frequencies(after_documents["document"], top_n=top_n),
    }
    for key, table in frequencies.items():
        tables[f"frequencies_{key}"] = save_table(table, target / f"word_frequencies_{key}.csv")

    sentiment = add_period(add_year(add_sentiment(prepared.reviews)), boundary)
    tables["sentiment"] = save_table(sentiment, target / "sentiment.csv")
    correlations = pearson_correlations(sentiment, boundary)
    tables["correlations"] = save_table(correlations, target / "correlations.csv")

    before_sentiment, after_sentiment = split_by_cutoff(sentiment, boundary)
    summary: dict[str, object] = {
        "game": name,
        "app_id": app_id,
        "cutoff": boundary.date().isoformat(),
        "reviews": len(prepared.reviews),
        "vocabulary_size": len(prepared.vocabulary),
        "overall": sentiment_summary(sentiment),
        "before": sentiment_summary(before_sentiment),
        "after": sentiment_summary(after_sentiment),
    }
    tables["summary"] = save_summary(summary, target / "summary.json")

    figures: dict[str, Path] = {}
    if save_plots:
        _save_wordcloud(
            " ".join(prepared.reviews["document"]),
            f"Word Cloud for {name}",
            target / "wordcloud_overall.png",
            figures,
        )
        _save_wordcloud(
            " ".join(before_documents["document"]),
            f"Word Cloud: Reviews Before {boundary.date()}",
            target / "wordcloud_before.png",
            figures,
        )
        _save_wordcloud(
            " ".join(after_documents["document"]),
            f"Word Cloud: Reviews On/After {boundary.date()}",
            target / "wordcloud_after.png",
            figures,
        )
        for year, group in sentiment.groupby("year"):
            _save_wordcloud(
                " ".join(group["document"]),
                f"Word Cloud for Year: {year}",
                target / f"wordcloud_year_{year}.png",
                figures,
            )
        _save_period_figure(
            lambda: plot_word_frequencies(frequencies["overall"], title="Top Word Frequencies"),
            target / "word_frequencies_overall.png",
            figures,
        )
        _save_period_figure(
            lambda: plot_word_frequencies(
                frequencies["before"], title=f"Top Words Before {boundary.date()}"
            ),
            target / "word_frequencies_before.png",
            figures,
        )
        _save_period_figure(
            lambda: plot_word_frequencies(
                frequencies["after"], title=f"Top Words On/After {boundary.date()}"
            ),
            target / "word_frequencies_after.png",
            figures,
        )
        _save_period_figure(
            lambda: plot_sentiment_distribution(
                before_sentiment["sentiment_score"],
                after_sentiment["sentiment_score"],
                cutoff=str(boundary.date()),
            ),
            target / "sentiment_distribution.png",
            figures,
        )
        _save_period_figure(
            lambda: plot_sentiment_categories(
                before_sentiment, after_sentiment, cutoff=str(boundary.date())
            ),
            target / "sentiment_categories.png",
            figures,
        )
        _save_period_figure(
            lambda: plot_yearly_sentiment(yearly_mean(sentiment)),
            target / "sentiment_yearly.png",
            figures,
        )
        _save_period_figure(
            lambda: plot_distribution_fits(
                before_sentiment["sentiment_score"], title="Before Cutoff"
            ),
            target / "distribution_fit_before.png",
            figures,
        )
        _save_period_figure(
            lambda: plot_distribution_fits(
                after_sentiment["sentiment_score"], title="After Cutoff"
            ),
            target / "distribution_fit_after.png",
            figures,
        )
        _save_period_figure(
            lambda: plot_sentiment_vs_recommendation(sentiment),
            target / "sentiment_vs_recommendation.png",
            figures,
        )
        _save_period_figure(
            lambda: plot_boxplot(
                sentiment,
                category_column="recommendation",
                value_column="sentiment_score",
                title="Sentiment Score Distribution by Recommendation",
            ),
            target / "sentiment_boxplot.png",
            figures,
        )
        _save_period_figure(
            lambda: plot_scatter(
                sentiment,
                x_column="hours_played",
                title="Sentiment Score vs. Hours Played",
            ),
            target / "sentiment_vs_hours.png",
            figures,
        )
        _save_period_figure(
            lambda: plot_scatter(
                sentiment,
                x_column="helpful_votes",
                title="Sentiment Score vs. Helpful Votes",
            ),
            target / "sentiment_vs_votes.png",
            figures,
        )
        engaged = engagement_split(sentiment)
        _save_period_figure(
            lambda: plot_engagement_boxplot(engaged),
            target / "sentiment_engagement_boxplot.png",
            figures,
        )

    return AnalysisReport(
        game=name,
        app_id=app_id,
        cutoff=boundary,
        output_dir=target,
        prepared=prepared,
        sentiment=sentiment,
        frequencies=frequencies,
        correlations=correlations,
        tables=tables,
        figures=figures,
        summary=summary,
    )
