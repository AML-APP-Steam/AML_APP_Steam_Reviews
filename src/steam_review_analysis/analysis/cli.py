"""Handlers for `steam-review-analysis analyse`; requires the `analysis` extra.

The options are defined in `steam_review_analysis.analysis.arguments`, which stays importable
without the heavy dependencies imported here.
"""

import argparse
import logging
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from steam_review_analysis import config
from steam_review_analysis.analysis.arguments import COMMANDS_WITHOUT_NLTK
from steam_review_analysis.analysis.artifacts import save_documents, save_table
from steam_review_analysis.analysis.dataset import (
    default_output_dir,
    game_name,
    load_reviews,
    split_by_cutoff,
)
from steam_review_analysis.analysis.nltk_resources import ensure_nltk_resources
from steam_review_analysis.analysis.plotting import plot_wordcloud, save_figure
from steam_review_analysis.analysis.presets import GamePreset, load_presets, preset_for
from steam_review_analysis.analysis.report import run_analysis
from steam_review_analysis.analysis.sentiment import add_sentiment, sentiment_summary
from steam_review_analysis.analysis.text import (
    PreprocessingConfig,
    prepare_documents,
    word_frequencies,
)
from steam_review_analysis.analysis.transformer_sentiment import (
    add_transformer_sentiment,
    transformer_sentiment_summary,
)
from steam_review_analysis.config import DEFAULT_PRESETS_FILE
from steam_review_analysis.io import read_processed

_logger = logging.getLogger(__name__)

EXIT_SUCCESS = 0
EXIT_FAILURE = 1


def _preset(args: argparse.Namespace) -> GamePreset | None:
    """Return the selected domain preset, if any.

    Args:
        args: Parsed analysis arguments.

    Returns:
        Selected preset or None.

    Raises:
        OSError: A preset was requested but the presets file cannot be read.
        KeyError: The presets file does not define the requested preset.
    """
    selected: str | None = args.preset
    if not selected:
        return None
    path: Path = args.presets or Path.cwd() / DEFAULT_PRESETS_FILE
    return preset_for(selected, load_presets(path))


def _config(args: argparse.Namespace) -> PreprocessingConfig | None:
    """Return preprocessing rules for the selected preset.

    Args:
        args: Parsed analysis arguments.

    Returns:
        Preset preprocessing config or None for default rules.
    """
    selected = _preset(args)
    return selected.preprocessing if selected is not None else None


def _cutoff(args: argparse.Namespace) -> str:
    """Resolve the cutoff from the command line or selected preset.

    Args:
        args: Parsed analysis arguments.

    Returns:
        ISO cutoff date.

    Raises:
        ValueError: No cutoff was given and the command has no preset.
    """
    if args.cutoff:
        return str(args.cutoff)
    selected = _preset(args)
    if selected is None:
        raise ValueError("--cutoff is required unless --preset supplies a default")
    return selected.cutoff


def _target_dir(args: argparse.Namespace, reviews: pd.DataFrame) -> Path:
    """Resolve the artifact directory for one game.

    Args:
        args: Parsed analysis arguments.
        reviews: Loaded reviews for the game.

    Returns:
        Requested or default analysis directory.
    """
    output: Path | None = args.output_dir
    if output is not None:
        return output
    return default_output_dir(game_name(reviews), config.analysis_dir(args.data_dir))


def _processed_dir(args: argparse.Namespace) -> Path:
    """Resolve the processed-data directory selected with `--data-dir`.

    Args:
        args: Parsed arguments, including the top-level `data_dir` option.

    Returns:
        Directory containing `reviews.csv`.
    """
    return config.processed_dir(args.data_dir)


def _run_report(args: argparse.Namespace) -> int:
    """Run the complete report workflow.

    Args:
        args: Parsed analysis arguments.

    Returns:
        Success exit code.
    """
    cutoff = _cutoff(args)
    report = run_analysis(
        args.game,
        cutoff,
        output_dir=args.output_dir,
        config=_config(args),
        min_word_freq=args.min_word_freq,
        top_n=args.top,
        processed_dir=_processed_dir(args),
    )
    _logger.info(
        "Analysed %s: %d reviews, %d tables, %d figures in %s",
        report.game,
        len(report.prepared.reviews),
        len(report.tables),
        len(report.figures),
        report.output_dir,
    )
    return EXIT_SUCCESS


def _run_preprocess(args: argparse.Namespace) -> int:
    """Write preprocessed documents and metadata for one game.

    Args:
        args: Parsed analysis arguments.

    Returns:
        Success exit code.
    """
    reviews = load_reviews(args.game, _processed_dir(args))
    prepared = prepare_documents(
        reviews,
        _config(args),
        min_word_freq=args.min_word_freq,
    )
    documents, metadata = save_documents(prepared, _target_dir(args, reviews))
    _logger.info(
        "Wrote %d documents (%d vocabulary terms) to %s and %s",
        len(prepared.reviews),
        len(prepared.vocabulary),
        documents,
        metadata,
    )
    return EXIT_SUCCESS


def _run_sentiment(args: argparse.Namespace) -> int:
    """Score reviews and write the sentiment table.

    Args:
        args: Parsed analysis arguments.

    Returns:
        Success exit code.
    """
    reviews = load_reviews(args.game, _processed_dir(args))
    prepared = prepare_documents(
        reviews,
        _config(args),
        min_word_freq=args.min_word_freq,
    )
    scored = add_sentiment(prepared.reviews)
    path = save_table(scored, _target_dir(args, reviews) / "sentiment.csv")
    summary = sentiment_summary(scored)
    _logger.info("Wrote %s; mean compound score %.3f", path, summary["mean_score"])
    return EXIT_SUCCESS


def _run_transformer_sentiment(args: argparse.Namespace) -> int:
    """Score reviews with a Jev-style transformer and write the sentiment table.

    Args:
        args: Parsed analysis arguments.

    Returns:
        Success exit code.

    Raises:
        RuntimeError: The optional sentiment dependencies are not installed.
    """
    reviews = load_reviews(args.game, _processed_dir(args))
    prepared = prepare_documents(
        reviews,
        _config(args),
        min_word_freq=args.min_word_freq,
    )
    scored = add_transformer_sentiment(
        prepared.reviews,
        model=args.model,
        batch_size=args.batch_size,
        device=args.device,
    )
    path = save_table(scored, _target_dir(args, reviews) / "transformer_sentiment.csv")
    summary = transformer_sentiment_summary(scored)
    _logger.info(
        "Wrote %s with %s; mean expected polarity %.3f",
        path,
        args.model,
        summary["mean_score"],
    )
    return EXIT_SUCCESS


def _run_frequencies(args: argparse.Namespace) -> int:
    """Write overall and period word-frequency tables.

    Args:
        args: Parsed analysis arguments.

    Returns:
        Success exit code.
    """
    cutoff = _cutoff(args)
    reviews = load_reviews(args.game, _processed_dir(args))
    prepared = prepare_documents(
        reviews,
        _config(args),
        min_word_freq=args.min_word_freq,
    )
    before, after = split_by_cutoff(prepared.reviews, cutoff)
    target = _target_dir(args, reviews)
    tables = {
        "overall": word_frequencies(prepared.reviews["document"], top_n=args.top),
        "before": word_frequencies(before["document"], top_n=args.top),
        "after": word_frequencies(after["document"], top_n=args.top),
    }
    for name, table in tables.items():
        path = save_table(table, target / f"word_frequencies_{name}.csv")
        _logger.info("Wrote %s", path)
    return EXIT_SUCCESS


def _run_wordcloud(args: argparse.Namespace) -> int:
    """Write overall and period word clouds.

    Args:
        args: Parsed analysis arguments.

    Returns:
        Success exit code.
    """
    cutoff = _cutoff(args)
    reviews = load_reviews(args.game, _processed_dir(args))
    prepared = prepare_documents(
        reviews,
        _config(args),
        min_word_freq=args.min_word_freq,
    )
    before, after = split_by_cutoff(prepared.reviews, cutoff)
    target = _target_dir(args, reviews)
    corpora = {
        "overall": (" ".join(prepared.reviews["document"]), f"Word Cloud for {game_name(reviews)}"),
        "before": (" ".join(before["document"]), f"Word Cloud: Reviews Before {cutoff}"),
        "after": (" ".join(after["document"]), f"Word Cloud: Reviews On/After {cutoff}"),
    }
    for name, (text, title) in corpora.items():
        if not text.strip():
            _logger.warning("Skipping empty %s word cloud", name)
            continue
        path = save_figure(plot_wordcloud(text, title=title), target / f"wordcloud_{name}.png")
        _logger.info("Wrote %s", path)
    return EXIT_SUCCESS


def _run_games(args: argparse.Namespace) -> int:
    """Log the games and review counts available in processed data.

    Args:
        args: Parsed analysis arguments.

    Returns:
        Success exit code, or failure when no processed reviews exist.
    """
    reviews = read_processed(_processed_dir(args))
    if reviews.empty:
        _logger.error(
            "No processed reviews found; run `steam-review-analysis scrape` and `clean` first"
        )
        return EXIT_FAILURE
    if "game" not in reviews.columns:
        reviews["game"] = reviews["app_id"].astype(str)
    grouped = reviews.groupby(["app_id", "game"], dropna=False).size().reset_index(name="reviews")
    for row in grouped.sort_values(["game", "app_id"]).itertuples():
        _logger.info("%s (app %s): %d reviews", row.game, row.app_id, row.reviews)
    return EXIT_SUCCESS


def download_nltk() -> int:
    """Download every NLTK corpus the analysis utilities need.

    Returns:
        Success exit code, or failure when a download fails.
    """
    try:
        ensure_nltk_resources(download=True)
    except RuntimeError as exc:
        _logger.error("%s", exc)
        return EXIT_FAILURE
    _logger.info("All NLTK resources are installed")
    return EXIT_SUCCESS


def run(args: argparse.Namespace) -> int:
    """Dispatch an analysis subcommand and translate expected errors to exit codes.

    Args:
        args: Parsed analysis arguments; `analysis_command` selects the handler.

    Returns:
        Success or failure exit code.
    """
    command: str = args.analysis_command
    handler = _HANDLERS[command]
    try:
        if command not in COMMANDS_WITHOUT_NLTK:
            ensure_nltk_resources(download=True)
        return int(handler(args))
    except (KeyError, ValueError, OSError, RuntimeError) as exc:
        _logger.error("Analysis failed: %s", exc)
        _logger.debug("Analysis failure details", exc_info=True)
        return EXIT_FAILURE


_HANDLERS: dict[str, Callable[[argparse.Namespace], int]] = {
    "report": _run_report,
    "preprocess": _run_preprocess,
    "sentiment": _run_sentiment,
    "transformer-sentiment": _run_transformer_sentiment,
    "frequencies": _run_frequencies,
    "wordcloud": _run_wordcloud,
    "games": _run_games,
}
