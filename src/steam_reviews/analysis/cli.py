"""Command-line access to the reusable review analysis utilities."""

import argparse
import logging
from pathlib import Path

import pandas as pd

from steam_reviews.analysis.artifacts import save_documents, save_table
from steam_reviews.analysis.dataset import (
    default_output_dir,
    game_name,
    load_reviews,
    split_by_cutoff,
)
from steam_reviews.analysis.plotting import plot_wordcloud, save_figure
from steam_reviews.analysis.presets import ANALYSIS_PRESETS, GamePreset, preset_for
from steam_reviews.analysis.report import run_analysis
from steam_reviews.analysis.sentiment import add_sentiment, sentiment_summary
from steam_reviews.analysis.text import PreprocessingConfig, prepare_documents, word_frequencies
from steam_reviews.analysis.transformer_sentiment import (
    DEFAULT_MODEL,
    MODEL_SPECS,
    add_transformer_sentiment,
    transformer_sentiment_summary,
)
from steam_reviews.io import read_processed

logger = logging.getLogger(__name__)

EXIT_SUCCESS = 0
EXIT_FAILURE = 1


def _positive_int(value: str) -> int:
    """Parse a positive integer for argparse.

    Args:
        value: Command-line value.

    Returns:
        Parsed positive integer.

    Raises:
        argparse.ArgumentTypeError: The value is not a positive integer.
    """
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return number


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add game, cutoff, preset and output options shared by analysis commands.

    Args:
        parser: Subcommand parser to populate.
    """
    parser.add_argument("--game", required=True, help="Steam app id or processed game name")
    parser.add_argument("--cutoff", help="before/after boundary date; preset default if omitted")
    parser.add_argument(
        "--preset",
        choices=sorted(ANALYSIS_PRESETS),
        help="domain preprocessing preset and default cutoff",
    )
    parser.add_argument("--output-dir", type=Path, help="artifact directory")
    parser.add_argument(
        "--min-word-freq",
        type=_positive_int,
        default=2,
        help="minimum token frequency retained in documents",
    )


def _add_top_argument(parser: argparse.ArgumentParser) -> None:
    """Add the most-frequent-words limit.

    Args:
        parser: Subcommand parser to populate.
    """
    parser.add_argument("--top", type=_positive_int, default=20, help="words per frequency table")


def _add_transformer_arguments(parser: argparse.ArgumentParser) -> None:
    """Add model, batch-size and device options for transformer sentiment.

    Args:
        parser: Subcommand parser to populate.
    """
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_SPECS),
        default=DEFAULT_MODEL,
        help="Jev-style transformer model; nano-jev is smallest and fastest",
    )
    parser.add_argument(
        "--batch-size",
        type=_positive_int,
        default=16,
        help="reviews per transformer forward pass",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "mps"),
        help="torch device; auto-selected when omitted",
    )


def add_parser(analyse: argparse.ArgumentParser) -> None:
    """Populate the `analyse` command with its subcommands.

    Args:
        analyse: Parser created for the `analyse` command.
    """
    commands = analyse.add_subparsers(dest="analysis_command", required=True)
    report = commands.add_parser("report", help="run preprocessing, sentiment and all plots")
    _add_common_arguments(report)
    _add_top_argument(report)
    report.set_defaults(analysis_handler=_run_report)

    preprocess = commands.add_parser("preprocess", help="write documents and metadata CSVs")
    _add_common_arguments(preprocess)
    preprocess.set_defaults(analysis_handler=_run_preprocess)

    sentiment = commands.add_parser("sentiment", help="score reviews with VADER")
    _add_common_arguments(sentiment)
    sentiment.set_defaults(analysis_handler=_run_sentiment)

    transformer = commands.add_parser(
        "transformer-sentiment",
        help="score reviews with a small Jev-style transformer model",
    )
    _add_common_arguments(transformer)
    _add_transformer_arguments(transformer)
    transformer.set_defaults(analysis_handler=_run_transformer_sentiment)

    frequencies = commands.add_parser("frequencies", help="write overall and period word counts")
    _add_common_arguments(frequencies)
    _add_top_argument(frequencies)
    frequencies.set_defaults(analysis_handler=_run_frequencies)

    wordcloud = commands.add_parser("wordcloud", help="write overall and period word clouds")
    _add_common_arguments(wordcloud)
    wordcloud.set_defaults(analysis_handler=_run_wordcloud)

    games = commands.add_parser("games", help="list games present in processed data")
    games.set_defaults(analysis_handler=_run_games)


def _preset(args: argparse.Namespace) -> GamePreset | None:
    """Return the selected domain preset, if any.

    Args:
        args: Parsed analysis arguments.

    Returns:
        Selected preset or None.
    """
    selected: str | None = args.preset
    return preset_for(selected) if selected else None


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
    return output if output is not None else default_output_dir(game_name(reviews))


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
    )
    logger.info(
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
    reviews = load_reviews(args.game)
    prepared = prepare_documents(
        reviews,
        _config(args),
        min_word_freq=args.min_word_freq,
    )
    documents, metadata = save_documents(prepared, _target_dir(args, reviews))
    logger.info(
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
    reviews = load_reviews(args.game)
    prepared = prepare_documents(
        reviews,
        _config(args),
        min_word_freq=args.min_word_freq,
    )
    scored = add_sentiment(prepared.reviews)
    path = save_table(scored, _target_dir(args, reviews) / "sentiment.csv")
    summary = sentiment_summary(scored)
    logger.info("Wrote %s; mean compound score %.3f", path, summary["mean_score"])
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
    reviews = load_reviews(args.game)
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
    logger.info(
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
    reviews = load_reviews(args.game)
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
        logger.info("Wrote %s", path)
    return EXIT_SUCCESS


def _run_wordcloud(args: argparse.Namespace) -> int:
    """Write overall and period word clouds.

    Args:
        args: Parsed analysis arguments.

    Returns:
        Success exit code.
    """
    cutoff = _cutoff(args)
    reviews = load_reviews(args.game)
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
            logger.warning("Skipping empty %s word cloud", name)
            continue
        path = save_figure(plot_wordcloud(text, title=title), target / f"wordcloud_{name}.png")
        logger.info("Wrote %s", path)
    return EXIT_SUCCESS


def _run_games(args: argparse.Namespace) -> int:
    """Log the games and review counts available in processed data.

    Args:
        args: Parsed analysis arguments.

    Returns:
        Success exit code, or failure when no processed reviews exist.
    """
    del args
    reviews = read_processed()
    if reviews.empty:
        logger.error("No processed reviews found; run `steam-reviews scrape` and `clean` first")
        return EXIT_FAILURE
    if "game" not in reviews.columns:
        reviews["game"] = reviews["app_id"].astype(str)
    grouped = reviews.groupby(["app_id", "game"], dropna=False).size().reset_index(name="reviews")
    for row in grouped.sort_values(["game", "app_id"]).itertuples():
        logger.info("%s (app %s): %d reviews", row.game, row.app_id, row.reviews)
    return EXIT_SUCCESS


def run(args: argparse.Namespace) -> int:
    """Dispatch an analysis subcommand and translate expected errors to exit codes.

    Args:
        args: Parsed analysis arguments with an `analysis_handler` default.

    Returns:
        Success or failure exit code.
    """
    handler = args.analysis_handler
    try:
        return int(handler(args))
    except (KeyError, ValueError, OSError, RuntimeError) as exc:
        logger.error("Analysis failed: %s", exc)
        logger.debug("Analysis failure details", exc_info=True)
        return EXIT_FAILURE
