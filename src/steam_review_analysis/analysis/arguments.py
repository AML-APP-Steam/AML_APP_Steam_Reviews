"""Command-line options of `steam-review-analysis analyse`, importable without the analysis extra.

This module only defines arguments. It deliberately avoids NLTK, SciPy and matplotlib so that
building the parser (and `--help`) stays fast and works on a core install; the command
handlers live in `steam_review_analysis.analysis.cli` and are imported only when a command runs.
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

from steam_review_analysis.analysis.transformer_sentiment import DEFAULT_MODEL, MODEL_SPECS
from steam_review_analysis.config import DEFAULT_PRESETS_FILE

COMMANDS_WITHOUT_NLTK = frozenset({"games"})
"""Analysis subcommands that do not preprocess text and therefore need no NLTK corpora."""


def positive_int(value: str) -> int:
    """Parse a positive integer for argparse.

    Args:
        value: Command-line value.

    Returns:
        Parsed positive integer.

    Raises:
        argparse.ArgumentTypeError: The value is not a positive integer.
    """
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid integer: {value!r}") from None
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
        help="named preset (preprocessing rules and default cutoff) from the presets file",
    )
    parser.add_argument(
        "--presets",
        type=Path,
        help=f"TOML presets file (default: ./{DEFAULT_PRESETS_FILE})",
    )
    parser.add_argument("--output-dir", type=Path, help="artifact directory")
    parser.add_argument(
        "--min-word-freq",
        type=positive_int,
        default=2,
        help="minimum token frequency retained in documents",
    )


def _add_top_argument(parser: argparse.ArgumentParser) -> None:
    """Add the most-frequent-words limit.

    Args:
        parser: Subcommand parser to populate.
    """
    parser.add_argument("--top", type=positive_int, default=20, help="words per frequency table")


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
        type=positive_int,
        default=16,
        help="reviews per transformer forward pass",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "mps"),
        help="torch device; auto-selected when omitted",
    )


def add_parser(
    analyse: argparse.ArgumentParser,
    parents: Sequence[argparse.ArgumentParser] = (),
) -> None:
    """Populate the `analyse` command with its subcommands.

    Args:
        analyse: Parser created for the `analyse` command.
        parents: Parsers whose options (such as `--data-dir`) every subcommand also accepts.
    """
    commands = analyse.add_subparsers(dest="analysis_command", required=True)
    shared = list(parents)

    report = commands.add_parser(
        "report", parents=shared, help="run preprocessing, sentiment and all plots"
    )
    _add_common_arguments(report)
    _add_top_argument(report)

    preprocess = commands.add_parser(
        "preprocess", parents=shared, help="write documents and metadata CSVs"
    )
    _add_common_arguments(preprocess)

    sentiment = commands.add_parser("sentiment", parents=shared, help="score reviews with VADER")
    _add_common_arguments(sentiment)

    transformer = commands.add_parser(
        "transformer-sentiment",
        parents=shared,
        help="score reviews with a small Jev-style transformer model",
    )
    _add_common_arguments(transformer)
    _add_transformer_arguments(transformer)

    frequencies = commands.add_parser(
        "frequencies", parents=shared, help="write overall and period word counts"
    )
    _add_common_arguments(frequencies)
    _add_top_argument(frequencies)

    wordcloud = commands.add_parser(
        "wordcloud", parents=shared, help="write overall and period word clouds"
    )
    _add_common_arguments(wordcloud)

    commands.add_parser("games", parents=shared, help="list games present in processed data")
