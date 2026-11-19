"""Expose scraping, cleaning and analysis workflows through a small command-line interface.

Only pandas is needed to build the parser, so `--help` is fast on any install. Selenium and the
analysis libraries are imported only when the command that needs them runs; without the
matching extra, the command reports how to install it.

Global options (`--data-dir`, `--log-level`) are accepted before or after the command name.
"""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

from steam_review_analysis import __version__, config
from steam_review_analysis.analysis import arguments as analysis_arguments
from steam_review_analysis.config import Game, load_games
from steam_review_analysis.io import read_raw_reviews, write_processed
from steam_review_analysis.processing.cleaning import clean_reviews
from steam_review_analysis.scraping.settings import ScraperSettings, ScrapingError

_logger = logging.getLogger(__name__)

EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_INTERRUPTED = 130
"""Conventional exit status after Ctrl+C (128 + SIGINT)."""
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")
DEFAULT_LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
SCRAPE_HINT = 'install the scraping dependencies with `pip install "steam-review-analysis[scrape]"`'
ANALYSIS_HINT = (
    'install the analysis dependencies with `pip install "steam-review-analysis[analysis]"`'
)


def _load_analysis_cli() -> ModuleType | None:
    """Import the analysis command module if its optional dependencies are installed.

    Returns:
        The `steam_review_analysis.analysis.cli` module, or None without the `analysis` extra.
    """
    try:
        from steam_review_analysis.analysis import cli as analysis_cli
    except ImportError:
        _logger.debug("Analysis dependencies are unavailable", exc_info=True)
        return None
    return analysis_cli


def _add_games_argument(parser: argparse.ArgumentParser) -> None:
    """Add the game-registry option shared by `scrape` and `clean`.

    Args:
        parser: Subcommand parser to populate.
    """
    parser.add_argument(
        "--games",
        type=Path,
        help=f"TOML game registry (default: ./{config.DEFAULT_GAMES_FILE} if it exists)",
    )


def _games(path: Path | None, *, required: bool) -> dict[int, Game]:
    """Load the game registry selected on the command line.

    Args:
        path: File given with `--games`, or None to use the default location if present.
        required: Fail instead of returning an empty registry when no file is found.

    Returns:
        Games keyed by app id; empty when no registry exists and none is required.

    Raises:
        FileNotFoundError: An explicit file is missing, or no file exists and one is required.
    """
    if path is not None:
        return load_games(path)
    default = Path.cwd() / config.DEFAULT_GAMES_FILE
    if default.is_file():
        return load_games(default)
    if required:
        raise FileNotFoundError(
            f"No game registry at ./{config.DEFAULT_GAMES_FILE}; pass app IDs or --games"
        )
    return {}


def _add_scrape_arguments(parser: argparse.ArgumentParser) -> None:
    """Define CLI options using the same defaults as the Python API.

    Args:
        parser: Scrape subcommand parser to populate.
    """
    defaults = ScraperSettings()
    parser.add_argument("app_ids", nargs="*", type=int, help="all registry games if omitted")
    _add_games_argument(parser)
    parser.add_argument(
        "--target",
        type=int,
        default=defaults.target_reviews,
        help="maximum unique reviews per game",
    )
    parser.add_argument(
        "--filter",
        dest="browse_filter",
        default=defaults.browse_filter,
        help="Steam review ordering",
    )
    parser.add_argument("--language", default=defaults.language, help="review content language")
    parser.add_argument("--headless", action="store_true", help="run Chrome without a window")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing CSVs only after a successful scrape",
    )
    parser.add_argument(
        "--scroll-pause",
        type=float,
        default=defaults.scroll_pause_s,
        help="seconds to wait for cards after each scroll (may be zero)",
    )
    parser.add_argument(
        "--wait-timeout",
        type=float,
        default=defaults.wait_timeout_s,
        help="maximum seconds to wait for the page body",
    )
    parser.add_argument(
        "--page-load-timeout",
        type=float,
        default=defaults.page_load_timeout_s,
        help="maximum seconds for page navigation",
    )
    parser.add_argument(
        "--max-stalled-scrolls",
        type=int,
        default=defaults.max_stalled_scrolls,
        help="consecutive scrolls without new valid reviews before stopping",
    )
    parser.add_argument(
        "--max-stale-retries",
        type=int,
        default=defaults.max_stale_retries,
        help="consecutive stale-page errors before failing",
    )


def _global_options(*, with_defaults: bool) -> argparse.ArgumentParser:
    """Build the options accepted both before and after the command name.

    Args:
        with_defaults: Set real defaults (top-level parser). Subcommand copies use
            `argparse.SUPPRESS` so they only override the top-level value when given.

    Returns:
        Parent parser without its own `--help`.
    """
    options = argparse.ArgumentParser(add_help=False)
    group = options.add_argument_group("global options")
    group.add_argument(
        "--data-dir",
        type=Path,
        default=None if with_defaults else argparse.SUPPRESS,
        help=f"directory holding raw/ and processed/ (default: ${config.DATA_DIR_ENV} or ./data)",
    )
    group.add_argument(
        "--log-level",
        choices=LOG_LEVELS,
        default=DEFAULT_LOG_LEVEL if with_defaults else argparse.SUPPRESS,
        help=f"logging verbosity (default: {DEFAULT_LOG_LEVEL})",
    )
    return options


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI with scraping parameters and configurable logging.

    Returns:
        Parser for `scrape`, `clean`, `download-nltk` and `analyse`.
    """
    parser = argparse.ArgumentParser(
        prog="steam-review-analysis",
        description="Scrape, clean and analyse Steam community reviews.",
        parents=[_global_options(with_defaults=True)],
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    shared = [_global_options(with_defaults=False)]
    commands = parser.add_subparsers(dest="command", required=True)
    scrape = commands.add_parser(
        "scrape",
        parents=shared,
        help="scrape reviews into <data-dir>/raw",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_scrape_arguments(scrape)
    clean = commands.add_parser(
        "clean",
        parents=shared,
        help="build <data-dir>/processed/reviews.csv from <data-dir>/raw",
    )
    _add_games_argument(clean)
    commands.add_parser(
        "download-nltk",
        parents=shared,
        help="download the NLTK corpora used by the analysis utilities",
    )
    analyse = commands.add_parser("analyse", parents=shared, help="analyse processed reviews")
    analysis_arguments.add_parser(analyse, parents=shared)
    return parser


def _scrape(args: argparse.Namespace) -> int:
    """Pass CLI options to the public scraping workflow.

    Args:
        args: Namespace produced by `build_parser()` for the scrape command.

    Returns:
        EXIT_SUCCESS when every selected game is saved, otherwise EXIT_FAILURE.

    Raises:
        ValueError: Scraping parameters or application IDs are invalid.
    """
    settings = ScraperSettings(
        target_reviews=args.target,
        browse_filter=args.browse_filter,
        language=args.language,
        scroll_pause_s=args.scroll_pause,
        wait_timeout_s=args.wait_timeout,
        page_load_timeout_s=args.page_load_timeout,
        max_stalled_scrolls=args.max_stalled_scrolls,
        max_stale_retries=args.max_stale_retries,
    )
    try:
        from selenium.common.exceptions import WebDriverException

        from steam_review_analysis.scraping.service import scrape_games
    except ImportError as exc:
        _logger.error("Scraping is unavailable (%s); %s", exc, SCRAPE_HINT)
        return EXIT_FAILURE
    try:
        app_ids: list[int] = args.app_ids or list(_games(args.games, required=True))
        scrape_games(
            app_ids,
            settings=settings,
            headless=args.headless,
            overwrite=args.overwrite,
            raw_dir=config.raw_dir(args.data_dir),
        )
    except (ScrapingError, WebDriverException, OSError) as exc:
        _logger.error("Scraping failed: %s", exc)
        _logger.debug("Scraping failure details", exc_info=True)
        return EXIT_FAILURE
    return EXIT_SUCCESS


def _clean(args: argparse.Namespace) -> int:
    """Validate and clean raw reviews before publishing the processed dataset.

    Args:
        args: Namespace produced by `build_parser()` for the clean command.

    Returns:
        EXIT_SUCCESS on success, EXIT_FAILURE for invalid input, empty output, or an I/O error.
    """
    try:
        raw_dir = config.raw_dir(args.data_dir)
        raw = read_raw_reviews(raw_dir)
        if raw.empty:
            _logger.error("No raw reviews in %s; run `steam-review-analysis scrape` first", raw_dir)
            return EXIT_FAILURE
        games = _games(args.games, required=False)
        cleaned = clean_reviews(raw, games)
        if cleaned.empty:
            _logger.error("No reviews remain after cleaning; existing processed data was preserved")
            return EXIT_FAILURE
        path = write_processed(cleaned, config.processed_dir(args.data_dir))
    except (ValueError, OSError) as exc:
        _logger.error("Cleaning failed: %s", exc)
        _logger.debug("Cleaning failure details", exc_info=True)
        return EXIT_FAILURE
    _logger.info("Saved cleaned reviews to %s", path)
    return EXIT_SUCCESS


def main(argv: Sequence[str] | None = None) -> int:
    """Configure logging and dispatch the requested workflow.

    Args:
        argv: CLI arguments; defaults to the process arguments when None.

    Returns:
        EXIT_SUCCESS on success, EXIT_FAILURE for a reported workflow failure, or
        EXIT_INTERRUPTED when the user pressed Ctrl+C.

    Raises:
        SystemExit: Help or the version was requested (code 0) or arguments are invalid
            (code 2).
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format=LOG_FORMAT)
    try:
        return _dispatch(parser, args)
    except KeyboardInterrupt:
        _logger.error("Interrupted by user")
        return EXIT_INTERRUPTED


def _dispatch(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Run the workflow selected on the command line.

    Args:
        parser: Parser used to report invalid scraping parameters as usage errors.
        args: Parsed arguments.

    Returns:
        Exit code of the selected workflow.
    """
    if args.command == "clean":
        return _clean(args)
    if args.command in {"analyse", "download-nltk"}:
        analysis_cli = _load_analysis_cli()
        if analysis_cli is None:
            _logger.error("The %s command is unavailable; %s", args.command, ANALYSIS_HINT)
            return EXIT_FAILURE
        if args.command == "download-nltk":
            return int(analysis_cli.download_nltk())
        return int(analysis_cli.run(args))
    try:
        return _scrape(args)
    except ValueError as exc:
        parser.error(str(exc))
