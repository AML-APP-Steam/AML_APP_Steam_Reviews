"""Expose scraping and cleaning workflows through a small command-line interface."""

import argparse
import logging
from collections.abc import Sequence

from selenium.common.exceptions import WebDriverException

from steam_reviews.analysis import cli as analysis_cli
from steam_reviews.config import load_games
from steam_reviews.io import read_raw_reviews, write_processed
from steam_reviews.processing.cleaning import clean_reviews
from steam_reviews.scraping.scraper import ScraperSettings, ScrapingError
from steam_reviews.scraping.service import scrape_games

logger = logging.getLogger(__name__)

EXIT_SUCCESS = 0
EXIT_FAILURE = 1
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def _add_scrape_arguments(parser: argparse.ArgumentParser) -> None:
    """Define CLI options using the same defaults as the Python API.

    Args:
        parser: Scrape subcommand parser to populate.
    """
    defaults = ScraperSettings()
    parser.add_argument("app_ids", nargs="*", type=int, help="all configured games if omitted")
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


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI with scraping parameters and configurable logging.

    Returns:
        Parser for `scrape` and `clean`, displaying option defaults in command help.
    """
    parser = argparse.ArgumentParser(
        prog="steam-reviews", description="Scrape and clean Steam community reviews."
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="logging verbosity (default: INFO)",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    scrape = commands.add_parser(
        "scrape",
        help="scrape reviews into data/raw",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_scrape_arguments(scrape)
    commands.add_parser("clean", help="build data/processed/reviews.csv from data/raw")
    analyse = commands.add_parser("analyse", help="analyse processed reviews")
    analysis_cli.add_parser(analyse)
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
        app_ids: list[int] = args.app_ids or list(load_games())
        scrape_games(app_ids, settings=settings, headless=args.headless, overwrite=args.overwrite)
    except (ScrapingError, WebDriverException, OSError) as exc:
        logger.error("Scraping failed: %s", exc)
        logger.debug("Scraping failure details", exc_info=True)
        return EXIT_FAILURE
    return EXIT_SUCCESS


def _clean() -> int:
    """Validate and clean raw reviews before publishing the processed dataset.

    Returns:
        EXIT_SUCCESS on success, EXIT_FAILURE for invalid input, empty output, or an I/O error.
    """
    try:
        raw = read_raw_reviews()
        if raw.empty:
            logger.error("No raw reviews found; run `steam-reviews scrape` first")
            return EXIT_FAILURE
        cleaned = clean_reviews(raw, load_games())
        if cleaned.empty:
            logger.error("No reviews remain after cleaning; existing processed data was preserved")
            return EXIT_FAILURE
        path = write_processed(cleaned)
    except (ValueError, OSError) as exc:
        logger.error("Cleaning failed: %s", exc)
        logger.debug("Cleaning failure details", exc_info=True)
        return EXIT_FAILURE
    logger.info("Saved cleaned reviews to %s", path)
    return EXIT_SUCCESS


def main(argv: Sequence[str] | None = None) -> int:
    """Configure logging and dispatch the requested workflow.

    Args:
        argv: CLI arguments; defaults to the process arguments when None.

    Returns:
        EXIT_SUCCESS on success, EXIT_FAILURE for a reported workflow failure.

    Raises:
        SystemExit: Help was requested (code 0) or arguments are invalid (code 2).
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format=LOG_FORMAT)
    if args.command == "clean":
        return _clean()
    if args.command == "analyse":
        return analysis_cli.run(args)
    try:
        return _scrape(args)
    except ValueError as exc:
        parser.error(str(exc))
