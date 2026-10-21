"""Command-line interface: `steam-reviews scrape` and `steam-reviews clean`."""

import argparse
import logging
from collections.abc import Sequence

from steam_reviews.config import load_games
from steam_reviews.io import raw_path, read_raw_reviews, write_processed, write_raw_reviews
from steam_reviews.processing.cleaning import clean_reviews
from steam_reviews.scraping.driver import chrome_driver
from steam_reviews.scraping.scraper import ReviewScraper, ScraperSettings

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with its `scrape` and `clean` subcommands.

    Returns:
        Parser whose parsed namespace has a `command` attribute, plus the options of that command.
    """
    defaults = ScraperSettings()
    parser = argparse.ArgumentParser(
        prog="steam-reviews", description="Scrape and clean Steam community reviews."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    scrape = commands.add_parser("scrape", help="scrape reviews into data/raw")
    scrape.add_argument("app_ids", nargs="*", type=int, help="default: all games in games.toml")
    scrape.add_argument("--target", type=int, default=defaults.target_reviews)
    scrape.add_argument("--filter", dest="browse_filter", default=defaults.browse_filter)
    scrape.add_argument("--language", default=defaults.language)
    scrape.add_argument("--headless", action="store_true")

    commands.add_parser("clean", help="build data/processed/reviews.csv from data/raw")
    return parser


def _scrape(args: argparse.Namespace) -> int:
    """Execute the `scrape` subcommand.

    Args:
        args: Parsed command-line arguments containing target app IDs and scraper settings.

    Returns:
        Process exit code: always 0.
    """
    app_ids: list[int] = args.app_ids or list(load_games())
    settings = ScraperSettings(
        target_reviews=args.target, browse_filter=args.browse_filter, language=args.language
    )
    with chrome_driver(headless=args.headless) as driver:
        scraper = ReviewScraper(driver, settings)
        for app_id in app_ids:
            reviews = scraper.scrape(app_id)
            write_raw_reviews(raw_path(app_id), reviews)
            logger.info("App %s: saved %d reviews", app_id, len(reviews))
    return 0


def _clean() -> int:
    """Execute the `clean` subcommand.

    Returns:
        Process exit code: 0 on success, 1 if no raw reviews are found.
    """
    raw = read_raw_reviews()
    if raw.empty:
        logger.error("No raw reviews found; run `steam-reviews scrape` first")
        return 1
    path = write_processed(clean_reviews(raw, load_games()))
    logger.info("Saved cleaned reviews to %s", path)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI.

    Args:
        argv: Arguments to parse; defaults to `sys.argv[1:]`.

    Returns:
        Process exit code: 0 on success, 1 if `clean` finds no raw data.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = build_parser().parse_args(argv)
    return _scrape(args) if args.command == "scrape" else _clean()
