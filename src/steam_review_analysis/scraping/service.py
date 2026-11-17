"""Provide the public scraping workflow for the CLI and future notebooks."""

import logging
from collections.abc import Sequence
from pathlib import Path

from steam_review_analysis.io import raw_path, write_raw_reviews
from steam_review_analysis.scraping.driver import chrome_driver
from steam_review_analysis.scraping.scraper import ReviewScraper
from steam_review_analysis.scraping.settings import ScraperSettings, validate_app_id

_logger = logging.getLogger(__name__)


def scrape_games(
    app_ids: Sequence[int],
    *,
    settings: ScraperSettings | None = None,
    headless: bool = False,
    overwrite: bool = False,
    raw_dir: Path | None = None,
) -> dict[int, Path]:
    """Open a browser, scrape selected games, and safely write their raw CSVs.

    Args:
        app_ids: Nonempty sequence of positive Steam IDs; repeated IDs run only once.
        settings: Scraper parameters; defaults to the documented `ScraperSettings()`.
        headless: Whether Chrome runs without a visible window.
        overwrite: Allow replacing existing raw CSVs after a successful scrape.
        raw_dir: Destination directory; defaults to `steam_review_analysis.config.raw_dir()`.

    Returns:
        Mapping of application IDs to their successfully written CSV paths.

    Raises:
        ValueError: No application IDs were given or an ID is invalid.
        FileExistsError: A destination exists and overwrite was not explicitly enabled.
        ScrapingError: A game could not be scraped. Its existing CSV is preserved;
            files successfully saved for earlier games remain available.
        WebDriverException: Chrome could not start.
        OSError: A CSV could not be published; its previous version is preserved.
    """
    parameters = settings or ScraperSettings()
    if not app_ids:
        raise ValueError("Provide at least one application ID")
    paths: dict[int, Path] = {}
    for app_id in app_ids:
        validate_app_id(app_id)
        path = raw_path(app_id, raw_dir)
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} already exists; use --overwrite to replace it")
        paths[app_id] = path

    with chrome_driver(headless=headless) as driver:
        scraper = ReviewScraper(driver, parameters)
        for app_id, path in paths.items():
            _logger.info("App %s: starting scrape", app_id)
            reviews = scraper.scrape(app_id)
            write_raw_reviews(path, reviews, overwrite=overwrite)
            _logger.info("App %s: saved %d reviews to %s", app_id, len(reviews), path)
    return paths
