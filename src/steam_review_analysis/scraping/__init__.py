"""Selenium-based scraping of Steam review pages into `steam_review_analysis.schema.Review` records.

Requires the `scrape` extra (`pip install "steam-review-analysis[scrape]"`) and Google Chrome.
The browser workflow lives in `steam_review_analysis.scraping.service.scrape_games`; settings and
errors below are importable without Selenium.
"""

from steam_review_analysis.scraping.settings import ScraperSettings, ScrapingError, validate_app_id

__all__ = ["ScraperSettings", "ScrapingError", "validate_app_id"]
