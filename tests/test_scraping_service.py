"""Test the public scraping workflow and CLI against a simulated Chrome session."""

from datetime import date
from pathlib import Path
from unittest.mock import Mock

import pytest
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.remote.webelement import WebElement

from steam_review_analysis import cli
from steam_review_analysis.io import read_raw_reviews
from steam_review_analysis.schema import Review
from steam_review_analysis.scraping import driver, scraper, service
from steam_review_analysis.scraping.scraper import ScraperSettings, ScrapingError


@pytest.fixture
def browser(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Replace Chrome startup and card parsing with deterministic test substitutes.

    Args:
        monkeypatch: Pytest fixture that restores the replaced functions after each test.

    Returns:
        WebDriver mock exposing one valid review card and inspectable browser method calls.
    """
    browser = Mock(spec=WebDriver)
    browser.find_elements.return_value = [Mock(spec=WebElement)]
    monkeypatch.setattr(driver.webdriver, "Chrome", Mock(return_value=browser))

    def parse_card(card: WebElement, app_id: int) -> Review:
        """Return the same valid review for each supplied game without reading a DOM.

        Args:
            card: Mock card accepted for compatibility with the production parser signature.
            app_id: Application ID included in the returned review record.

        Returns:
            Deterministic review with a fixed date, text, recommendation, and playtime.
        """
        return Review(app_id, date(2024, 5, 21), "Great game", "12.5", "Recommended", "0")

    monkeypatch.setattr(scraper, "parse_card", parse_card)
    return browser


def test_public_workflow_saves_each_game_once(tmp_path: Path, browser: Mock) -> None:
    """Save one CSV per distinct requested game and close the shared browser session."""
    paths = service.scrape_games(
        [42, 43, 42],
        settings=ScraperSettings(target_reviews=1, scroll_pause_s=0),
        raw_dir=tmp_path,
    )
    assert paths == {42: tmp_path / "42_reviews.csv", 43: tmp_path / "43_reviews.csv"}
    assert read_raw_reviews(tmp_path)["Game"].tolist() == [42, 43]
    assert browser.get.call_count == 2
    browser.quit.assert_called_once()


def test_existing_file_fails_before_browser_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check all destinations before launching Chrome or creating any batch output."""
    factory = Mock()
    monkeypatch.setattr(driver.webdriver, "Chrome", factory)
    path = tmp_path / "43_reviews.csv"
    path.write_bytes(b"existing dataset")
    with pytest.raises(FileExistsError):
        service.scrape_games([42, 43], raw_dir=tmp_path)
    factory.assert_not_called()
    assert not (tmp_path / "42_reviews.csv").exists()
    assert path.read_bytes() == b"existing dataset"


@pytest.mark.parametrize("failure_stage", ["get", "find_elements"])
def test_cli_browser_failure_preserves_csv_and_reports_failure(
    tmp_path: Path,
    browser: Mock,
    caplog: pytest.LogCaptureFixture,
    failure_stage: str,
) -> None:
    """Log browser failures, return exit code 1, and preserve the previous CSV."""
    path = tmp_path / "raw" / "42_reviews.csv"
    path.parent.mkdir()
    path.write_bytes(b"existing dataset")
    getattr(browser, failure_stage).side_effect = WebDriverException("browser disconnected")

    data_dir = ["--data-dir", str(tmp_path)]
    assert cli.main([*data_dir, "scrape", "42", "--overwrite", "--scroll-pause", "0"]) == 1
    assert "Scraping failed" in caplog.text
    assert path.read_bytes() == b"existing dataset"
    browser.quit.assert_called_once()


def test_partial_results_are_not_published_on_browser_failure(
    tmp_path: Path, browser: Mock
) -> None:
    """Discard a partial scrape when the browser fails after the first valid card."""
    path = tmp_path / "42_reviews.csv"
    path.write_bytes(b"existing dataset")
    browser.find_elements.side_effect = [[Mock()], WebDriverException("lost browser")]
    with pytest.raises(ScrapingError):
        service.scrape_games(
            [42],
            settings=ScraperSettings(target_reviews=2, scroll_pause_s=0),
            raw_dir=tmp_path,
            overwrite=True,
        )
    assert path.read_bytes() == b"existing dataset"
    browser.quit.assert_called_once()


def test_empty_page_creates_no_csv(tmp_path: Path, browser: Mock) -> None:
    """Fail an empty scrape without publishing a header-only CSV."""
    browser.find_elements.return_value = []
    with pytest.raises(ScrapingError, match="no valid reviews"):
        service.scrape_games(
            [42],
            settings=ScraperSettings(scroll_pause_s=0, max_stalled_scrolls=2),
            raw_dir=tmp_path,
        )
    assert list(tmp_path.iterdir()) == []
    browser.quit.assert_called_once()


def test_shutdown_failure_does_not_mask_scraping_failure(tmp_path: Path, browser: Mock) -> None:
    """Keep the original navigation failure when browser shutdown also raises an error."""
    browser.get.side_effect = WebDriverException("navigation failed")
    browser.quit.side_effect = WebDriverException("shutdown failed")
    with pytest.raises(ScrapingError, match="browser failed") as caught:
        service.scrape_games([42], raw_dir=tmp_path)
    assert caught.value.__cause__ is browser.get.side_effect


@pytest.mark.parametrize("app_ids", [[], [0], [-1]])
def test_invalid_game_selection_does_not_start_browser(
    app_ids: list[int],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject empty or invalid game selections before launching Chrome."""
    factory = Mock()
    monkeypatch.setattr(driver.webdriver, "Chrome", factory)
    with pytest.raises(ValueError):
        service.scrape_games(app_ids, raw_dir=tmp_path)
    factory.assert_not_called()


def test_cli_reports_browser_startup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Report Chrome startup errors through logging and exit code 1 without writing data."""
    factory = Mock(side_effect=WebDriverException("Chrome unavailable"))
    monkeypatch.setattr(driver.webdriver, "Chrome", factory)
    assert cli.main(["--data-dir", str(tmp_path), "scrape", "42"]) == 1
    assert "Chrome unavailable" in caplog.text
    assert list(tmp_path.iterdir()) == []
