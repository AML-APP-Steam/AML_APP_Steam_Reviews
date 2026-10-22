"""Browser setup for the scraper."""

import logging
from collections.abc import Generator
from contextlib import contextmanager

from selenium import webdriver
from selenium.common.exceptions import WebDriverException

logger = logging.getLogger(__name__)


@contextmanager
def chrome_driver(*, headless: bool = False) -> Generator[webdriver.Chrome, None, None]:
    """Open Chrome for the duration of a `with` block and always quit it afterwards.

    Args:
        headless: Run without a visible browser window.

    Yields:
        The Chrome driver. Selenium Manager downloads a matching chromedriver if needed.

    Raises:
        WebDriverException: Chrome could not be started. Shutdown errors are logged so
            they do not mask an earlier scraping failure or invalidate a saved result.
    """
    options = webdriver.ChromeOptions()
    options.add_argument("--lang=en-US")
    if headless:
        options.add_argument("--headless=new")
    driver = webdriver.Chrome(options=options)
    try:
        yield driver
    finally:
        try:
            driver.quit()
        except WebDriverException:
            logger.warning("Chrome could not close cleanly", exc_info=True)
