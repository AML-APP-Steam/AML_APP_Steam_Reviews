"""Browser setup for the scraper."""

from collections.abc import Iterator
from contextlib import contextmanager

from selenium import webdriver


@contextmanager
def chrome_driver(*, headless: bool = False) -> Iterator[webdriver.Chrome]:
    """Open Chrome for the duration of a `with` block and always quit it afterwards.

    Args:
        headless: Run without a visible browser window.

    Yields:
        The Chrome driver. Selenium Manager downloads a matching chromedriver if needed.
    """
    options = webdriver.ChromeOptions()
    if headless:
        options.add_argument("--headless=new")
    driver = webdriver.Chrome(options=options)
    try:
        yield driver
    finally:
        driver.quit()
