# Scraping and cleaning

The data pipeline has two steps: `scrape` writes one raw CSV per game, and `clean` turns all raw
CSVs into a single processed dataset. Column meanings are in [data_schema.md](data_schema.md).

## Scraping

Requires the `scrape` extra and Google Chrome. Selenium Manager downloads a matching
chromedriver automatically.

```bash
steam-review-analysis scrape                        # every game in ./configs/games.toml
steam-review-analysis scrape 105600 --target 1000   # one game, more reviews
steam-review-analysis scrape 105600 --headless --overwrite   # replace an existing CSV
steam-review-analysis scrape --games my_games.toml  # another game registry
steam-review-analysis scrape 105600 --log-level DEBUG        # detailed diagnostics
```

### Options

`ScraperSettings` (in `steam_review_analysis.scraping`) holds the defaults for both the CLI and
the Python API. Invalid counts, negative waits, zero timeouts and non-finite times are rejected
before Chrome starts.

| CLI option | Default | Meaning |
| --- | --- | --- |
| `--target` | `600` | Maximum unique reviews per game. |
| `--filter` | `toprated` | Steam review ordering; use `mostrecent` for recent reviews. |
| `--language` | `english` | Review content language; UI labels are always requested in English. |
| `--scroll-pause` | `2.0` | Seconds to allow cards to load after a scroll. |
| `--wait-timeout` | `5.0` | Maximum seconds waiting for the page body. |
| `--page-load-timeout` | `30.0` | Maximum seconds for navigation. |
| `--max-stalled-scrolls` | `5` | Consecutive scrolls without new valid reviews before stopping. |
| `--max-stale-retries` | `3` | Consecutive stale-page errors before failing. |
| `--headless` | off | Run Chrome without a window. |
| `--overwrite` | off | Replace existing CSVs (only after a successful scrape). |

### From Python

```python
import logging

from steam_review_analysis.scraping import ScraperSettings
from steam_review_analysis.scraping.service import scrape_games

logging.basicConfig(level=logging.INFO)

settings = ScraperSettings(target_reviews=100)
saved_paths = scrape_games([105600], settings=settings, headless=True)  # {app_id: Path}
```

Repeated ids are scraped once. The function manages Chrome's lifecycle and logs progress. For
records without saving, use `ReviewScraper(driver).scrape(app_id)`, which returns
`list[Review]`.

### Output and failure behaviour

- Raw files are `<data-dir>/raw/<app_id>_reviews.csv`.
- Existing files are protected: without `--overwrite` (`overwrite=True`) the command fails
  before the browser starts. All destinations are checked first.
- CSVs are written to a temporary file next to the destination and published only when
  complete. Browser failures, empty results, exhausted stale retries and interrupted writes
  keep the old CSV.
- If loading stops after some valid reviews were collected, the smaller sample is saved with a
  warning. `--target` is an upper bound, not a guarantee.
- A batch stops on the first failure; CSVs already saved for earlier games remain.
- Dates accept full or abbreviated English month names, either day/month order, and a missing
  year (taken as the current year; `parse_review_date(..., reference_date=...)` handles
  archived labels). Deduplication compares all raw fields; Steam review ids are not available.

## Cleaning

```bash
steam-review-analysis clean                     # <data-dir>/raw -> <data-dir>/processed/reviews.csv
steam-review-analysis clean --games games.toml  # add display names from another registry
```

`clean` reads every raw CSV, calls `clean_reviews(raw, games)` and writes the result
atomically. In Python, `clean_reviews` takes a DataFrame and returns a new one without
modifying its input. All six raw columns are required; extra columns are ignored.

- Empty review text and exact duplicates are removed; the counts are logged.
- Text keeps its case, punctuation and internal whitespace. Literal reviews such as `NA`,
  `NULL` or `001` survive reading and writing.
- App ids must be positive whole numbers. Dates accept ISO dates and timestamps. Missing or
  invalid ids or dates raise a clear validation error.
- Recommendation labels are normalised for whitespace and case; unknown labels are an error
  rather than silently becoming negative examples.
- Hours are floats; missing, invalid, negative or infinite values become NaN. Helpful-vote
  labels become integer counts; labels without a number count as zero.
- With a game registry, a `game` column is added; ids missing from the registry stay empty
  and are logged.
- Invalid input or an empty result fails with exit code 1 before writing, so the existing
  processed dataset is kept.
