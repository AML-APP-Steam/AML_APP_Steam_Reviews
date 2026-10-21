# AML_APP_Steam_Reviews

Scrape [Steam](https://steamcommunity.com) community reviews and analyse them.

## Layout

```
configs/games.toml          games to scrape (app id + name)
data/raw/                   scraped CSVs, one per game        (gitignored)
data/processed/             cleaned, analysis-ready data      (gitignored)
docs/data_schema.md         what each CSV column means
notebooks/                  exploration and analysis
src/steam_reviews/
    scraping/               Selenium only: driver, page parsing, scroll loop
    processing/             pandas only: raw -> typed, deduplicated data
    config.py  schema.py    paths, game registry, column names, Review record
    io.py                   reading and writing the CSVs
    cli.py                  `steam-reviews scrape | clean`
tests/
```

Data flows one way: `scraping` writes `data/raw` -> `processing` writes `data/processed` ->
notebooks read it with `steam_reviews.io.read_processed()`. `scraping` and `processing` never
import each other; both depend only on `schema`, `config` and `io`.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Google Chrome.

```bash
make install          # uv sync + install git hooks
uv sync --group notebooks   # optional: JupyterLab, matplotlib
```

## Usage

```bash
uv run steam-reviews scrape                       # every game in configs/games.toml
uv run steam-reviews scrape 105600 --target 1000  # one game, more reviews
uv run steam-reviews scrape --headless
uv run steam-reviews clean                        # data/raw -> data/processed/reviews.csv
```

To add a game, append an entry to `configs/games.toml`.

## API documentation

Generated from docstrings (Google style) and type hints with [pdoc](https://pdoc.dev):

```bash
make docs         # live preview in the browser
make docs-build   # static HTML in build/docs/ (gitignored)
```

Every public module, class, function and method needs a docstring; ruff's `D` rules enforce it.

## Quality checks

```bash
make check    # ruff (lint + format check), mypy --strict, pytest with coverage
make format   # auto-fix lint and formatting
```

Pre-commit runs ruff, mypy, file hygiene checks, `uv lock` consistency and `nbstripout`
(notebook outputs are never committed) on every commit, and pytest on push. CI runs the same
checks on pull requests.

Changes go through pull requests; scraped data is never committed.
