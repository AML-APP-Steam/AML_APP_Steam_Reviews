# steam-review-analysis

[![CI](https://github.com/G-a-b-b-s/AML_APP_Steam_Reviews/actions/workflows/ci.yml/badge.svg)](https://github.com/G-a-b-b-s/AML_APP_Steam_Reviews/actions/workflows/ci.yml)

Collect [Steam](https://steamcommunity.com) community reviews, clean them into an
analysis-ready dataset, and analyse them: text preprocessing, word frequencies and word clouds,
VADER and transformer sentiment, before/after comparisons and statistics. Use it from the
command line or from Python.

## Installation

Requires Python 3.11 or newer. The core install only needs pandas; heavier features are
optional extras:

```bash
pip install steam-review-analysis                 # read, write and clean review data
pip install "steam-review-analysis[scrape]"       # + scraper (Selenium; needs Google Chrome)
pip install "steam-review-analysis[analysis]"     # + NLTK preprocessing, VADER, statistics, plots
pip install "steam-review-analysis[sentiment]"    # + transformer sentiment models (PyTorch)
pip install "steam-review-analysis[all]"          # everything
```

A command that needs a missing extra stops with the exact `pip install` line to run.

## Quick start

```bash
# 1. Scrape up to 600 reviews of Terraria (app id 105600) into ./data/raw
steam-review-analysis scrape 105600 --headless

# 2. Clean all raw CSVs into ./data/processed/reviews.csv
steam-review-analysis clean

# 3. Analyse one game around a before/after cutoff date
steam-review-analysis analyse report --game 105600 --cutoff 2024-01-01
```

The report writes word frequencies, sentiment scores, statistics, a JSON summary and plots to
`./data/processed/analysis/<game>/`, where `<game>` is the game's name from the game registry
(see below) or its app id. Run `steam-review-analysis --help`, or `--help` after
any command, for all options; `steam-review-analysis --version` prints the version.

The same steps from Python. Unlike the `analyse` commands, library functions never download
the NLTK corpora themselves, so install them once first (or run
`steam-review-analysis download-nltk`):

```python
import steam_review_analysis as sra
from steam_review_analysis.analysis.nltk_resources import ensure_nltk_resources
from steam_review_analysis.analysis.report import run_analysis

ensure_nltk_resources(download=True)  # one-off; a no-op once the corpora are installed

cleaned = sra.clean_reviews(sra.read_raw_reviews())
sra.write_processed(cleaned)

report = run_analysis(105600, "2024-01-01")
print(report.output_dir, sorted(report.tables))
```

Worked examples live in the Jupyter notebooks in
[`notebooks/`](https://github.com/G-a-b-b-s/AML_APP_Steam_Reviews/tree/main/notebooks), which
run on the review sample bundled with the repository (they are not part of the installed
package):

- [`01_quickstart`](https://github.com/G-a-b-b-s/AML_APP_Steam_Reviews/blob/main/notebooks/01_quickstart.ipynb):
  raw reviews → cleaned dataset → full report with one call, plus the CLI equivalents
- [`02_event_impact_across_games`](https://github.com/G-a-b-b-s/AML_APP_Steam_Reviews/blob/main/notebooks/02_event_impact_across_games.ipynb):
  the analysis step by step, comparing Counter-Strike, PUBG and The Sims 4 before and after a
  major change to each game
- [`03_sentiment_models`](https://github.com/G-a-b-b-s/AML_APP_Steam_Reviews/blob/main/notebooks/03_sentiment_models.ipynb):
  VADER and the transformer models scored against the reviewers' own recommendations
  (needs the `sentiment` extra)

## Commands

| Command | What it does | Extra |
| --- | --- | --- |
| `scrape [APP_ID ...]` | Scrape reviews into `<data-dir>/raw/<app_id>_reviews.csv` | `scrape` |
| `clean` | Validate, deduplicate and type all raw CSVs into `<data-dir>/processed/reviews.csv` | core |
| `analyse games` | List the games in the processed data | `analysis` |
| `analyse report` | Run the full analysis for one game and save every table and plot | `analysis` |
| `analyse preprocess \| sentiment \| frequencies \| wordcloud` | Run one analysis step | `analysis` |
| `analyse transformer-sentiment` | Score reviews with a small transformer model | `sentiment` |
| `download-nltk` | Download the NLTK corpora used by the analysis | `analysis` |

Global options work before or after the command name: `--data-dir PATH` and
`--log-level DEBUG|INFO|WARNING|ERROR`. Exit codes: `0` success, `1` workflow failure,
`2` invalid arguments, `130` interrupted with Ctrl+C.

## Data and configuration

**Data directory.** Data is never written next to the installed package. The directory is
`--data-dir` (or the `raw_dir=` / `processed_dir=` argument in Python), otherwise the
`STEAM_REVIEW_ANALYSIS_DATA_DIR` environment variable, otherwise `./data` in the current
working directory. Inside it: `raw/`, `processed/` and `processed/analysis/<game>/`. Files are
written to a temporary file first and swapped in, so an interrupted run never leaves a
half-written dataset.

**Game registry** (optional, `./configs/games.toml` or `--games FILE`): the games scraped when
no app ids are given, and the display names added during cleaning.

```toml
[[games]]
app_id = 105600          # the number in https://store.steampowered.com/app/<app_id>/
name = "Terraria"
```

**Analysis presets** (optional, `./configs/presets.toml` or `--presets FILE`): a default
cutoff and game-specific word rules, selected with `--preset NAME`.

```toml
[presets.PUBG]
app_id = 578080
cutoff = "2022-03-23"
preserve_keywords = ["pubg", "fpp", "tpp"]   # kept even if they look like stopwords

[presets.PUBG.replacements]                  # applied after lemmatization
battlegrounds = "pubg"
hacker = "cheat"
```

**NLTK corpora.** The `analyse` commands download missing corpora automatically. Library
functions never download anything themselves; they raise an error that names the missing
corpora. Install them once with `steam-review-analysis download-nltk`.

## Documentation

- [Scraping and cleaning](https://github.com/G-a-b-b-s/AML_APP_Steam_Reviews/blob/main/docs/data_pipeline.md):
  scraper options, failure behaviour and cleaning rules
- [Data schema](https://github.com/G-a-b-b-s/AML_APP_Steam_Reviews/blob/main/docs/data_schema.md):
  every column of the raw and processed CSVs
- [Analysis and sentiment models](https://github.com/G-a-b-b-s/AML_APP_Steam_Reviews/blob/main/docs/analysis.md):
  analysis commands, transformer models, comparing sentiment methods
- [API reference](https://g-a-b-b-s.github.io/AML_APP_Steam_Reviews/): generated from the
  docstrings
- [Contributing](https://github.com/G-a-b-b-s/AML_APP_Steam_Reviews/blob/main/CONTRIBUTING.md)
  and [changelog](https://github.com/G-a-b-b-s/AML_APP_Steam_Reviews/blob/main/CHANGELOG.md)

## Responsible use

- **Steam's terms.** The scraper reads public Steam Community pages with a real browser.
  You are responsible for complying with the
  [Steam Subscriber Agreement](https://store.steampowered.com/subscriber_agreement/) and for
  scraping at a modest rate; keep the default pauses and do not run many scrapers in parallel.
- **Personal data.** Scraped reviews are user-generated content. Store and share datasets
  accordingly, and do not republish them as your own.
- **Model licenses.** The default transformer model, `nano-jev`, has weights derived from
  MS MARCO, which is licensed for **non-commercial use only**. For commercial work choose an
  Apache-2.0 model such as `--model laya-english`. See
  [the model table](https://github.com/G-a-b-b-s/AML_APP_Steam_Reviews/blob/main/docs/analysis.md#transformer-sentiment).

## License

MIT; see [LICENSE](https://github.com/G-a-b-b-s/AML_APP_Steam_Reviews/blob/main/LICENSE).
