# Changelog

All notable changes are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-11-19

First packaged release.

## [0.1.1] - 2026-11-23

### Fixed
- Require nltk 3.9.1, since 3.9.0 fails to import without the wordnet corpus.

### Changed
- Replaced the per-game EDA notebooks with three example notebooks that use the package.
- CONTRIBUTING.md: repository layout as a tree, import rule and test structure.

### Added

- `steam-review-analysis` command with `scrape`, `clean`, `analyse` and `download-nltk`
  subcommands, `--version`, and global `--data-dir` / `--log-level` options accepted before or
  after the command name.
- Selenium scraper for Steam Community reviews with validated settings and safe,
  create-only CSV publication.
- Cleaning of raw reviews into a typed, deduplicated dataset.
- Analysis: NLTK preprocessing, word frequencies and word clouds, VADER sentiment,
  before/after statistics, plots and a full `report` workflow.
- Optional Jev-style transformer sentiment models and a comparison of sentiment methods
  against Steam recommendations.
- Analysis presets loaded from TOML files (`--preset`, `--presets`).
- Optional extras: `scrape`, `analysis`, `sentiment` and `all`; the core needs only pandas.
- Type information (`py.typed`) and a top-level API (`import steam_review_analysis`).

### Changed

- Data is read from and written to `--data-dir`, `$STEAM_REVIEW_ANALYSIS_DATA_DIR` or
  `./data`, never relative to the installed package.
- Processed data and analysis outputs are written atomically.
- Library code no longer downloads NLTK corpora implicitly; the CLI does, and
  `download-nltk` installs them explicitly.

### Fixed

- Zipped NLTK corpora (`wordnet`, `vader_lexicon`) were reported as missing, which triggered a
  download check on every call.
- Minimum versions: `matplotlib>=3.9` and `wordcloud>=1.9.3` (older versions fail with the
  plotting code).
