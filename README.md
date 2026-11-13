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
    scraping/               browser, parsers, collector and public scraping workflow
    processing/             pandas only: raw -> typed, deduplicated data
    analysis/               preprocessing, VADER and transformer sentiment, statistics, plots and the analysis CLI
    config.py  schema.py    paths, game registry, column names, Review record
    io.py                   reading and writing the CSVs
    cli.py                  `steam-reviews scrape | clean | analyse`
tests/
```

Data flows one way: `scraping` writes `data/raw` -> `processing` writes `data/processed` ->
notebooks read it with `steam_reviews.io.read_processed()`. `scraping` and `processing` never
import each other; both depend only on `schema`, `config` and `io`.

## Setup

Requires Python 3.11+, [uv](https://docs.astral.sh/uv/) and Google Chrome.
`pyproject.toml` defines dependencies, tool configuration and the command entry point;
`uv.lock` pins the resolved dependency versions. Ruff handles both linting and formatting;
there is no separate Black configuration.

From the repository root, on Windows PowerShell or a Unix shell:

```bash
uv sync --locked
uv run pre-commit install
```

If `make` is installed, the existing shortcuts are also available:

```bash
make install          # uv sync + install git hooks
uv sync --group notebooks   # optional: JupyterLab
uv sync --extra sentiment   # optional: Jev-style transformer sentiment models
```

## Usage

```bash
uv run steam-reviews scrape                       # every game in configs/games.toml
uv run steam-reviews scrape 105600 --target 1000  # one game, more reviews
uv run steam-reviews scrape --headless
uv run steam-reviews scrape 105600 --headless --overwrite  # explicitly replace an existing CSV
uv run steam-reviews --log-level DEBUG scrape 105600      # detailed diagnostics
uv run steam-reviews scrape --help                       # all parameters and their defaults
uv run steam-reviews clean                        # data/raw -> data/processed/reviews.csv
```

To add a game, append an entry to `configs/games.toml`.

### Scraper parameters

`ScraperSettings` in `src/steam_reviews/scraping/scraper.py` is the source of default scraping
parameters for both the CLI and Python API. Its docstring explains each parameter; invalid
counts, negative waits, zero timeouts, and non-finite times are rejected before Chrome starts.

| CLI option | Default | Meaning |
| --- | --- | --- |
| `--target` | `600` | Maximum unique reviews per game. |
| `--filter` | `toprated` | Steam review ordering; use `mostrecent` for recent reviews. |
| `--language` | `english` | Review content language; UI labels are requested in English. |
| `--scroll-pause` | `2.0` | Seconds to allow cards to load after a scroll. |
| `--wait-timeout` | `5.0` | Maximum seconds waiting for the page body. |
| `--page-load-timeout` | `30.0` | Maximum seconds for navigation. |
| `--max-stalled-scrolls` | `5` | Consecutive scrolls without new valid reviews before stopping. |
| `--max-stale-retries` | `3` | Consecutive stale-page errors before failing. |

### Calling the scraper from Python or a notebook

Keep orchestration in one call; open the function definitions for implementation details:

```python
import logging

from steam_reviews.scraping.scraper import ScraperSettings
from steam_reviews.scraping.service import scrape_games

logging.basicConfig(level=logging.INFO)

settings = ScraperSettings(target_reviews=100, scroll_pause_s=2.0)
saved_paths = scrape_games([105600], settings=settings, headless=True)
```

The input is a sequence of Steam app IDs and optional settings. The output is a
`dict[int, pathlib.Path]` mapping each app ID to its raw CSV. Repeated IDs are scraped once.
The function manages Chrome's lifecycle and logs progress; the low-level
`ReviewScraper.scrape(app_id)` returns `list[Review]` when records are needed without saving.

### Output and failure behavior

- Raw files remain `data/raw/<app_id>_reviews.csv`, with the unchanged columns
  `Game`, `Date`, `Review`, `Hours Played`, `Recommendation`, and `Helpful Votes`.
- Existing files are protected by default. Use `--overwrite` or `overwrite=True` only when
  replacing a dataset is intended. All destinations are checked before the browser starts.
- CSVs are staged beside their destination and published only after writing completes.
  Browser failures, empty results, exhausted stale retries, and interrupted writes preserve
  the old CSV. Create-only publication uses a hard link and fails safely if the filesystem
  does not support it; explicit replacement uses an atomic rename.
- If loading stops after collecting some valid reviews, the smaller sample is saved with a
  warning. The target is an upper bound, not a guarantee that enough reviews exist.
- A batch stops on the first failure; CSVs successfully saved for earlier games remain.
  Exit codes are `0` for success, `1` for a workflow failure, and `2` for invalid arguments.
- English dates accept full/abbreviated month names, either day/month order, and a missing
  year. A missing year uses the current year; archived labels can be parsed with an explicit
  `reference_date` in `parse_review_date`. Deduplication uses all raw review fields, not a
  persistent Steam review ID.

The CLI configures logging once;  notebooks and other callers control their own logging setup. Pure text parsing remains in
`scraping/parsers.py`, Selenium access in `driver.py` and `scraper.py`, orchestration in
`scraping/service.py`, and filesystem writes in `io.py`.

### Cleaning behavior

`steam-reviews clean` calls `processing.cleaning.clean_reviews(raw, games)` and writes the
result through `io.py`. The same function accepts a pandas DataFrame in a notebook and
returns a new DataFrame without modifying its input. All six raw columns are required;
additional metadata columns are ignored.

- Empty review text and exact duplicates are removed, with counts recorded in logs.
- Text keeps its case, punctuation and internal whitespace. Literal reviews such as `NA`,
  `NULL` and `001` are preserved when reading raw and processed CSVs.
- IDs must be positive whole numbers. Dates accept ISO dates and timestamps from older
  exports. Missing or invalid IDs/dates cause a clear validation error.
- Recommendation labels are normalized for outer whitespace and case. Unknown labels cause
  an error instead of becoming negative training examples.
- Hours are always floats; missing, invalid, negative or infinite values become NaN.
  Helpful-vote labels yield integer counts; labels without a number retain the zero convention.
- When a game mapping is supplied, unknown game names remain missing and are logged.
- Invalid input or an empty cleaned result ends the command with exit code 1 before writing,
  preserving the existing processed dataset.

## Analysis and notebooks

The `analysis` package reads `data/processed/reviews.csv` and reuses one workflow from both the
CLI and the notebooks. It preprocesses text with NLTK, builds documents and metadata, counts
words, scores sentiment with VADER, compares before/after periods and writes CSVs, JSON and PNGs
under `data/processed/analysis/<game>/`.

```bash
uv run steam-reviews analyse games                 # games available in processed data
uv run steam-reviews analyse report --game 105600 --cutoff 2024-01-01
uv run steam-reviews analyse preprocess --game 105600 --cutoff 2024-01-01
uv run steam-reviews analyse sentiment --game 105600 --cutoff 2024-01-01
uv run steam-reviews analyse frequencies --game 105600 --cutoff 2024-01-01 --top 20
uv run steam-reviews analyse wordcloud --game 105600 --cutoff 2024-01-01
```

`--preset Counter_Strike|PUBG|SIMS` applies the game-specific keywords, replacements and default
cutoff. `--output-dir` overrides the artifact directory. Invalid games, cutoffs and limits are
reported with exit code 1 or 2 without touching processed data.

The notebook in `notebooks/sample_EDA_on_reviews.ipynb` is Python notebook that demonstrates
the analysis utilities step by step: loading reviews, preprocessing text, word frequencies, word
clouds, VADER and transformer sentiment scoring, period comparisons, engagement splits and
correlations. Each plot is created and displayed in the cell that computes it.

Edit the `GAME` and `CUTOFF` values in its setup cell to analyse another game or move the
before/after boundary.

Missing NLTK corpora are downloaded on first use.

### Transformer sentiment

`transformer_sentiment.py` adds optional Jev-style Hugging Face decision models.
They are non-generative, so each review gets calibrated three-class probabilities in one
forward pass instead of generated text:

| Model key | Parameters | Backend | Language | License |
| --- | --- | --- | --- | --- |
| `nano-jev` (default) | 22.7M | `nano-jev` | English | Apache-2.0 + MS MARCO non-commercial note |
| `laya-multilingual` | 322M | `laya` | 100+ languages | Apache-2.0 |
| `laya-multilingual-8bit` | 322M | `laya` + Quanto | 100+ languages | Apache-2.0 |
| `laya-english` | 421M | `laya` | English | Apache-2.0 |
| `laya-english-8bit` | 421M | `laya` + Quanto | English | Apache-2.0 |
| `tiny-jev-0.6b` | 596M | `transformers` | English | Apache-2.0 |
| `tiny-jev-0.6b-8bit` | 596M | `transformers` + Quanto | English | Apache-2.0 |

`nano-jev` is the fastest option and is pinned to version `v0.1` because its generic option
matching works better for custom sentiment labels than `v1.0`; its MS MARCO lineage makes
the weights unsuitable for commercial use. `laya-multilingual` trades speed for multilingual
coverage and more reliable neutral detection. `laya-english` is the English Laya checkpoint.
`tiny-jev-0.6b` is a 0.6B model trained on review sentiment and is the slowest unquantized
option.

On a 20-review Sims 4 sample, macro F1 against recommendation labels was 0.55 for
`laya-english`, 0.48 for `laya-multilingual`, 0.43 for VADER, 0.41 for `tiny-jev-0.6b` and
0.29 for `nano-jev`. That sample is only an illustration, not a benchmark.

The 22.7M `sdmlai/nano-jev` is a different model from the 0.6B `C-Tianyu/NanoJev`. The latter
is a game-action model and is not sentiment-trained, so it is not included. The 0.6B slot
here uses `lostargon/Tiny-Jev` instead. All models are applied zero-shot unless a checkpoint
was already trained for sentiment. `jaredpalmer/kev-0.5b` is not included: it is a superseded
prototype that is not published to PyPI and needs the Kev source checkout or a separate HTTP
server.

`tiny-jev-0.6b` loads custom model code with `trust_remote_code=True`. The revision is pinned
in `ModelSpec` so the code cannot change silently.

The `-8bit` variants use Optimum-Quanto weight-only quantization after loading. Laya has no
native `quantize` argument in the installed release, and Tiny-Jev's native `--quantize 8`
flag is currently MLX-only, so both are quantized through Optimum-Quanto. The ready-made
`p-yan/laya-quanto` checkpoint was tested, but its CUDA autocast path conflicts with the
quantized weights, so the English variant uses runtime Quanto for consistent CPU and CUDA
behaviour. Quantization happens after the full weights are loaded, so its peak memory is the
same as the unquantized model, but steady-state memory is lower.

When CUDA is available, unquantized Laya, Tiny-Jev and Nano-Jev models use `torch.compile`
automatically. Compilation happens at load time; on the 4GB GTX 1650 used here it took about
139s for `laya-english`, 75s for `laya-multilingual`, 28s for `nano-jev` and 2s for
`tiny-jev-0.6b`. Quantized variants skip compile. Load one model at a time with
`clear_model_cache()` and pass `--device cpu` if a checkpoint does not fit.

GPUs below compute capability 8.0, such as the GTX 1650, do not support native bf16
compilation. Laya already selects fp16 there. Tiny-Jev is cast to fp16 before
quantization or compile, and Nano-Jev skips compile because its library forces bf16
autocast. Inductor's low-SM GEMM autotune notice is disabled and filtered.

Measured peak VRAM on the 3.63GiB GTX 1650: `laya-english-8bit` 1.64GiB,
`laya-multilingual-8bit` 1.25GiB, `tiny-jev-0.6b-8bit` 1.40GiB. Unquantized models fit too
with compile, peaking at 1.60GiB, 1.26GiB, 1.15GiB and 0.11GiB respectively.

Install the optional dependencies with `uv sync --extra sentiment`. The first call downloads
the weights from Hugging Face; later calls reuse them. The base install and the VADER
analysis workflow do not require either package.

```bash
uv run steam-reviews analyse transformer-sentiment --game 105600
uv run steam-reviews analyse transformer-sentiment --game 105600 --model laya-multilingual
uv run steam-reviews analyse transformer-sentiment --game 105600 --device cpu --batch-size 8
```

The command writes `transformer_sentiment.csv` beside the other analysis artifacts. From
Python, keep the existing VADER columns and append the transformer columns:

```python
from steam_reviews.analysis.sentiment import add_sentiment
from steam_reviews.analysis.transformer_sentiment import (
    add_transformer_sentiment,
    predict_sentiment,
    transformer_sentiment_summary,
)

scored = add_transformer_sentiment(add_sentiment(prepared.reviews))
summary = transformer_sentiment_summary(scored)
single = predict_sentiment(["Great game, but the server lag is awful."])[0]
print(single.label, single.score, single.confidence)
```

Added columns are `transformer_score` (expected polarity `P(positive) - P(negative)` in
`[-1, 1]`), `transformer_category` (`Positive`, `Neutral` or `Negative`),
`transformer_confidence`, one probability column per class, and `transformer_model`.
`predict_sentiment` returns labels and probabilities; `sentiment_scores` returns only the
polarity scores. Empty input returns an empty result without loading a model.

Unknown model keys, non-positive batch sizes and malformed probability distributions raise
`ValueError`. A missing optional dependency raises `RuntimeError` with the install hint.
Model weights are cached per process, so notebook and CLI calls with the same model and
device reuse one loaded checkpoint.

### Comparing sentiment methods

`sentiment_comparison.py` compares VADER and transformer predictions against the Steam
recommendation as binary truth (`Recommended` -> positive, `Not Recommended` -> negative).
It reports coverage, neutral rate, accuracy, macro precision/recall/F1, weighted F1,
Cohen's kappa, Matthews correlation coefficient and a three-class confusion matrix. Neutral
predictions stay in the confusion matrix but are excluded from binary accuracy and F1, so
methods that abstain often are not rewarded.

```python
from steam_reviews.analysis.sentiment_comparison import (
    compare_sentiment_methods,
    confusion_matrix,
    recommendation_truth,
)

scored["recommendation_label"] = recommendation_truth(scored)
table = compare_sentiment_methods(scored, ["vader", "transformer"])
matrix = confusion_matrix(scored["recommendation_label"], scored["transformer"])
```

The notebook section uses a 20-review sample, caches each model's scores under
`data/processed/analysis/<game>/sentiment_cache/`, and clears the model cache between runs.
Rerunning the notebook reuses the cached CSVs instead of rescoring the reviews.

## API documentation

Generated from docstrings (Google style) and type hints with [pdoc](https://pdoc.dev):

```bash
make docs         # live preview in the browser
make docs-build   # static HTML in artifacts/docs/ (gitignored)
```

Every public module, class, function and method needs a docstring; ruff's `D` rules enforce it.

Tests also have docstrings describing the scenario and expected behavior. Fixtures and test
helpers document their arguments and returned or yielded data where applicable. Ordinary test
functions use short descriptions without repeating pytest fixture arguments in an `Args` section.
Ruff checks test documentation as well; only that argument-section requirement is exempted.

## Quality checks

```bash
make check    # ruff (lint + format check), mypy --strict, pytest with coverage
make format   # auto-fix lint and formatting
```

Without `make`, run the same checks directly (including on Windows):

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```

Tests use simulated browsers and temporary directories; they do not contact Steam or
overwrite repository datasets.
After hook installation, pre-commit runs Ruff, mypy, file hygiene checks and `uv lock`
on commits, and pytest on push. CI checks the lockfile, linting, formatting, types and tests
on pull requests and pushes to main, using Python 3.11 and 3.13.
