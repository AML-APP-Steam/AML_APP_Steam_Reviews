# Data schema

## Raw CSVs: `<data-dir>/raw/<app_id>_reviews.csv`

One file per game, written by `scrape`. Values are stored exactly as shown on the review page;
parsing into numbers happens during cleaning.

| Column | Meaning | Where it appears on Steam | Example |
| --- | --- | --- | --- |
| `Game` | Steam app id of the reviewed game | The number in the store URL | `105600` |
| `Date` | Day the review was posted (ISO date) | Grey "Posted: …" line | `2024-05-10` |
| `Review` | Review text, line breaks collapsed to spaces | Main paragraph of the review card | `Great co-op…` |
| `Hours Played` | Hours on record when the review was written, as shown | "123.4 hrs on record" | `1,234.5` |
| `Recommendation` | The reviewer's verdict | Title next to the thumb icon | `Recommended` |
| `Helpful Votes` | Helpful-vote label as shown, or `No helpful votes` | "X people found this review helpful" | `56 people found this review helpful` |

## Processed CSV: `<data-dir>/processed/reviews.csv`

One file for all games, written by `clean`. Read it with
`steam_review_analysis.read_processed()`, which restores the types below.

| Column | Type | Meaning |
| --- | --- | --- |
| `app_id` | int | Steam app id |
| `game` | string, may be empty | Display name from the game registry (only when a registry was used) |
| `date` | datetime | Day the review was posted |
| `text` | string | Review text with outer whitespace removed |
| `hours_played` | float, may be NaN | Hours on record |
| `recommendation` | `Recommended` / `Not Recommended` | The reviewer's verdict |
| `helpful_votes` | int | Number of helpful votes (0 when none) |
| `is_recommended` | bool | `recommendation == "Recommended"` |

## Analysis outputs: `<data-dir>/processed/analysis/<game>/`

Written by the `analyse` commands; `report` writes all of them except `transformer_sentiment.csv`,
which comes from `analyse transformer-sentiment`.

| File | Contents |
| --- | --- |
| `documents.csv`, `metadata.csv` | Preprocessed review documents and their metadata (e.g. for topic models) |
| `word_frequencies_{overall,before,after}.csv` | Most frequent words overall and before/after the cutoff |
| `sentiment.csv` | Reviews with VADER scores, sentiment category, year and period |
| `transformer_sentiment.csv` | Reviews with transformer scores and class probabilities |
| `correlations.csv` | Pearson correlations of sentiment with hours played and helpful votes, before and after the cutoff |
| `summary.json` | Counts and sentiment summaries overall and per period |
| `*.png` | Word clouds (overall, per period, per year), word-frequency charts, sentiment distributions and fits, yearly trends and engagement plots |
