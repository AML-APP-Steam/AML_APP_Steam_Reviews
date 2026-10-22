# %% [markdown]
# # Counter-Strike review analysis
#
# This notebook analyses the processed Steam reviews for Counter-Strike (CS:GO/CS2).
# All reusable logic lives in `steam_reviews.analysis`, so the notebook only selects the
# game, runs the workflow and inspects the tables and figures.
#
# Set `STEAM_REVIEWS_GAME` to analyse a different processed game, for example
# `STEAM_REVIEWS_GAME=Terraria uv run python notebooks/PROJECT_Counter_Strike.py`.
# Set `STEAM_REVIEWS_CUTOFF` to override the before/after date.

# %%
import os

import matplotlib.pyplot as plt
import pandas as pd

from steam_reviews.analysis.presets import preset_for
from steam_reviews.analysis.report import run_analysis

pd.set_option("display.max_colwidth", 120)

PRESET = preset_for("Counter_Strike")
GAME = os.environ.get("STEAM_REVIEWS_GAME", str(PRESET.app_id))
CUTOFF = os.environ.get("STEAM_REVIEWS_CUTOFF", PRESET.cutoff)
CONFIG = PRESET.preprocessing if GAME in {str(PRESET.app_id), PRESET.name} else None

# %% [markdown]
# ## Run the analysis
#
# The workflow preprocesses text, builds documents and metadata, computes word frequencies,
# scores sentiment with VADER, compares the periods and writes every artifact under
# `data/processed/analysis/<game>/`.

# %%
report = run_analysis(GAME, CUTOFF, config=CONFIG)

print(f"Game: {report.game} (app {report.app_id})")
print(f"Cutoff: {report.cutoff.date()}")
print(f"Reviews analysed: {len(report.prepared.reviews)}")
print(f"Vocabulary size: {len(report.prepared.vocabulary)}")
print(f"Artifacts: {report.output_dir}")

# %% [markdown]
# ## Summary statistics

# %%
for period in ("overall", "before", "after"):
    print(f"\n{period.title()}:")
    for name, value in report.summary[period].items():
        print(f"  {name}: {value:.4f}")

# %% [markdown]
# ## Processed documents
#
# `processed_review` contains lemmatized tokens; `document` keeps only tokens that occur
# at least twice in the corpus. Documents and metadata are also written as CSVs for topic
# models such as STM.

# %%
report.prepared.reviews[["date", "text", "processed_review", "document"]].head(10)

# %% [markdown]
# ## Word frequencies

# %%
for period, table in report.frequencies.items():
    print(f"\nTop words ({period}):")
    print(table.head(20).to_string(index=False))

# %% [markdown]
# ## Sentiment and metadata
#
# The correlations compare sentiment with playtime and helpful votes before and after
# the cutoff.

# %%
print(report.correlations.to_string(index=False))
print()
print(
    report.sentiment[["date", "sentiment_score", "sentiment_category", "year", "period"]].head(10)
)

# %% [markdown]
# ## Generated figures
#
# Every figure is saved as a PNG. The loop below also renders the saved images inline when
# the notebook runs with a display.

# %%
for name, path in sorted(report.figures.items()):
    print(f"{name}: {path}")
    image = plt.imread(path)
    figure, axes = plt.subplots(figsize=(10, 5))
    axes.imshow(image)
    axes.axis("off")
    axes.set_title(name)
    plt.show()
    plt.close(figure)

# %% [markdown]
# ## Written tables

# %%
for name, path in sorted(report.tables.items()):
    print(f"{name}: {path}")
