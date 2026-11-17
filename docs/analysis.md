# Analysis and sentiment models

Requires the `analysis` extra (transformer models additionally need `sentiment`). All commands
read `<data-dir>/processed/reviews.csv` and write to `<data-dir>/processed/analysis/<game>/`
unless `--output-dir` is given; [data_schema.md](data_schema.md) lists every output file.

## Commands

```bash
steam-review-analysis analyse games                 # games available in the processed data
steam-review-analysis analyse report --game 105600 --cutoff 2024-01-01
steam-review-analysis analyse preprocess --game 105600
steam-review-analysis analyse sentiment --game 105600
steam-review-analysis analyse frequencies --game 105600 --cutoff 2024-01-01 --top 20
steam-review-analysis analyse wordcloud --game "The Sims 4" --cutoff 2024-01-01
steam-review-analysis analyse report --game 578080 --preset PUBG   # cutoff from the preset
```

- `--game` takes an app id or a display name (case-insensitive).
- `--cutoff` splits reviews into *before* and *after* (reviews on the date count as after). It
  is required by `report`, `frequencies` and `wordcloud` unless a preset supplies it.
- `--preset NAME` applies a preset's word rules and default cutoff, read from
  `./configs/presets.toml` or `--presets FILE`; the format is shown in the README.
- `--min-word-freq` drops rare words from documents (default 2); `--top` sets how many words
  the frequency tables keep (default 20).
- Invalid games, cutoffs and limits fail with exit code 1 or 2 without touching existing data.

## Python API

The CLI is a thin layer over these functions, so notebooks can run the same steps one by one:

```python
from steam_review_analysis.analysis.dataset import load_reviews, split_by_cutoff
from steam_review_analysis.analysis.presets import load_presets
from steam_review_analysis.analysis.sentiment import add_sentiment, sentiment_summary
from steam_review_analysis.analysis.text import prepare_documents, word_frequencies

preset = load_presets("configs/presets.toml")["PUBG"]
reviews = load_reviews("PUBG: BATTLEGROUNDS")
prepared = prepare_documents(reviews, preset.preprocessing)
scored = add_sentiment(prepared.reviews)
before, after = split_by_cutoff(scored, preset.cutoff)
print(sentiment_summary(before), sentiment_summary(after))
print(word_frequencies(prepared.reviews["document"], top_n=10))
```

`steam_review_analysis.analysis.report.run_analysis(game, cutoff)` runs everything and returns
the in-memory tables plus the paths of all written files. Plot helpers in
`steam_review_analysis.analysis.plotting` return matplotlib figures, so notebooks can display
them directly. The project notebooks (`notebooks/*_EDA.ipynb`) explore the Counter-Strike,
PUBG and The Sims 4 datasets.

Text preprocessing needs NLTK corpora. Library functions raise an error naming any missing
corpus instead of downloading it; install them once with `steam-review-analysis download-nltk`
or `ensure_nltk_resources(download=True)` from `steam_review_analysis.analysis.nltk_resources`.

## Transformer sentiment

`steam_review_analysis.analysis.transformer_sentiment` adds optional Jev-style Hugging Face
decision models. They are non-generative: each review gets calibrated three-class
probabilities in one forward pass. Install with `pip install "steam-review-analysis[sentiment]"`;
the first use downloads the weights from Hugging Face and later runs reuse them.

| Model key | Parameters | Backend | Language | License |
| --- | --- | --- | --- | --- |
| `nano-jev` (default) | 22.7M | `nano-jev` | English | Apache-2.0 code; **weights non-commercial** (MS MARCO) |
| `laya-multilingual` | 322M | `laya` | 100+ languages | Apache-2.0 |
| `laya-multilingual-8bit` | 322M | `laya` + Quanto | 100+ languages | Apache-2.0 |
| `laya-english` | 421M | `laya` | English | Apache-2.0 |
| `laya-english-8bit` | 421M | `laya` + Quanto | English | Apache-2.0 |
| `tiny-jev-0.6b` | 596M | `transformers` | English | Apache-2.0 |
| `tiny-jev-0.6b-8bit` | 596M | `transformers` + Quanto | English | Apache-2.0 |

```bash
steam-review-analysis analyse transformer-sentiment --game 105600
steam-review-analysis analyse transformer-sentiment --game 105600 --model laya-english
steam-review-analysis analyse transformer-sentiment --game 105600 --device cpu --batch-size 8
```

From Python, the transformer columns can be added next to the VADER ones:

```python
from steam_review_analysis.analysis.transformer_sentiment import (
    add_transformer_sentiment,
    predict_sentiment,
    transformer_sentiment_summary,
)

scored = add_transformer_sentiment(add_sentiment(prepared.reviews), model="laya-english")
summary = transformer_sentiment_summary(scored)
single = predict_sentiment(["Great game, but the server lag is awful."])[0]
print(single.label, single.score, single.confidence)
```

Added columns: `transformer_score` (expected polarity `P(positive) - P(negative)` in
`[-1, 1]`), `transformer_category` (`Positive`, `Neutral` or `Negative`),
`transformer_confidence`, one probability column per class, and `transformer_model`. Empty
input returns an empty result without loading a model. Unknown model keys, non-positive batch
sizes and malformed probabilities raise `ValueError`; a missing optional dependency raises
`RuntimeError` with the install command. Loaded models are cached per process; free memory
between models with `clear_model_cache()`.

### Choosing a model

- `nano-jev` is the fastest. It is pinned to revision `v0.1`, whose generic option matching
  works better for custom sentiment labels than `v1.0`. Its weights are not licensed for
  commercial use.
- `laya-multilingual` trades speed for language coverage and more reliable neutral detection;
  `laya-english` is the English Laya checkpoint.
- `tiny-jev-0.6b` was trained on review sentiment and is the slowest unquantized option. It
  loads custom model code with `trust_remote_code=True`; the revision is pinned in `ModelSpec`
  so that code cannot change silently.

On a 20-review The Sims 4 sample, macro F1 against the recommendation labels was 0.55 for
`laya-english`, 0.48 for `laya-multilingual`, 0.43 for VADER, 0.41 for `tiny-jev-0.6b` and
0.29 for `nano-jev`. That sample is an illustration, not a benchmark.

Not included: `C-Tianyu/NanoJev` (a 0.6B game-action model, not sentiment-trained; the 0.6B slot
uses `lostargon/Tiny-Jev` instead) and `jaredpalmer/kev-0.5b` (a superseded prototype that is
not on PyPI). All models are applied zero-shot unless the checkpoint was trained for sentiment.

### Performance notes

- **Quantization.** The `-8bit` variants use Optimum-Quanto weight-only quantization after
  loading (Laya has no native option, and Tiny-Jev's native flag is MLX-only). Peak memory
  while loading equals the unquantized model; steady-state memory is lower. The ready-made
  `p-yan/laya-quanto` checkpoint was tested but its CUDA autocast path conflicts with the
  quantized weights.
- **Compilation.** With CUDA, unquantized models use `torch.compile` at load time. On a 4 GB
  GTX 1650 this took about 139 s for `laya-english`, 75 s for `laya-multilingual`, 28 s for
  `nano-jev` and 2 s for `tiny-jev-0.6b`. Quantized variants skip compilation.
- **Older GPUs.** GPUs below compute capability 8.0 have no native bf16. Laya selects fp16
  there, Tiny-Jev is cast to fp16, and Nano-Jev skips compilation because its library forces
  bf16 autocast.
- **Memory.** Measured peak VRAM on the 3.63 GiB GTX 1650: `laya-english-8bit` 1.64 GiB,
  `laya-multilingual-8bit` 1.25 GiB, `tiny-jev-0.6b-8bit` 1.40 GiB; unquantized with compile:
  1.60, 1.26, 1.15 and 0.11 GiB. Use `--device cpu` if a model does not fit.

## Comparing sentiment methods

`steam_review_analysis.analysis.sentiment_comparison` compares VADER and transformer
predictions against the Steam recommendation as binary truth (`Recommended` → positive,
`Not Recommended` → negative). It reports coverage, neutral rate, accuracy, macro
precision/recall/F1, weighted F1, Cohen's kappa, the Matthews correlation coefficient and a
three-class confusion matrix. Neutral predictions stay in the confusion matrix but are left
out of binary accuracy and F1, so methods that often abstain are not rewarded.

```python
from steam_review_analysis.analysis.sentiment_comparison import (
    compare_sentiment_methods,
    confusion_matrix,
    recommendation_truth,
)

# `scored` has both the VADER and the transformer columns (see above).
scored["recommendation_label"] = recommendation_truth(scored)
table = compare_sentiment_methods(scored, ["sentiment_category", "transformer_category"])
matrix = confusion_matrix(scored["recommendation_label"], scored["transformer_category"])
```
