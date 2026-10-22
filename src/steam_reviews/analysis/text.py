"""Text preprocessing, tokenization, vocabulary filtering and word counts."""

import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from types import MappingProxyType

import pandas as pd
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from steam_reviews.analysis.nltk_resources import ensure_nltk_resources

DEFAULT_EXTRA_STOPWORDS = frozenset(
    {
        "o",
        "etc",
        "would",
        "could",
        "us",
        "the",
        "to",
        "of",
        "in",
        "it",
        "and",
        "this",
        "that",
        "for",
    }
)
_NON_WORD = re.compile(r"[^\w\s]")
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class PreprocessingConfig:
    """Control lemmatization, stopwords and domain-specific word normalization.

    Attributes:
        preserve_keywords: Lowercase tokens kept unchanged even if they are stopwords.
        replacements: Lemmatized-token replacements, such as `cheater` to `cheat`.
        extra_stopwords: Additional lowercase stopwords removed after lemmatization.
        min_word_length: Shortest token length that is retained.
    """

    preserve_keywords: frozenset[str] = frozenset()
    replacements: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
    extra_stopwords: frozenset[str] = DEFAULT_EXTRA_STOPWORDS
    min_word_length: int = 2

    def __post_init__(self) -> None:
        """Validate the minimum token length.

        Raises:
            ValueError: The minimum length is below one.
        """
        if self.min_word_length < 1:
            raise ValueError("min_word_length must be at least 1")


@dataclass(frozen=True, slots=True)
class PreparedDocuments:
    """Hold reviews enriched with processed text and the shared vocabulary.

    Attributes:
        reviews: Copy of the input with `processed_review` and `document` columns.
        vocabulary: Tokens occurring at least the requested number of times.
    """

    reviews: pd.DataFrame
    vocabulary: frozenset[str]


def wordnet_pos(treebank_tag: str) -> str:
    """Map an NLTK Part-of-Speech tag to a WordNet part of speech.

    Args:
        treebank_tag: Penn Treebank tag such as `NN`, `VBG` or `JJR`.

    Returns:
        WordNet noun, verb, adjective or adverb tag; noun for unknown tags.
    """
    if treebank_tag.startswith("J"):
        return "a"
    if treebank_tag.startswith("V"):
        return "v"
    if treebank_tag.startswith("N"):
        return "n"
    if treebank_tag.startswith("R"):
        return "r"
    return "n"


@lru_cache(maxsize=1)
def _english_stopwords() -> frozenset[str]:
    """Return the cached English NLTK stopword set.

    Returns:
        Stopwords shared by every preprocessing call.
    """
    return frozenset(stopwords.words("english"))


def preprocess_text(text: str, config: PreprocessingConfig | None = None) -> str:
    """Normalize a review into lowercase lemmatized tokens.

    Args:
        text: Review body in any case, optionally containing punctuation.
        config: Domain keywords and replacements; defaults to no domain rules.

    Returns:
        Space-separated lemmatized tokens, or an empty string when no token remains.
    """
    settings = config or PreprocessingConfig()
    ensure_nltk_resources()
    cleaned = _NON_WORD.sub(" ", text)
    cleaned = _WHITESPACE.sub(" ", cleaned).strip().lower()
    if not cleaned:
        return ""

    tokens = word_tokenize(cleaned)
    stop_words = _english_stopwords() | settings.extra_stopwords
    lemmatizer = WordNetLemmatizer()
    normalized: list[str] = []
    for word, tag in pos_tag(tokens):
        if word in settings.preserve_keywords:
            normalized.append(word)
        elif (
            word not in stop_words
            and len(word) >= settings.min_word_length
            and not word.isnumeric()
        ):
            base = lemmatizer.lemmatize(word, wordnet_pos(tag))
            normalized.append(settings.replacements.get(base, base))
    return " ".join(normalized)


def tokenize(text: str) -> list[str]:
    """Split processed text into NLTK word tokens.

    Args:
        text: Whitespace-separated processed review text.

    Returns:
        Tokens in their original order.
    """
    ensure_nltk_resources()
    tokens: list[str] = word_tokenize(text)
    return tokens


def build_vocabulary(
    token_documents: Iterable[Sequence[str]],
    min_word_freq: int = 2,
) -> frozenset[str]:
    """Collect tokens occurring at least a minimum number of times.

    Args:
        token_documents: One sequence of tokens per review.
        min_word_freq: Smallest frequency retained in the vocabulary.

    Returns:
        Set of retained tokens.

    Raises:
        ValueError: The minimum frequency is below one.
    """
    if min_word_freq < 1:
        raise ValueError("min_word_freq must be at least 1")
    frequencies: Counter[str] = Counter()
    for tokens in token_documents:
        frequencies.update(tokens)
    return frozenset(word for word, count in frequencies.items() if count >= min_word_freq)


def filter_tokens(tokens: Sequence[str], vocabulary: frozenset[str]) -> list[str]:
    """Remove tokens outside a shared vocabulary.

    Args:
        tokens: Tokens from one review.
        vocabulary: Tokens retained across the corpus.

    Returns:
        Tokens present in the vocabulary, in their original order.
    """
    return [token for token in tokens if token in vocabulary]


def prepare_documents(
    reviews: pd.DataFrame,
    config: PreprocessingConfig | None = None,
    *,
    text_column: str = "text",
    min_word_freq: int = 2,
) -> PreparedDocuments:
    """Preprocess, tokenize and vocabulary-filter every review.

    Args:
        reviews: Processed reviews from `steam_reviews.io.read_processed`.
        config: Domain keywords and replacements; defaults to no domain rules.
        text_column: Column containing the original review text.
        min_word_freq: Smallest token frequency retained in documents.

    Returns:
        A copy of `reviews` with `processed_review` and `document` columns, plus the
        vocabulary. Rows whose processed text becomes empty are removed.

    Raises:
        KeyError: The requested text column does not exist.
        ValueError: The minimum frequency is below one.
    """
    if text_column not in reviews.columns:
        raise KeyError(f"Missing review text column: {text_column}")
    df = reviews.copy()
    processed = [preprocess_text(str(text), config) for text in df[text_column].fillna("")]
    df["processed_review"] = pd.Series(processed, index=df.index, dtype=object)
    df = df.loc[df["processed_review"].str.len() > 0].reset_index(drop=True)
    token_documents = [tokenize(text) for text in df["processed_review"]]
    vocabulary = build_vocabulary(token_documents, min_word_freq)
    documents = [" ".join(filter_tokens(tokens, vocabulary)) for tokens in token_documents]
    df["document"] = pd.Series(documents, index=df.index, dtype=object)
    return PreparedDocuments(reviews=df, vocabulary=vocabulary)


def word_frequencies(texts: Iterable[object], *, top_n: int | None = None) -> pd.DataFrame:
    """Count lowercase word occurrences across texts.

    Args:
        texts: Review texts or processed documents.
        top_n: Optional limit on the number of most frequent words returned.

    Returns:
        Frame with `word` and `frequency` columns, ordered by descending frequency.

    Raises:
        ValueError: A nonpositive `top_n` is requested.
    """
    if top_n is not None and top_n < 1:
        raise ValueError("top_n must be at least 1")
    counts: Counter[str] = Counter()
    for value in texts:
        if value is None or value is pd.NA:
            continue
        if isinstance(value, float) and value != value:
            continue
        cleaned = _NON_WORD.sub(" ", str(value).lower())
        counts.update(cleaned.split())
    items = counts.most_common(top_n)
    return pd.DataFrame(items, columns=["word", "frequency"])
