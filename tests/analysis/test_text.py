"""Test preprocessing, vocabulary filtering and word-frequency utilities."""

from types import MappingProxyType

import pandas as pd
import pytest

from steam_review_analysis.analysis.text import (
    PreprocessingConfig,
    build_vocabulary,
    filter_tokens,
    prepare_documents,
    preprocess_text,
    word_frequencies,
    wordnet_pos,
)


@pytest.mark.parametrize(
    ("tag", "expected"),
    [("JJ", "a"), ("VBG", "v"), ("NN", "n"), ("RB", "r"), ("PRP", "n")],
)
def test_wordnet_pos_maps_treebank_tags(tag: str, expected: str) -> None:
    """Map adjective, verb, noun and adverb tags to WordNet parts of speech."""
    assert wordnet_pos(tag) == expected


@pytest.mark.usefixtures("nltk_corpora")
def test_preprocess_text_normalizes_and_applies_domain_rules() -> None:
    """Lowercase, strip punctuation, lemmatize and apply replacements and preserved words."""
    config = PreprocessingConfig(
        preserve_keywords=frozenset({"cs"}),
        replacements=MappingProxyType({"cheater": "cheat", "csgo": "cs"}),
    )

    assert preprocess_text("Cheater plays CSGO and CS!", config) == "cheat play cs cs"


@pytest.mark.usefixtures("nltk_corpora")
def test_preprocess_text_removes_stopwords_short_tokens_and_numbers() -> None:
    """Drop stopwords, one-character tokens and numeric tokens."""
    assert preprocess_text("I have 2 a game") == "game"


@pytest.mark.usefixtures("nltk_corpora")
def test_preprocess_text_returns_empty_for_blank_input() -> None:
    """Return an empty string when the review has no usable text."""
    assert preprocess_text("   ...   ") == ""


def test_preprocess_config_rejects_invalid_length() -> None:
    """Reject a minimum token length below one."""
    with pytest.raises(ValueError, match="min_word_length"):
        PreprocessingConfig(min_word_length=0)


def test_build_vocabulary_filters_by_frequency() -> None:
    """Keep only tokens meeting the requested corpus frequency."""
    assert build_vocabulary([["a", "b"], ["b", "c"]], 2) == frozenset({"b"})


def test_build_vocabulary_rejects_invalid_frequency() -> None:
    """Reject a nonpositive minimum frequency."""
    with pytest.raises(ValueError, match="min_word_freq"):
        build_vocabulary([["a"]], 0)


def test_filter_tokens_preserves_order() -> None:
    """Remove out-of-vocabulary tokens without reordering the rest."""
    assert filter_tokens(["b", "a", "c"], frozenset({"b", "c"})) == ["b", "c"]


@pytest.mark.usefixtures("nltk_corpora")
def test_prepare_documents_adds_processed_text_and_vocabulary(sample_frame: pd.DataFrame) -> None:
    """Prepare documents, filter rare tokens and drop rows with no processed text."""
    frame = pd.concat(
        [
            sample_frame,
            pd.DataFrame(
                [
                    [3, "Gamma", "2024-01-01", "the and of", 1.0, "Recommended", 0, True],
                ],
                columns=sample_frame.columns,
            ),
        ],
        ignore_index=True,
    )
    config = PreprocessingConfig(
        preserve_keywords=frozenset({"game"}),
        replacements=MappingProxyType({"cheater": "cheat"}),
    )

    prepared = prepare_documents(frame, config, min_word_freq=2)

    assert "processed_review" in prepared.reviews.columns
    assert "document" in prepared.reviews.columns
    assert 3 not in prepared.reviews["app_id"].tolist()
    assert "game" in prepared.vocabulary
    assert all(
        token in prepared.vocabulary
        for document in prepared.reviews["document"]
        for token in document.split()
    )


def test_prepare_documents_rejects_missing_text_column(sample_frame: pd.DataFrame) -> None:
    """Report a missing text column with a clear error."""
    with pytest.raises(KeyError, match="content"):
        prepare_documents(sample_frame, text_column="content")


def test_word_frequencies_counts_lowercase_words() -> None:
    """Count words case-insensitively while dropping punctuation."""
    result = word_frequencies(["Hello hello, WORLD!", None])

    assert result.to_dict("records") == [
        {"word": "hello", "frequency": 2},
        {"word": "world", "frequency": 1},
    ]


def test_word_frequencies_limits_results() -> None:
    """Return only the most frequent requested words."""
    result = word_frequencies(["a b b c c c"], top_n=2)

    assert result["word"].tolist() == ["c", "b"]


def test_word_frequencies_rejects_invalid_limit() -> None:
    """Reject a nonpositive result limit."""
    with pytest.raises(ValueError, match="top_n"):
        word_frequencies(["text"], top_n=0)
