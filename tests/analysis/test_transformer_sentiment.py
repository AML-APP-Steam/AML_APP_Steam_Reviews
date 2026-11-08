"""Test the small Jev-style transformer sentiment tooling."""

from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from steam_reviews import cli
from steam_reviews.analysis import cli as analysis_cli
from steam_reviews.analysis import transformer_sentiment
from steam_reviews.analysis.transformer_sentiment import (
    DEFAULT_MODEL,
    MODEL_SPECS,
    SENTIMENT_LABELS,
    _LayaBackend,
    _NanoJevBackend,
    add_transformer_sentiment,
    available_models,
    model_spec,
    predict_sentiment,
    sentiment_scores,
    transformer_sentiment_summary,
)


class FakeBackend:
    """Return deterministic three-class probabilities for every text."""

    def __init__(self, probabilities: dict[str, float] | None = None) -> None:
        """Store the probability distribution returned for each text.

        Args:
            probabilities: Optional distribution; defaults to mostly positive.
        """
        self.probabilities = probabilities or {
            "negative": 0.1,
            "neutral": 0.2,
            "positive": 0.7,
        }
        self.calls: list[tuple[Sequence[str], int]] = []

    def predict_proba(
        self,
        texts: Sequence[str],
        batch_size: int,
    ) -> list[dict[str, float]]:
        """Return one copy of the stored distribution per text."""
        self.calls.append((list(texts), batch_size))
        return [dict(self.probabilities) for _ in texts]


@pytest.fixture(autouse=True)
def _clear_backend_cache() -> None:
    """Clear the module-level model cache around every test.

    Yields:
        Control while the test runs.
    """
    transformer_sentiment._cached_backend.cache_clear()
    yield
    transformer_sentiment._cached_backend.cache_clear()


@pytest.fixture
def fake_backend(monkeypatch: pytest.MonkeyPatch) -> FakeBackend:
    """Install a fake inference backend.

    Args:
        monkeypatch: Pytest monkeypatch helper.

    Returns:
        Backend used by the patched loader.
    """
    backend = FakeBackend()
    monkeypatch.setattr(transformer_sentiment, "_load_backend", Mock(return_value=backend))
    return backend


def test_available_models_are_ordered_by_size() -> None:
    """Expose exactly the two supported models, smallest first."""
    models = available_models()

    assert [model.key for model in models] == ["nano-jev", "laya-multilingual"]
    assert [model.parameters for model in models] == sorted(
        model.parameters for model in MODEL_SPECS.values()
    )
    assert models[0].parameters < models[1].parameters


def test_model_spec_rejects_unknown_model() -> None:
    """Report an unknown model with the available choices."""
    with pytest.raises(ValueError, match="nano-jev, laya-multilingual"):
        model_spec("missing-model")


def test_predict_sentiment_returns_label_score_and_confidence(fake_backend: FakeBackend) -> None:
    """Convert probabilities into an argmax label and expected polarity."""
    predictions = predict_sentiment(["great game", "bad game"], batch_size=1)

    assert len(predictions) == 2
    prediction = predictions[0]
    assert prediction.label == "positive"
    assert prediction.score == pytest.approx(0.6)
    assert prediction.confidence == pytest.approx(0.7)
    assert prediction.probabilities == pytest.approx(
        {"negative": 0.1, "neutral": 0.2, "positive": 0.7}
    )
    assert fake_backend.calls == [(["great game", "bad game"], 1)]


def test_predict_sentiment_normalises_backend_output(fake_backend: FakeBackend) -> None:
    """Accept unnormalised backend output and normalize it before scoring."""
    fake_backend.probabilities = {"negative": 2.0, "neutral": 1.0, "positive": 7.0}

    prediction = predict_sentiment(["text"])[0]

    assert prediction.probabilities == pytest.approx(
        {"negative": 0.2, "neutral": 0.1, "positive": 0.7}
    )
    assert prediction.score == pytest.approx(0.5)


def test_predict_sentiment_rejects_invalid_backend_output(fake_backend: FakeBackend) -> None:
    """Reject negative probabilities instead of silently clipping them."""
    fake_backend.probabilities = {"negative": -0.1, "neutral": 0.2, "positive": 0.9}

    with pytest.raises(ValueError, match="Invalid probability"):
        predict_sentiment(["text"])


def test_predict_sentiment_handles_empty_input(fake_backend: FakeBackend) -> None:
    """Return no predictions without loading a backend for empty input."""
    assert predict_sentiment([]) == []
    assert fake_backend.calls == []


def test_predict_sentiment_rejects_nonpositive_batch_size(fake_backend: FakeBackend) -> None:
    """Reject a zero batch size before loading a model."""
    with pytest.raises(ValueError, match="batch_size"):
        predict_sentiment(["text"], batch_size=0)
    assert fake_backend.calls == []


def test_sentiment_scores_returns_expected_polarity(fake_backend: FakeBackend) -> None:
    """Expose expected polarity for callers that only need scores."""
    assert sentiment_scores(["text"]) == pytest.approx([0.6])


def test_add_transformer_sentiment_preserves_input_and_adds_columns(
    fake_backend: FakeBackend,
    sample_frame: pd.DataFrame,
) -> None:
    """Score processed reviews without modifying the caller's frame."""
    original = sample_frame.assign(processed_review=sample_frame["text"])
    before = original.copy(deep=True)

    result = add_transformer_sentiment(original)

    pd.testing.assert_frame_equal(original, before)
    assert {
        "transformer_score",
        "transformer_category",
        "transformer_confidence",
        "transformer_negative",
        "transformer_neutral",
        "transformer_positive",
        "transformer_model",
    }.issubset(result.columns)
    assert result["transformer_category"].eq("Positive").all()
    assert result["transformer_score"].eq(0.6).all()
    assert result["transformer_model"].eq(DEFAULT_MODEL).all()
    assert len(fake_backend.calls) == 1


def test_add_transformer_sentiment_rejects_missing_column(sample_frame: pd.DataFrame) -> None:
    """Report a missing text column with a clear error."""
    with pytest.raises(KeyError, match="processed_review"):
        add_transformer_sentiment(sample_frame)


def test_transformer_sentiment_summary_reports_counts_and_shares() -> None:
    """Summarize count, mean polarity and category shares."""
    reviews = pd.DataFrame(
        {
            "transformer_score": [0.6, 0.0, -0.7, 0.2],
            "transformer_category": ["Positive", "Neutral", "Negative", "Positive"],
        }
    )

    summary = transformer_sentiment_summary(reviews)

    assert summary["count"] == 4
    assert summary["mean_score"] == pytest.approx(0.025)
    assert summary["positive_share"] == 0.5
    assert summary["neutral_share"] == 0.25
    assert summary["negative_share"] == 0.25


def test_transformer_sentiment_summary_handles_empty_frame() -> None:
    """Return NaN aggregates when the frame has no reviews."""
    summary = transformer_sentiment_summary(
        pd.DataFrame(
            {
                "transformer_score": pd.Series(dtype="float64"),
                "transformer_category": pd.Series(dtype="object"),
            }
        )
    )

    assert summary["count"] == 0
    assert summary["mean_score"] != summary["mean_score"]
    assert summary["positive_share"] != summary["positive_share"]


def test_transformer_sentiment_summary_rejects_missing_columns() -> None:
    """Report missing transformer columns with a clear error."""
    with pytest.raises(KeyError, match="transformer_score"):
        transformer_sentiment_summary(pd.DataFrame({"transformer_category": ["Positive"]}))


def test_nano_jev_backend_batches_decisions() -> None:
    """Split texts into the requested batch size and select the sentiment labels."""
    decider = Mock()
    decider.decide_many.side_effect = lambda items: [
        {"negative": 0.1, "neutral": 0.2, "positive": 0.7} for _ in items
    ]

    probabilities = _NanoJevBackend(decider).predict_proba(["one", "two", "three"], 2)

    assert len(probabilities) == 3
    assert decider.decide_many.call_count == 2
    first_items = decider.decide_many.call_args_list[0].args[0]
    assert first_items[0]["options"] == list(SENTIMENT_LABELS)
    assert first_items[0]["state"] == "one"


def test_laya_backend_reads_choice_probabilities() -> None:
    """Adapt Laya's batched answer shape to plain probability mappings."""
    agent = Mock()
    agent.predict_batch.return_value = [
        {
            "answers": {
                "sentiment": {"probabilities": {"negative": 0.1, "neutral": 0.2, "positive": 0.7}}
            }
        }
    ]

    probabilities = _LayaBackend(agent).predict_proba(["text"], 8)

    assert probabilities == [{"negative": 0.1, "neutral": 0.2, "positive": 0.7}]
    agent.predict_batch.assert_called_once()


def test_laya_backend_rejects_missing_probabilities() -> None:
    """Report an unexpected Laya response shape."""
    agent = Mock()
    agent.predict_batch.return_value = [{"answers": {}}]

    with pytest.raises(ValueError, match="sentiment probabilities"):
        _LayaBackend(agent).predict_proba(["text"], 8)


def test_missing_optional_dependency_reports_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Translate an ImportError into an actionable runtime error."""
    monkeypatch.setattr(
        transformer_sentiment.importlib,
        "import_module",
        Mock(side_effect=ImportError("missing")),
    )

    with pytest.raises(RuntimeError, match="uv sync --extra sentiment"):
        transformer_sentiment._load_backend(MODEL_SPECS["nano-jev"], None)


def test_transformer_sentiment_arguments_parse() -> None:
    """Parse the transformer sentiment subcommand and its model options."""
    args = cli.build_parser().parse_args(
        [
            "analyse",
            "transformer-sentiment",
            "--game",
            "Alpha",
            "--model",
            "laya-multilingual",
            "--batch-size",
            "4",
            "--device",
            "cpu",
        ]
    )

    assert args.analysis_command == "transformer-sentiment"
    assert args.model == "laya-multilingual"
    assert args.batch_size == 4
    assert args.device == "cpu"


def test_transformer_sentiment_cli_writes_table(
    monkeypatch: pytest.MonkeyPatch,
    sample_frame: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """Dispatch the CLI command and write the returned table."""
    prepared = SimpleNamespace(reviews=sample_frame.assign(processed_review=sample_frame["text"]))
    monkeypatch.setattr(analysis_cli, "load_reviews", Mock(return_value=sample_frame))
    monkeypatch.setattr(analysis_cli, "prepare_documents", Mock(return_value=prepared))
    scored = sample_frame.assign(
        transformer_score=0.5,
        transformer_category="Positive",
    )
    workflow = Mock(return_value=scored)
    monkeypatch.setattr(analysis_cli, "add_transformer_sentiment", workflow)

    code = cli.main(
        [
            "analyse",
            "transformer-sentiment",
            "--game",
            "Alpha",
            "--output-dir",
            str(tmp_path),
            "--batch-size",
            "3",
        ]
    )

    assert code == 0
    assert workflow.call_args.kwargs["batch_size"] == 3
    assert (tmp_path / "transformer_sentiment.csv").exists()
