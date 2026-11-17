"""Score review sentiment with small Jev-style transformer decision models.

The tools are optional. Install them with
``pip install "steam-review-analysis[sentiment]"``; the first call downloads model weights
from Hugging Face. All models are non-generative decision encoders, so they return
calibrated probabilities in one forward pass.
"""

from __future__ import annotations

import importlib
import logging
import math
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, ClassVar, Literal, Protocol, cast

import pandas as pd

SENTIMENT_LABELS = ("negative", "neutral", "positive")
SENTIMENT_QUESTION = "What is the overall sentiment of this Steam game review?"
DEFAULT_MODEL = "nano-jev"
EXTRAS_HINT = (
    "install the transformer sentiment dependencies with "
    '`pip install "steam-review-analysis[sentiment]"`'
)


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Describe one supported Jev-style sentiment model.

    Attributes:
        key: Short name accepted by the public functions and the CLI.
        repo_id: Hugging Face repository identifier.
        backend: Inference library that loads the repository.
        parameters: Approximate parameter count used for the smallest-first ordering.
        description: One-line summary shown in documentation and the CLI.
        license: Weight license.
        revision: Optional repository tag or revision.
        subfolder: Optional checkpoint folder inside the repository.
        quantize: Optional Optimum-Quanto bit width applied after loading.
    """

    key: str
    repo_id: str
    backend: Literal["nanojev", "laya", "tinyjev"]
    parameters: int
    description: str
    license: str
    revision: str | None = None
    subfolder: str | None = None
    quantize: int = 0


MODEL_SPECS: dict[str, ModelSpec] = {
    "nano-jev": ModelSpec(
        key="nano-jev",
        repo_id="sdmlai/nano-jev",
        backend="nanojev",
        parameters=22_700_000,
        description="22.7M-parameter cross-encoder, fastest option and English-only.",
        license="Apache-2.0 with MS MARCO non-commercial note",
        revision="v0.1",
    ),
    "laya-multilingual": ModelSpec(
        key="laya-multilingual",
        repo_id="convaiinnovations/laya",
        backend="laya",
        parameters=321_900_000,
        description="322M-parameter multilingual decision model covering 100+ languages.",
        license="Apache-2.0",
        subfolder="multilingual",
    ),
    "laya-english": ModelSpec(
        key="laya-english",
        repo_id="convaiinnovations/laya",
        backend="laya",
        parameters=421_300_000,
        description="421M-parameter English Laya checkpoint, stronger on English text.",
        license="Apache-2.0",
    ),
    "tiny-jev-0.6b": ModelSpec(
        key="tiny-jev-0.6b",
        repo_id="lostargon/Tiny-Jev",
        backend="tinyjev",
        parameters=595_778_561,
        description="596M-parameter Jev-style decision model trained on review sentiment.",
        license="Apache-2.0",
        revision="62449fb3272d68fb8c8fe8a33082566643a76d0b",
    ),
    "laya-multilingual-8bit": ModelSpec(
        key="laya-multilingual-8bit",
        repo_id="convaiinnovations/laya",
        backend="laya",
        parameters=321_900_000,
        description="322M multilingual Laya quantized to 8-bit Optimum-Quanto weights.",
        license="Apache-2.0",
        subfolder="multilingual",
        quantize=8,
    ),
    "laya-english-8bit": ModelSpec(
        key="laya-english-8bit",
        repo_id="convaiinnovations/laya",
        backend="laya",
        parameters=421_300_000,
        description="421M English Laya quantized to 8-bit Optimum-Quanto weights.",
        license="Apache-2.0",
        quantize=8,
    ),
    "tiny-jev-0.6b-8bit": ModelSpec(
        key="tiny-jev-0.6b-8bit",
        repo_id="lostargon/Tiny-Jev",
        backend="tinyjev",
        parameters=595_778_561,
        description="596M Tiny-Jev quantized to 8-bit Optimum-Quanto weights.",
        license="Apache-2.0",
        revision="62449fb3272d68fb8c8fe8a33082566643a76d0b",
        quantize=8,
    ),
}


@dataclass(frozen=True, slots=True)
class SentimentPrediction:
    """Store one three-class sentiment prediction.

    Attributes:
        label: Highest-probability label.
        score: Expected polarity, `P(positive) - P(negative)` in `[-1, 1]`.
        confidence: Probability of the predicted label.
        probabilities: Full probability distribution over the three labels.
    """

    label: str
    score: float
    confidence: float
    probabilities: dict[str, float]


class _ProbabilityBackend(Protocol):
    """Convert review texts into label probability distributions."""

    def predict_proba(
        self,
        texts: Sequence[str],
        batch_size: int,
    ) -> list[dict[str, float]]:
        """Score a batch of texts.

        Args:
            texts: Review texts.
            batch_size: Maximum number of texts per forward pass.

        Returns:
            One probability mapping per input text.
        """


class _NanoJevBackend:
    """Adapt the Nano-Jev typed-decision API to sentiment probabilities."""

    def __init__(self, decider: Any) -> None:
        """Store a loaded Nano-Jev decider.

        Args:
            decider: Object returned by ``nanojev.load``.
        """
        self._decider = decider

    def predict_proba(
        self,
        texts: Sequence[str],
        batch_size: int,
    ) -> list[dict[str, float]]:
        """Score texts with Nano-Jev option matching.

        Args:
            texts: Review texts.
            batch_size: Maximum number of decisions per call.

        Returns:
            One probability mapping per input text.
        """
        items = [
            {
                "question": SENTIMENT_QUESTION,
                "options": list(SENTIMENT_LABELS),
                "state": text,
            }
            for text in texts
        ]
        results: list[dict[str, float]] = []
        for start in range(0, len(items), batch_size):
            chunk = items[start : start + batch_size]
            results.extend(self._decider.decide_many(chunk))
        return results


class _LayaBackend:
    """Adapt the Laya batch-prediction API to sentiment probabilities."""

    _QUESTIONS: ClassVar[dict[str, dict[str, object]]] = {
        "sentiment": {
            "type": "choice",
            "instructions": SENTIMENT_QUESTION,
            "criteria": {
                "negative": "the reviewer is dissatisfied, complains, or rejects the game",
                "neutral": "mixed, factual, or lukewarm without a clear stance",
                "positive": "the reviewer is satisfied, praises the game, or recommends it",
            },
        }
    }

    def __init__(self, agent: Any) -> None:
        """Store a loaded Laya agent.

        Args:
            agent: Object returned by ``laya.load``.
        """
        self._agent = agent

    def predict_proba(
        self,
        texts: Sequence[str],
        batch_size: int,
    ) -> list[dict[str, float]]:
        """Score texts with Laya's batched choice decision.

        Args:
            texts: Review texts.
            batch_size: Maximum number of states per forward pass.

        Returns:
            One probability mapping per input text.

        Raises:
            ValueError: A result does not contain the expected choice decision.
        """
        results = self._agent.predict_batch(
            list(texts),
            self._QUESTIONS,
            batch_size=batch_size,
        )
        probabilities: list[dict[str, float]] = []
        for result in results:
            answer = result.get("answers", {}).get("sentiment", {})
            values = answer.get("probabilities")
            if not isinstance(values, dict):
                raise ValueError("Laya did not return sentiment probabilities")
            probabilities.append(cast(dict[str, float], values))
        return probabilities


class _TinyJevBackend:
    """Adapt the Tiny-Jev choice API to sentiment probabilities."""

    _CRITERIA: ClassVar[dict[str, str]] = {
        "negative": "the reviewer is dissatisfied, complains, or rejects the game",
        "neutral": "mixed, factual, or lukewarm without a clear stance",
        "positive": "the reviewer is satisfied, praises the game, or recommends it",
    }

    def __init__(self, tokenizer: Any, model: Any) -> None:
        """Store a loaded Tiny-Jev tokenizer and model.

        Args:
            tokenizer: Hugging Face tokenizer.
            model: Model returned by ``AutoModel.from_pretrained`` with remote code.
        """
        self._tokenizer = tokenizer
        self._model = model

    def predict_proba(
        self,
        texts: Sequence[str],
        batch_size: int,
    ) -> list[dict[str, float]]:
        """Score texts one by one with Tiny-Jev's choice primitive.

        Args:
            texts: Review texts.
            batch_size: Accepted for interface compatibility.

        Returns:
            One probability mapping per input text.

        Raises:
            ValueError: A result does not contain a probabilities mapping.
        """
        del batch_size
        probabilities: list[dict[str, float]] = []
        for text in texts:
            result = self._model.choice(
                self._tokenizer,
                text,
                SENTIMENT_QUESTION,
                self._CRITERIA,
            )
            values = result.get("probabilities")
            if not isinstance(values, dict):
                raise ValueError("Tiny-Jev did not return sentiment probabilities")
            probabilities.append(cast(dict[str, float], values))
        return probabilities


def available_models() -> tuple[ModelSpec, ...]:
    """List supported models from smallest to largest.

    Returns:
        Model specifications ordered by parameter count.
    """
    return tuple(sorted(MODEL_SPECS.values(), key=lambda spec: spec.parameters))


def clear_model_cache() -> None:
    """Release loaded transformer models so another checkpoint can use the same memory."""
    _cached_backend.cache_clear()


def _cuda_available(device: str | None) -> bool:
    """Return whether CUDA is available for the requested device.

    Args:
        device: Torch device name, or None for automatic selection.

    Returns:
        True when the request can run on CUDA.
    """
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return False
    available = bool(torch.cuda.is_available())
    return available if device is None else str(device).startswith("cuda") and available


class _LowSmFilter(logging.Filter):
    """Drop the informational Inductor message about disabling GEMM autotuning."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Keep every record except the low-SM autotune notice.

        Args:
            record: Log record being filtered.

        Returns:
            False for the low-SM autotune notice, True otherwise.
        """
        return "Not enough SMs to use max_autotune_gemm mode" not in record.getMessage()


def _configure_inductor(torch: Any, device: str | None = None) -> None:
    """Disable GEMM autotuning and hide its low-SM notice on small GPUs.

    Args:
        torch: Imported torch module.
        device: Torch device name, or None for the current CUDA device.
    """
    target = torch.device(device or "cuda")
    index = target.index if target.index is not None else torch.cuda.current_device()
    if torch.cuda.get_device_properties(index).multi_processor_count >= 68:
        return
    config = getattr(getattr(torch, "_inductor", None), "config", None)
    if config is not None and hasattr(config, "max_autotune_gemm"):
        config.max_autotune_gemm = False
    logger = logging.getLogger("torch._inductor.utils")
    if not any(isinstance(entry, _LowSmFilter) for entry in logger.filters):
        logger.addFilter(_LowSmFilter())


def _prefers_fp16(torch: Any, device: str) -> bool:
    """Return whether a CUDA device predates native bfloat16 support.

    Args:
        torch: Imported torch module.
        device: Resolved torch device string.

    Returns:
        True for CUDA devices below compute capability 8.0.
    """
    if not device.startswith("cuda"):
        return False
    return bool(torch.cuda.get_device_capability(device)[0] < 8)


def _compile_model(model: Any, torch: Any, device: str | None = None) -> Any:
    """Configure Inductor and compile a model.

    Args:
        model: Torch module to compile.
        torch: Imported torch module.
        device: Torch device name, or None for the current CUDA device.

    Returns:
        Compiled module.
    """
    _configure_inductor(torch, device)
    return torch.compile(model)


def _quantize_model(model: Any, bits: int) -> Any:
    """Quantize a torch module with Optimum-Quanto weight-only quantization.

    Args:
        model: Torch module to quantize in place.
        bits: Quantization width, 8 or 4.

    Returns:
        The quantized and frozen model.

    Raises:
        RuntimeError: Optimum-Quanto is not installed.
    """
    try:
        from optimum.quanto import freeze, qint4, qint8, quantize
    except ImportError as exc:
        raise RuntimeError(f"Quantization needs optimum-quanto; {EXTRAS_HINT}") from exc
    quantize(model, weights=qint8 if bits == 8 else qint4)
    freeze(model)
    return model


def model_spec(model: str) -> ModelSpec:
    """Resolve a model key to its specification.

    Args:
        model: Key from :data:`MODEL_SPECS`.

    Returns:
        Matching specification.

    Raises:
        ValueError: The key is not supported.
    """
    try:
        return MODEL_SPECS[model]
    except KeyError:
        choices = ", ".join(spec.key for spec in available_models())
        raise ValueError(
            f"Unknown transformer sentiment model {model!r}; choose from: {choices}"
        ) from None


def _load_backend(spec: ModelSpec, device: str | None) -> _ProbabilityBackend:
    """Import the inference library and load one checkpoint.

    Args:
        spec: Model to load.
        device: Torch device name, or None for automatic selection.

    Returns:
        Loaded probability backend.

    Raises:
        RuntimeError: The optional dependency is not installed.
    """
    if spec.backend == "nanojev":
        try:
            module = importlib.import_module("nanojev")
        except ImportError as exc:
            raise RuntimeError(f"Nano-Jev is not installed; {EXTRAS_HINT}") from exc
        loader = cast(Any, module).load
        decider = loader(spec.revision or "v1.0", device=device)
        if _cuda_available(device):
            torch = importlib.import_module("torch")
            target = device or "cuda"
            if not _prefers_fp16(torch, str(target)):
                decider.model = _compile_model(decider.model, torch, str(target))
        return _NanoJevBackend(decider)

    if spec.backend == "laya":
        try:
            module = importlib.import_module("laya")
        except ImportError as exc:
            raise RuntimeError(f"Laya is not installed; {EXTRAS_HINT}") from exc
        loader = cast(Any, module).load
        compile_model = _cuda_available(device) and not spec.quantize
        if compile_model:
            _configure_inductor(importlib.import_module("torch"), device)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"laya: this checkpoint ships invalid temperatures.*",
                category=RuntimeWarning,
            )
            agent = loader(
                spec.repo_id,
                subfolder=spec.subfolder,
                device=device,
                compile=compile_model,
            )
        if spec.quantize:
            _quantize_model(agent.model, spec.quantize)
            agent.amp_enabled = False
        return _LayaBackend(agent)

    try:
        transformers = importlib.import_module("transformers")
        torch = importlib.import_module("torch")
    except ImportError as exc:
        raise RuntimeError(f"Tiny-Jev needs transformers and torch; {EXTRAS_HINT}") from exc
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        spec.repo_id,
        revision=spec.revision,
        trust_remote_code=True,
    )
    model = transformers.AutoModel.from_pretrained(
        spec.repo_id,
        revision=spec.revision,
        trust_remote_code=True,
    )
    target = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(target).eval()
    if _prefers_fp16(torch, str(target)):
        model = model.to(torch.float16)
    if spec.quantize:
        model = _quantize_model(model, spec.quantize)
    elif _cuda_available(device):
        model = _compile_model(model, torch, target)
    return _TinyJevBackend(tokenizer, model)


@lru_cache(maxsize=16)
def _cached_backend(model: str, device: str | None) -> _ProbabilityBackend:
    """Load and cache a backend for one model and device.

    Args:
        model: Supported model key.
        device: Torch device name, or None for automatic selection.

    Returns:
        Cached probability backend.
    """
    return _load_backend(MODEL_SPECS[model], device)


def _normalise_probabilities(raw: dict[str, float]) -> dict[str, float]:
    """Validate and normalize one probability distribution over sentiment labels.

    Args:
        raw: Model output mapping labels to probabilities.

    Returns:
        Probabilities for every sentiment label that sum to one.

    Raises:
        ValueError: The distribution is negative, non-finite, or sums to zero.
    """
    values: dict[str, float] = {}
    for label in SENTIMENT_LABELS:
        value = float(raw.get(label, 0.0))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"Invalid probability for label {label!r}: {value!r}")
        values[label] = value
    total = sum(values.values())
    if total <= 0.0:
        raise ValueError("Sentiment probabilities sum to zero")
    return {label: value / total for label, value in values.items()}


def predict_sentiment(
    texts: Iterable[str],
    *,
    model: str = DEFAULT_MODEL,
    batch_size: int = 16,
    device: str | None = None,
) -> list[SentimentPrediction]:
    """Score review texts with a small Jev-style decision model.

    Args:
        texts: Review texts.
        model: Supported model key, smallest and fastest by default.
        batch_size: Maximum number of texts per forward pass.
        device: Torch device name such as ``cpu`` or ``cuda``; None selects automatically.

    Returns:
        One prediction per input text, in the same order.

    Raises:
        ValueError: The model or batch size is invalid, or the backend returns bad probabilities.
        RuntimeError: The optional dependency is missing or the model cannot be loaded.
    """
    model_spec(model)
    if batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    values = [str(text) for text in texts]
    if not values:
        return []
    backend = _cached_backend(model, device)
    raw_predictions = backend.predict_proba(values, batch_size)
    if len(raw_predictions) != len(values):
        raise ValueError("Backend returned a different number of predictions than texts")
    predictions: list[SentimentPrediction] = []
    for raw in raw_predictions:
        probabilities = _normalise_probabilities(raw)
        label = max(SENTIMENT_LABELS, key=lambda name: probabilities[name])
        predictions.append(
            SentimentPrediction(
                label=label,
                score=probabilities["positive"] - probabilities["negative"],
                confidence=probabilities[label],
                probabilities=probabilities,
            )
        )
    return predictions


def sentiment_scores(
    texts: Iterable[str],
    *,
    model: str = DEFAULT_MODEL,
    batch_size: int = 16,
    device: str | None = None,
) -> list[float]:
    """Compute expected polarity scores in `[-1, 1]` for review texts.

    Args:
        texts: Review texts.
        model: Supported model key.
        batch_size: Maximum number of texts per forward pass.
        device: Torch device name, or None for automatic selection.

    Returns:
        Expected polarity for every input, in the same order.
    """
    return [
        prediction.score
        for prediction in predict_sentiment(
            texts,
            model=model,
            batch_size=batch_size,
            device=device,
        )
    ]


def add_transformer_sentiment(
    reviews: pd.DataFrame,
    *,
    text_column: str = "processed_review",
    model: str = DEFAULT_MODEL,
    batch_size: int = 16,
    device: str | None = None,
) -> pd.DataFrame:
    """Attach transformer sentiment columns to reviews.

    Added columns are ``transformer_score`` (expected polarity), ``transformer_category``
    (argmax label), ``transformer_confidence``, the three class probabilities, and
    ``transformer_model``. Existing columns, including VADER scores, are preserved.

    Args:
        reviews: Reviews containing the text column to score.
        text_column: Column with processed review text.
        model: Supported model key.
        batch_size: Maximum number of texts per forward pass.
        device: Torch device name, or None for automatic selection.

    Returns:
        Copy of the frame with the transformer sentiment columns appended.

    Raises:
        KeyError: The requested text column does not exist.
        ValueError: The model or batch size is invalid.
        RuntimeError: The optional dependency is missing or the model cannot be loaded.
    """
    if text_column not in reviews.columns:
        raise KeyError(f"Missing review text column: {text_column}")
    predictions = predict_sentiment(
        reviews[text_column].tolist(),
        model=model,
        batch_size=batch_size,
        device=device,
    )
    result = reviews.copy()
    index = result.index
    result["transformer_score"] = pd.Series(
        [prediction.score for prediction in predictions], index=index, dtype="float64"
    )
    result["transformer_category"] = pd.Series(
        [prediction.label.capitalize() for prediction in predictions], index=index, dtype=object
    )
    result["transformer_confidence"] = pd.Series(
        [prediction.confidence for prediction in predictions], index=index, dtype="float64"
    )
    for label in SENTIMENT_LABELS:
        result[f"transformer_{label}"] = pd.Series(
            [prediction.probabilities[label] for prediction in predictions],
            index=index,
            dtype="float64",
        )
    result["transformer_model"] = pd.Series([model] * len(result), index=index, dtype=object)
    return result


def transformer_sentiment_summary(reviews: pd.DataFrame) -> dict[str, float]:
    """Summarize transformer scores and category shares for one review frame.

    Args:
        reviews: Frame containing the transformer sentiment columns.

    Returns:
        Count, mean expected polarity and the fraction of each category.

    Raises:
        KeyError: The transformer sentiment columns are missing.
    """
    for column in ("transformer_score", "transformer_category"):
        if column not in reviews.columns:
            raise KeyError(f"Missing transformer sentiment column: {column}")
    total = len(reviews)
    summary = {
        "count": float(total),
        "mean_score": float(reviews["transformer_score"].mean()) if total else float("nan"),
    }
    categories = reviews["transformer_category"].astype(str).str.capitalize()
    for label in SENTIMENT_LABELS:
        count = int((categories == label.capitalize()).sum())
        summary[f"{label}_share"] = count / total if total else float("nan")
    return summary
