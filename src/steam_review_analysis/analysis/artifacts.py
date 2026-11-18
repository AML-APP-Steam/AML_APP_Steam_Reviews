"""Persist analysis tables and prepared documents beside the processed dataset."""

import json
import math
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from steam_review_analysis.analysis.text import PreparedDocuments
from steam_review_analysis.io import atomic_path


def save_table(frame: pd.DataFrame, path: Path) -> Path:
    """Write a data frame as CSV, creating its parent directory.

    Args:
        frame: Table to write.
        path: Destination CSV path.

    Returns:
        The destination path.
    """
    with atomic_path(path) as temporary:
        frame.to_csv(temporary, index=False)
    return path


def save_documents(prepared: PreparedDocuments, output_dir: Path) -> tuple[Path, Path]:
    """Write filtered documents and their metadata for topic modelling or STM.

    Args:
        prepared: Reviews with `processed_review` and `document` columns.
        output_dir: Directory receiving `documents.csv` and `metadata.csv`.

    Returns:
        Paths to the documents and metadata CSVs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    documents_path = output_dir / "documents.csv"
    metadata_path = output_dir / "metadata.csv"
    documents = prepared.reviews[["document"]].rename(columns={"document": "Documents"})
    with atomic_path(documents_path) as temporary:
        documents.to_csv(temporary, index=False)
    metadata = prepared.reviews.drop(
        columns=["text", "processed_review", "document"],
        errors="ignore",
    )
    with atomic_path(metadata_path) as temporary:
        metadata.to_csv(temporary, index=False)
    return documents_path, metadata_path


def _finite_or_none(value: object) -> object:
    """Replace NaN and infinite floats with None, recursing into mappings and sequences.

    Args:
        value: JSON-like value, possibly nested.

    Returns:
        The value with every non-finite float replaced by None.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {key: _finite_or_none(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_finite_or_none(item) for item in value]
    return value


def save_summary(summary: Mapping[str, object], path: Path) -> Path:
    """Write a JSON summary of counts and aggregated statistics.

    Non-finite numbers, such as the NaN statistics of an empty before/after period, are
    written as `null` so the file stays valid standard JSON.

    Args:
        summary: JSON-serializable mapping.
        path: Destination JSON path.

    Returns:
        The destination path.

    Raises:
        ValueError: The summary contains a value JSON cannot represent.
    """
    text = json.dumps(_finite_or_none(summary), indent=2, sort_keys=True, allow_nan=False) + "\n"
    with atomic_path(path) as temporary:
        temporary.write_text(text, encoding="utf-8")
    return path
