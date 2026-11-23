"""Test that analysis artifacts are written as valid, standard files."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from steam_review_analysis.analysis.artifacts import save_summary


def test_save_summary_writes_non_finite_values_as_null(tmp_path: Path) -> None:
    """Replace NaN and infinities, however deeply nested, with JSON null."""
    # Act
    path = save_summary(
        {
            "before": {"count": 0.0, "mean_score": math.nan},
            "after": {"mean_score": np.float64("inf"), "shares": [0.5, -math.inf]},
            "game": "Alpha",
        },
        tmp_path / "summary.json",
    )

    # Assert
    text = path.read_text(encoding="utf-8")
    assert "NaN" not in text
    assert "Infinity" not in text
    assert json.loads(text) == {
        "after": {"mean_score": None, "shares": [0.5, None]},
        "before": {"count": 0.0, "mean_score": None},
        "game": "Alpha",
    }


def test_save_summary_rejects_unserializable_values(tmp_path: Path) -> None:
    """Fail loudly rather than write a value JSON cannot represent."""
    with pytest.raises(TypeError):
        save_summary({"value": object()}, tmp_path / "summary.json")
