"""Test loading analysis presets from TOML files."""

from pathlib import Path

import pytest

from steam_review_analysis.analysis.presets import DEFAULT_PRESETS_FILE, load_presets, preset_for
from steam_review_analysis.analysis.text import DEFAULT_EXTRA_STOPWORDS


def test_load_presets_builds_preprocessing_config(tmp_path: Path) -> None:
    """Map TOML keys onto the preset and its preprocessing rules."""
    path = tmp_path / "presets.toml"
    path.write_text(
        '[presets.A]\napp_id = 1\ncutoff = "2024-01-01"\npreserve_keywords = ["gg"]\n'
        'min_word_length = 3\n[presets.A.replacements]\nwon = "win"\n',
        encoding="utf-8",
    )

    preset = preset_for("A", load_presets(path))

    assert (preset.name, preset.app_id, preset.cutoff) == ("A", 1, "2024-01-01")
    assert preset.preprocessing.preserve_keywords == frozenset({"gg"})
    assert preset.preprocessing.replacements == {"won": "win"}
    assert preset.preprocessing.min_word_length == 3
    assert preset.preprocessing.extra_stopwords == DEFAULT_EXTRA_STOPWORDS


def test_malformed_preset_is_rejected(tmp_path: Path) -> None:
    """Require an app id and cutoff for every preset."""
    path = tmp_path / "presets.toml"
    path.write_text("[presets.A]\napp_id = 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Preset 'A'"):
        load_presets(path)


def test_unknown_preset_lists_available_names() -> None:
    """Name the available presets when a lookup fails."""
    with pytest.raises(KeyError, match="available: none"):
        preset_for("missing", {})


def test_repository_presets_file_is_valid() -> None:
    """Keep the project presets used by the notebooks and CLI loadable."""
    repository_root = Path(__file__).resolve().parents[2]

    presets = load_presets(repository_root / DEFAULT_PRESETS_FILE)

    assert {"Counter_Strike", "PUBG", "SIMS"} <= set(presets)
    assert presets["PUBG"].cutoff == "2022-03-23"
