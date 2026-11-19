"""Named analysis presets: a default cutoff plus game-specific preprocessing rules.

Presets are data, not code. Define them in a TOML file and load them with `load_presets`:

```toml
[presets.PUBG]
app_id = 578080
cutoff = "2022-03-23"
preserve_keywords = ["pubg", "fpp", "tpp"]

[presets.PUBG.replacements]
battlegrounds = "pubg"
```

The CLI reads `./configs/presets.toml` by default, or the file given with `--presets`.
"""

import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from steam_review_analysis.analysis.text import PreprocessingConfig
from steam_review_analysis.config import DEFAULT_PRESETS_FILE as DEFAULT_PRESETS_FILE


@dataclass(frozen=True, slots=True)
class GamePreset:
    """Describe the default analysis configuration for one game.

    Attributes:
        name: Preset name used on the command line.
        app_id: Steam application ID the preset was written for.
        cutoff: Default before/after boundary date in ISO format.
        preprocessing: Keywords and replacements tailored to the game's vocabulary.
    """

    name: str
    app_id: int
    cutoff: str
    preprocessing: PreprocessingConfig


def _preset_from_entry(name: str, entry: Mapping[str, Any]) -> GamePreset:
    """Build one preset from its TOML table.

    Args:
        name: Table name under `[presets]`.
        entry: Parsed table with `app_id`, `cutoff` and optional preprocessing keys.

    Returns:
        Validated preset.

    Raises:
        ValueError: A required key is missing or a value has the wrong type.
    """
    try:
        app_id = int(entry["app_id"])
        cutoff = str(entry["cutoff"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Preset {name!r} needs an integer app_id and a cutoff date") from exc
    options: dict[str, Any] = {
        "preserve_keywords": frozenset(entry.get("preserve_keywords", ())),
        "replacements": MappingProxyType(dict(entry.get("replacements", {}))),
    }
    if "extra_stopwords" in entry:
        options["extra_stopwords"] = frozenset(entry["extra_stopwords"])
    if "min_word_length" in entry:
        options["min_word_length"] = int(entry["min_word_length"])
    return GamePreset(name, app_id, cutoff, PreprocessingConfig(**options))


def load_presets(path: str | os.PathLike[str]) -> dict[str, GamePreset]:
    """Read named analysis presets from a TOML file.

    Args:
        path: TOML file with one `[presets.<name>]` table per preset.

    Returns:
        Presets keyed by name; empty if the file defines none.

    Raises:
        OSError: The file cannot be read.
        ValueError: A preset is malformed.
    """
    with Path(path).open("rb") as file:
        tables = tomllib.load(file).get("presets", {})
    return {name: _preset_from_entry(name, entry) for name, entry in tables.items()}


def preset_for(name: str, presets: Mapping[str, GamePreset]) -> GamePreset:
    """Look up a preset by name.

    Args:
        name: Preset name such as `PUBG`.
        presets: Available presets, e.g. from `load_presets`.

    Returns:
        Matching preset.

    Raises:
        KeyError: The name is not among the available presets.
    """
    try:
        return presets[name]
    except KeyError:
        available = ", ".join(sorted(presets)) or "none"
        raise KeyError(f"Unknown analysis preset {name!r}; available: {available}") from None
