"""Domain presets for the legacy Counter-Strike, PUBG and The Sims notebooks."""

from dataclasses import dataclass
from types import MappingProxyType

from steam_reviews.analysis.text import PreprocessingConfig


@dataclass(frozen=True, slots=True)
class GamePreset:
    """Describe the default analysis configuration for one game.

    Attributes:
        name: Notebook-facing preset name.
        app_id: Steam application ID used when processed data is available.
        cutoff: Default before/after boundary date in ISO format.
        preprocessing: Keywords and replacements tailored to the game's vocabulary.
    """

    name: str
    app_id: int
    cutoff: str
    preprocessing: PreprocessingConfig


COUNTER_STRIKE = GamePreset(
    name="Counter_Strike",
    app_id=730,
    cutoff="2023-09-29",
    preprocessing=PreprocessingConfig(
        preserve_keywords=frozenset({"cs", "vac", "c4", "mm", "5v5", "1v1"}),
        replacements=MappingProxyType(
            {
                "cheating": "cheat",
                "cheater": "cheat",
                "cheated": "cheat",
                "csgo": "cs",
                "cs2": "cs",
                "go": "cs",
            }
        ),
    ),
)

PUBG = GamePreset(
    name="PUBG",
    app_id=578080,
    cutoff="2022-03-23",
    preprocessing=PreprocessingConfig(
        preserve_keywords=frozenset({"pubg", "fps", "fpp", "tpp", "mm", "solo", "squad"}),
        replacements=MappingProxyType(
            {
                "cheating": "cheat",
                "cheater": "cheat",
                "cheated": "cheat",
                "hacker": "cheat",
                "hack": "cheat",
                "aimbot": "cheat",
                "wallhack": "cheat",
                "battlegrounds": "pubg",
            }
        ),
    ),
)

SIMS = GamePreset(
    name="SIMS",
    app_id=1222670,
    cutoff="2022-10-18",
    preprocessing=PreprocessingConfig(
        preserve_keywords=frozenset(
            {"sims", "dlc", "woohoo", "mod", "mods", "buildmode", "livemode", "cas"}
        ),
        replacements=MappingProxyType(
            {
                "cheating": "cheat",
                "cheater": "cheat",
                "cheated": "cheat",
                "build": "buildmode",
                "live": "livemode",
                "createasim": "cas",
                "sim": "sims",
                "babies": "baby",
                "houses": "house",
                "families": "family",
                "expansion": "dlc",
                "expansions": "dlc",
                "pack": "dlc",
                "packs": "dlc",
                "modding": "mod",
                "dlcs": "dlc",
                "mods": "mod",
            }
        ),
    ),
)

ANALYSIS_PRESETS = MappingProxyType({"Counter_Strike": COUNTER_STRIKE, "PUBG": PUBG, "SIMS": SIMS})


def preset_for(name: str) -> GamePreset:
    """Look up a game preset by name.

    Args:
        name: Preset name such as `Counter_Strike`, `PUBG` or `SIMS`.

    Returns:
        Matching preset.

    Raises:
        KeyError: The name is not a known preset.
    """
    try:
        return ANALYSIS_PRESETS[name]
    except KeyError:
        available = ", ".join(sorted(ANALYSIS_PRESETS))
        raise KeyError(f"Unknown analysis preset {name!r}; available: {available}") from None
