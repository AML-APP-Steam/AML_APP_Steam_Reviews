"""Data-directory resolution and the registry of games to scrape.

The package never stores data next to its own source files. Every location is resolved
when it is needed, in this order:

1. an explicit argument (e.g. `raw_dir=...` or the CLI's `--data-dir`);
2. the `STEAM_REVIEW_ANALYSIS_DATA_DIR` environment variable;
3. `./data` relative to the current working directory.
"""

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

DATA_DIR_ENV = "STEAM_REVIEW_ANALYSIS_DATA_DIR"
"""Environment variable that overrides the default data directory."""

DEFAULT_GAMES_FILE = Path("configs") / "games.toml"
"""Game registry looked up relative to the current working directory by the CLI."""

DEFAULT_PRESETS_FILE = Path("configs") / "presets.toml"
"""Analysis presets looked up relative to the current working directory by the CLI."""


def data_dir(root: Path | None = None) -> Path:
    """Resolve the root directory holding `raw/` and `processed/` data.

    Args:
        root: Explicit directory; takes precedence over the environment and the default.

    Returns:
        `root`, else `$STEAM_REVIEW_ANALYSIS_DATA_DIR`, else `./data` in the working directory.
    """
    if root is not None:
        return root
    configured = os.environ.get(DATA_DIR_ENV)
    return Path(configured) if configured else Path.cwd() / "data"


def raw_dir(root: Path | None = None) -> Path:
    """Resolve the directory holding one raw CSV per scraped game.

    Args:
        root: Data directory; resolved with `data_dir` when None.

    Returns:
        `<data dir>/raw`.
    """
    return data_dir(root) / "raw"


def processed_dir(root: Path | None = None) -> Path:
    """Resolve the directory holding the cleaned `reviews.csv`.

    Args:
        root: Data directory; resolved with `data_dir` when None.

    Returns:
        `<data dir>/processed`.
    """
    return data_dir(root) / "processed"


def analysis_dir(root: Path | None = None) -> Path:
    """Resolve the parent directory of per-game analysis artifacts.

    Args:
        root: Data directory; resolved with `data_dir` when None.

    Returns:
        `<data dir>/processed/analysis`.
    """
    return processed_dir(root) / "analysis"


@dataclass(frozen=True, slots=True)
class Game:
    """Describe a Steam game.

    Attributes:
        app_id: Steam app id, the number in the store URL.
        name: Display name.
    """

    app_id: int
    name: str


def load_games(path: str | os.PathLike[str]) -> dict[int, Game]:
    """Read a game registry from a TOML file.

    Args:
        path: TOML file with a `[[games]]` array of `app_id` / `name` entries.

    Returns:
        Games keyed by app id; empty if the file has no `[[games]]` entries.

    Raises:
        OSError: The file cannot be read.
    """
    with Path(path).open("rb") as file:
        entries = tomllib.load(file).get("games", [])
    return {entry["app_id"]: Game(entry["app_id"], entry["name"]) for entry in entries}
