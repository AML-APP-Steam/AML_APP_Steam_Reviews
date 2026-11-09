"""Project paths and the registry of games to scrape."""

import tomllib
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
GAMES_FILE = PROJECT_ROOT / "configs" / "games.toml"


@dataclass(frozen=True, slots=True)
class Game:
    """Describe a Steam game.

    Attributes:
        app_id: Steam app id, the number in the store URL.
        name: Display name.
    """

    app_id: int
    name: str


def load_games(path: Path = GAMES_FILE) -> dict[int, Game]:
    """Read the game registry from a TOML file.

    Args:
        path: TOML file with a `[[games]]` array of `app_id` / `name` entries.

    Returns:
        Games keyed by app id; empty if the file has no `[[games]]` entries.
    """
    with path.open("rb") as file:
        entries = tomllib.load(file).get("games", [])
    return {entry["app_id"]: Game(entry["app_id"], entry["name"]) for entry in entries}
