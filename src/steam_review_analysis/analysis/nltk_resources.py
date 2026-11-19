"""Verify, and on request download, the NLTK corpora used by the analysis utilities.

Library functions only check that the corpora exist; they never download anything on their
own. Install the corpora once, either from Python with
`ensure_nltk_resources(download=True)` or from a shell with
`steam-review-analysis download-nltk`. The `steam-review-analysis analyse` commands download missing
corpora automatically.
"""

import logging
from collections.abc import Callable

import nltk

_logger = logging.getLogger(__name__)

REQUIRED_RESOURCES: dict[str, str] = {
    "tokenizers/punkt": "punkt",
    "tokenizers/punkt_tab": "punkt_tab",
    "corpora/stopwords": "stopwords",
    "taggers/averaged_perceptron_tagger": "averaged_perceptron_tagger",
    "taggers/averaged_perceptron_tagger_eng": "averaged_perceptron_tagger_eng",
    "corpora/wordnet": "wordnet",
    "sentiment/vader_lexicon": "vader_lexicon",
}
INSTALL_HINT = (
    "run `steam-review-analysis download-nltk` or call "
    "`steam_review_analysis.analysis.nltk_resources.ensure_nltk_resources(download=True)`"
)


def missing_nltk_resources() -> list[str]:
    """List the required NLTK packages that are not installed.

    Returns:
        NLTK package names, in `REQUIRED_RESOURCES` order; empty when all are present.
    """
    return [package for resource, package in REQUIRED_RESOURCES.items() if not _installed(resource)]


def _installed(resource: str) -> bool:
    """Check whether an NLTK resource is installed, unpacked or as a zip archive.

    `nltk.download` leaves some packages, such as `wordnet` and `vader_lexicon`, zipped,
    and `nltk.data.find` only matches those under their `.zip` name.

    Args:
        resource: Resource path such as `corpora/wordnet`.

    Returns:
        True when the resource or its zip archive is on NLTK's search path.
    """
    for candidate in (resource, f"{resource}.zip"):
        try:
            nltk.data.find(candidate)
        except LookupError:
            continue
        return True
    return False


def ensure_nltk_resources(
    *,
    download: bool = False,
    downloader: Callable[[str], bool] = nltk.download,
) -> None:
    """Check that the NLTK corpora required by preprocessing and sentiment are installed.

    Args:
        download: Download missing corpora into NLTK's default data directory instead of
            failing. Off by default so library calls never reach the network.
        downloader: Callable that downloads one named NLTK package and reports success.

    Raises:
        RuntimeError: A required package is missing and `download` is false, or the
            download failed.
    """
    missing = missing_nltk_resources()
    if not missing:
        return
    if not download:
        raise RuntimeError(f"Missing NLTK resources {', '.join(missing)}; {INSTALL_HINT}")
    for package in missing:
        _logger.info("Downloading NLTK resource %s", package)
        if not downloader(package):
            raise RuntimeError(f"Could not download the NLTK resource {package!r}")
