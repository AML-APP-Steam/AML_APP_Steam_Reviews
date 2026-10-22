"""Download and verify the NLTK corpora used by the analysis utilities."""

import logging
from collections.abc import Callable

import nltk

logger = logging.getLogger(__name__)

REQUIRED_RESOURCES: dict[str, str] = {
    "tokenizers/punkt": "punkt",
    "tokenizers/punkt_tab": "punkt_tab",
    "corpora/stopwords": "stopwords",
    "taggers/averaged_perceptron_tagger": "averaged_perceptron_tagger",
    "taggers/averaged_perceptron_tagger_eng": "averaged_perceptron_tagger_eng",
    "corpora/wordnet": "wordnet",
    "sentiment/vader_lexicon": "vader_lexicon",
}


def ensure_nltk_resources(downloader: Callable[[str], bool] = nltk.download) -> None:
    """Install missing NLTK corpora required by preprocessing and sentiment analysis.

    Args:
        downloader: Callable that downloads one named NLTK package and reports success.

    Raises:
        RuntimeError: A required package is missing and could not be downloaded.
    """
    for resource, package in REQUIRED_RESOURCES.items():
        try:
            nltk.data.find(resource)
        except LookupError:
            logger.info("Downloading NLTK resource %s", package)
            if not downloader(package):
                msg = f"Could not download the NLTK resource {package!r}"
                raise RuntimeError(msg) from None
