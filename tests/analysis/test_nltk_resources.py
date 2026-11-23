"""Test that library code never downloads NLTK data unless asked to."""

from unittest.mock import Mock

import pytest

from steam_review_analysis.analysis import nltk_resources


def test_missing_resources_raise_without_downloading(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail with an install hint instead of reaching the network by default."""
    # Arrange
    monkeypatch.setattr(nltk_resources, "missing_nltk_resources", lambda: ["wordnet"])
    downloader = Mock(return_value=True)

    # Act & Assert
    with pytest.raises(RuntimeError, match="download-nltk"):
        nltk_resources.ensure_nltk_resources(downloader=downloader)
    downloader.assert_not_called()


def test_download_fetches_only_missing_resources(monkeypatch: pytest.MonkeyPatch) -> None:
    """Download exactly the missing packages when explicitly requested."""
    # Arrange
    monkeypatch.setattr(nltk_resources, "missing_nltk_resources", lambda: ["wordnet"])
    downloader = Mock(return_value=True)

    # Act
    nltk_resources.ensure_nltk_resources(download=True, downloader=downloader)

    # Assert
    downloader.assert_called_once_with("wordnet")


def test_failed_download_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    """Raise when NLTK reports a failed download."""
    # Arrange
    monkeypatch.setattr(nltk_resources, "missing_nltk_resources", lambda: ["wordnet"])

    # Act & Assert
    with pytest.raises(RuntimeError, match="wordnet"):
        nltk_resources.ensure_nltk_resources(download=True, downloader=Mock(return_value=False))


def test_zipped_resources_count_as_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Accept corpora that NLTK keeps as zip archives, such as wordnet."""

    # Arrange
    def find(resource: str) -> str:
        """Pretend only zip archives exist.

        Args:
            resource: Resource path looked up by NLTK.

        Returns:
            The resource path when it names a zip archive.

        Raises:
            LookupError: The resource is not a zip archive.
        """
        if not resource.endswith(".zip"):
            raise LookupError(resource)
        return resource

    monkeypatch.setattr(nltk_resources.nltk.data, "find", find)

    # Act & Assert
    assert nltk_resources.missing_nltk_resources() == []
