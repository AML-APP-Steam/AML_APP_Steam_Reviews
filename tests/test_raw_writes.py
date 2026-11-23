"""Test atomic CSV publication and preservation of existing review datasets."""

from collections.abc import Iterator
from datetime import date
from pathlib import Path

import pytest

from steam_review_analysis.io import read_raw_reviews, write_raw_reviews
from steam_review_analysis.schema import Review


@pytest.fixture
def review() -> Review:
    """Build a review that exercises CSV quoting and embedded line breaks.

    Returns:
        Raw review containing a comma, double quotes, and a newline in its text.
    """
    return Review(
        42, date(2024, 5, 21), 'Text, with "quotes" and\na newline', "1.5", "Recommended", "0"
    )


def test_existing_csv_requires_explicit_overwrite(tmp_path: Path, review: Review) -> None:
    """Preserve an existing CSV unless replacement was explicitly requested."""
    # Arrange
    path = tmp_path / "42_reviews.csv"
    path.write_bytes(b"previous dataset")

    # Act
    with pytest.raises(FileExistsError):
        write_raw_reviews(path, [review])

    # Assert
    assert path.read_bytes() == b"previous dataset"


@pytest.mark.parametrize("exists", [False, True])
def test_empty_input_never_changes_destination(tmp_path: Path, exists: bool) -> None:
    """Refuse empty input without creating a file or altering existing data."""
    # Arrange
    path = tmp_path / "42_reviews.csv"
    if exists:
        path.write_bytes(b"previous dataset")

    # Act
    with pytest.raises(ValueError, match="empty"):
        write_raw_reviews(path, [], overwrite=True)

    # Assert
    assert path.read_bytes() == b"previous dataset" if exists else not path.exists()


@pytest.mark.parametrize("exists", [False, True])
def test_interrupted_generation_preserves_previous_file(
    tmp_path: Path,
    review: Review,
    exists: bool,
) -> None:
    """Keep the destination unchanged and remove temporary output if iteration fails."""

    # Arrange
    def interrupted_reviews() -> Iterator[Review]:
        """Simulate an iterable that fails after supplying its first review.

        Yields:
            The review fixture once, allowing the writer to begin staging a CSV.

        Raises:
            OSError: Always raised when the writer requests the next review.
        """
        yield review
        raise OSError("simulated write interruption")

    path = tmp_path / "42_reviews.csv"
    if exists:
        path.write_bytes(b"previous dataset")

    # Act
    with pytest.raises(OSError, match="interruption"):
        write_raw_reviews(path, interrupted_reviews(), overwrite=True)

    # Assert
    assert path.read_bytes() == b"previous dataset" if exists else not path.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ([path.name] if exists else [])


def test_successful_overwrite_keeps_csv_contract(tmp_path: Path, review: Review) -> None:
    """Replace an existing CSV with a readable dataset and leave no staging file behind."""
    # Arrange
    path = tmp_path / "42_reviews.csv"
    path.write_bytes(b"previous dataset")

    # Act
    write_raw_reviews(path, [review], overwrite=True)
    frame = read_raw_reviews(tmp_path)

    # Assert
    assert len(frame) == 1
    assert frame.loc[0, "Review"] == review.text
    assert list(tmp_path.iterdir()) == [path]


def test_concurrent_destination_is_not_overwritten(tmp_path: Path, review: Review) -> None:
    """Preserve a destination created by another writer while the new CSV is staged."""
    # Arrange
    path = tmp_path / "42_reviews.csv"

    def concurrent_reviews() -> Iterator[Review]:
        """Create a competing destination after the writer starts consuming records.

        Yields:
            The review fixture once, before simulating another writer's completed output.
        """
        yield review
        path.write_bytes(b"another writer's dataset")

    # Act
    with pytest.raises(FileExistsError):
        write_raw_reviews(path, concurrent_reviews())

    # Assert
    assert path.read_bytes() == b"another writer's dataset"
    assert list(tmp_path.iterdir()) == [path]


def test_publish_failure_preserves_previous_file_and_removes_temporary(
    tmp_path: Path,
    review: Review,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve old data and clean up staging output if the final file replacement fails."""

    # Arrange
    def fail_publish(source: Path, target: Path) -> Path:
        """Simulate failure to replace the destination with a completed temporary CSV.

        Args:
            source: Temporary source path supplied to the patched Path.replace method.
            target: Intended destination; it remains unchanged.

        Raises:
            OSError: Always raised to simulate a filesystem publication failure.
        """
        raise OSError("simulated publish failure")

    path = tmp_path / "42_reviews.csv"
    path.write_bytes(b"existing dataset")
    monkeypatch.setattr(Path, "replace", fail_publish)

    # Act
    with pytest.raises(OSError, match="publish failure"):
        write_raw_reviews(path, [review], overwrite=True)

    # Assert
    assert path.read_bytes() == b"existing dataset"
    assert list(tmp_path.iterdir()) == [path]
