"""Test that processed data and analysis outputs are replaced atomically."""

from pathlib import Path

import pandas as pd
import pytest

from steam_review_analysis.io import atomic_path, read_processed, write_processed


def test_atomic_path_replaces_destination_on_success(tmp_path: Path) -> None:
    """Publish the new content and leave no temporary file behind."""
    path = tmp_path / "nested" / "out.csv"

    with atomic_path(path) as temporary:
        assert temporary.parent == path.parent
        assert temporary.suffix == ".csv"
        temporary.write_text("new", encoding="utf-8")

    assert path.read_text(encoding="utf-8") == "new"
    assert [item.name for item in path.parent.iterdir()] == ["out.csv"]


@pytest.mark.parametrize("error", [RuntimeError("disk full"), KeyboardInterrupt()])
def test_atomic_path_keeps_previous_file_when_interrupted(
    tmp_path: Path, error: BaseException
) -> None:
    """Keep the old file and clean up when writing fails or is interrupted."""
    path = tmp_path / "out.csv"
    path.write_text("old", encoding="utf-8")

    with pytest.raises(type(error)), atomic_path(path) as temporary:
        temporary.write_text("partial", encoding="utf-8")
        raise error

    assert path.read_text(encoding="utf-8") == "old"
    assert [item.name for item in tmp_path.iterdir()] == ["out.csv"]


def test_write_processed_preserves_dataset_on_interrupted_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Leave the existing processed dataset readable when a rewrite is interrupted."""
    original = pd.DataFrame({"app_id": [1], "date": ["2024-01-01"], "text": ["kept"]})
    write_processed(original, tmp_path)

    targets: list[Path] = []

    def interrupted(self: pd.DataFrame, *args: object, **kwargs: object) -> None:
        """Simulate Ctrl+C after part of the CSV has reached the disk.

        Writing truncated content first matters: a non-atomic writer would pass this test
        if the fake raised before touching the file.

        Args:
            self: Frame being written.
            *args: Destination path first; the rest is ignored.
            **kwargs: Ignored keyword arguments.

        Raises:
            KeyboardInterrupt: Always, after the partial write.
        """
        target = Path(str(args[0]))
        targets.append(target)
        target.write_text("app_id\n2", encoding="utf-8")
        raise KeyboardInterrupt

    monkeypatch.setattr(pd.DataFrame, "to_csv", interrupted)
    with pytest.raises(KeyboardInterrupt):
        write_processed(pd.DataFrame({"app_id": [2]}), tmp_path)
    monkeypatch.undo()

    assert targets, "write_processed never reached to_csv"
    assert not targets[0].exists(), "the partially written file was left behind"
    assert read_processed(tmp_path)["text"].tolist() == ["kept"]
