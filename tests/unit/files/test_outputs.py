from __future__ import annotations

import os
from pathlib import Path

import pytest

from elevenlabs_toolkit.files import outputs
from elevenlabs_toolkit.files.outputs import (
    AtomicPublishError,
    OutputConflictError,
    atomic_write_bytes,
    atomic_write_text,
    ensure_atomic_no_clobber_supported,
    resolve_conflict_target,
)
from elevenlabs_toolkit.models.jobs import ArtifactStatus, ConflictPolicy


def test_resolve_conflict_returns_requested_available_path(tmp_path: Path) -> None:
    target = tmp_path / "result.txt"

    assert resolve_conflict_target(target, ConflictPolicy.ERROR) == target


def test_resolve_conflict_raises_by_default(tmp_path: Path) -> None:
    target = tmp_path / "result.txt"
    target.write_text("existing", encoding="utf-8")

    with pytest.raises(OutputConflictError) as caught:
        resolve_conflict_target(target)

    assert caught.value.path == target


def test_resolve_conflict_skips_existing_target(tmp_path: Path) -> None:
    target = tmp_path / "result.txt"
    target.write_text("existing", encoding="utf-8")

    assert resolve_conflict_target(target, ConflictPolicy.SKIP) is None


def test_resolve_conflict_replaces_existing_target(tmp_path: Path) -> None:
    target = tmp_path / "result.txt"
    target.write_text("existing", encoding="utf-8")

    assert resolve_conflict_target(target, ConflictPolicy.REPLACE) == target


def test_resolve_conflict_renames_deterministically(tmp_path: Path) -> None:
    target = tmp_path / "result.txt"
    target.touch()
    (tmp_path / "result (2).txt").touch()

    assert resolve_conflict_target(target, ConflictPolicy.RENAME) == (tmp_path / "result (3).txt")


def test_atomic_write_text_creates_parent_and_writes_utf8(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "result.txt"

    written_path, status = atomic_write_text(target, "O'zbekcha — салом")

    assert (written_path, status) == (target, ArtifactStatus.WRITTEN)
    assert target.read_bytes() == "O'zbekcha — салом".encode()


def test_atomic_write_bytes_preserves_data(tmp_path: Path) -> None:
    target = tmp_path / "result.bin"
    data = b"\x00\xff\x10binary"

    written_path, status = atomic_write_bytes(target, data)

    assert (written_path, status) == (target, ArtifactStatus.WRITTEN)
    assert target.read_bytes() == data


def test_atomic_no_clobber_probe_leaves_directory_empty(tmp_path: Path) -> None:
    ensure_atomic_no_clobber_supported(tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_atomic_no_clobber_probe_reports_unsupported_filesystem(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_link(*args: object, **kwargs: object) -> None:
        raise OSError("hard links unavailable")

    monkeypatch.setattr(outputs.os, "link", fail_link)

    with pytest.raises(AtomicPublishError, match="hard links unavailable"):
        ensure_atomic_no_clobber_supported(tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_atomic_write_skip_preserves_file_and_does_not_prepare_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "result.txt"
    target.write_text("existing", encoding="utf-8")

    def unexpected_mkdir(*args: object, **kwargs: object) -> None:
        raise AssertionError("mkdir must not be called for a skipped output")

    monkeypatch.setattr(Path, "mkdir", unexpected_mkdir)

    skipped_path, status = atomic_write_text(target, "replacement", ConflictPolicy.SKIP)

    assert (skipped_path, status) == (target, ArtifactStatus.SKIPPED)
    assert target.read_text(encoding="utf-8") == "existing"


def test_atomic_write_replace_uses_same_directory_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "result.txt"
    target.write_text("old", encoding="utf-8")
    real_replace = os.replace
    observed: dict[str, Path | str] = {}

    def inspect_replace(source: str | os.PathLike[str], destination: str | os.PathLike[str]) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        observed["source"] = source_path
        observed["content_before_replace"] = destination_path.read_text(encoding="utf-8")
        real_replace(source_path, destination_path)

    monkeypatch.setattr(outputs.os, "replace", inspect_replace)

    written_path, status = atomic_write_text(target, "new", ConflictPolicy.REPLACE)

    assert (written_path, status) == (target, ArtifactStatus.WRITTEN)
    assert Path(observed["source"]).parent == target.parent
    assert observed["content_before_replace"] == "old"
    assert target.read_text(encoding="utf-8") == "new"


def test_atomic_write_rename_writes_selected_target(tmp_path: Path) -> None:
    target = tmp_path / "result.txt"
    target.write_text("old", encoding="utf-8")

    written_path, status = atomic_write_text(target, "new", ConflictPolicy.RENAME)

    assert (written_path, status) == (
        tmp_path / "result (2).txt",
        ArtifactStatus.WRITTEN,
    )
    assert target.read_text(encoding="utf-8") == "old"
    assert written_path.read_text(encoding="utf-8") == "new"


def test_atomic_write_removes_temporary_file_after_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "nested" / "result.txt"

    def fail_link(*args: object, **kwargs: object) -> None:
        raise OSError("link failed")

    monkeypatch.setattr(outputs.os, "link", fail_link)

    with pytest.raises(AtomicPublishError, match="link failed"):
        atomic_write_bytes(target, b"content")

    assert not target.exists()
    assert list(target.parent.iterdir()) == []


@pytest.mark.parametrize(
    ("policy", "expected_status", "expected_name"),
    [
        (ConflictPolicy.SKIP, ArtifactStatus.SKIPPED, "result.txt"),
        (ConflictPolicy.RENAME, ArtifactStatus.WRITTEN, "result (2).txt"),
    ],
)
def test_no_clobber_race_preserves_competing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    policy: ConflictPolicy,
    expected_status: ArtifactStatus,
    expected_name: str,
) -> None:
    target = tmp_path / "result.txt"
    real_link = os.link
    first = True

    def race_link(source: str | os.PathLike[str], destination: str | os.PathLike[str]) -> None:
        nonlocal first
        if first:
            first = False
            Path(destination).write_text("racer", encoding="utf-8")
        real_link(source, destination)

    monkeypatch.setattr(outputs.os, "link", race_link)

    written, status = atomic_write_text(target, "ours", policy)

    assert status is expected_status
    assert written.name == expected_name
    assert target.read_text(encoding="utf-8") == "racer"
    if status is ArtifactStatus.WRITTEN:
        assert written.read_text(encoding="utf-8") == "ours"


def test_error_policy_never_overwrites_racing_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "result.txt"
    real_link = os.link

    def race_link(source: str | os.PathLike[str], destination: str | os.PathLike[str]) -> None:
        Path(destination).write_text("racer", encoding="utf-8")
        real_link(source, destination)

    monkeypatch.setattr(outputs.os, "link", race_link)

    with pytest.raises(OutputConflictError):
        atomic_write_text(target, "ours", ConflictPolicy.ERROR)

    assert target.read_text(encoding="utf-8") == "racer"
