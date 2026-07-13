from pathlib import Path
from threading import Event, Thread

import pytest

from elevenlabs_toolkit.files import FileLockUnavailableError, exclusive_file_lock


def test_lock_is_nonblocking_and_does_not_modify_its_file(tmp_path: Path) -> None:
    path = tmp_path / ".cache.lock"
    path.write_bytes(b"existing metadata")

    with exclusive_file_lock(path):
        with pytest.raises(FileLockUnavailableError):
            with exclusive_file_lock(path):
                raise AssertionError("unreachable")

    assert path.read_bytes() == b"existing metadata"


def test_lock_waiter_acquires_after_current_owner_releases(tmp_path: Path) -> None:
    path = tmp_path / ".cache.lock"
    started = Event()
    acquired = Event()
    errors: list[Exception] = []

    def wait_for_lock() -> None:
        started.set()
        try:
            with exclusive_file_lock(path, timeout_seconds=1.0, poll_interval_seconds=0.01):
                acquired.set()
        except Exception as exc:  # pragma: no cover - asserted through errors
            errors.append(exc)

    with exclusive_file_lock(path):
        waiter = Thread(target=wait_for_lock)
        waiter.start()
        assert started.wait(1.0)
        assert not acquired.wait(0.05)

    waiter.join(1.0)
    assert not waiter.is_alive()
    assert acquired.is_set()
    assert errors == []


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"timeout_seconds": float("nan")}, "timeout_seconds"),
        ({"timeout_seconds": float("inf")}, "timeout_seconds"),
        ({"timeout_seconds": -1.0}, "timeout_seconds"),
        ({"poll_interval_seconds": 0.0}, "poll_interval_seconds"),
    ],
)
def test_lock_rejects_invalid_wait_values(tmp_path: Path, kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        with exclusive_file_lock(tmp_path / ".cache.lock", **kwargs):
            raise AssertionError("unreachable")


def test_lock_rejects_symlink_path(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_bytes(b"do not touch")
    link = tmp_path / ".cache.lock"
    try:
        link.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"file symlinks unavailable: {exc}")

    with pytest.raises((FileLockUnavailableError, OSError)):
        with exclusive_file_lock(link):
            raise AssertionError("unreachable")

    assert target.read_bytes() == b"do not touch"
