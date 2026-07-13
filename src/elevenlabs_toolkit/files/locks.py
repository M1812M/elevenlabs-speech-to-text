"""Small cross-platform advisory locks for cache ownership."""

from __future__ import annotations

import math
import os
import stat
import time
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from typing import BinaryIO, cast


class FileLockUnavailableError(RuntimeError):
    """Raised when another process currently owns an advisory file lock."""


def _try_lock(stream: BinaryIO) -> None:
    stream.seek(0)
    if os.name == "nt":
        import msvcrt

        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise FileLockUnavailableError("lock is already held by another process") from exc
        return

    fcntl = import_module("fcntl")

    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        raise FileLockUnavailableError("lock is already held by another process") from exc


def _unlock(stream: BinaryIO) -> None:
    stream.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        return

    fcntl = import_module("fcntl")

    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _open_lock_file(path: Path) -> BinaryIO:
    flags = os.O_CREAT | os.O_RDWR
    flags |= getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags, 0o666)
        opened = os.fstat(descriptor)
        linked = os.stat(path, follow_symlinks=False)
        if stat.S_ISLNK(linked.st_mode) or (opened.st_dev, opened.st_ino) != (linked.st_dev, linked.st_ino):
            raise FileLockUnavailableError(f"cache lock path is not a stable regular file: {path}")
        stream = cast(BinaryIO, os.fdopen(descriptor, "r+b"))
        descriptor = None
        return stream
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _lock_path_is_stable(stream: BinaryIO, path: Path) -> bool:
    try:
        opened = os.fstat(stream.fileno())
        linked = os.stat(path, follow_symlinks=False)
    except OSError:
        return False
    return not stat.S_ISLNK(linked.st_mode) and (opened.st_dev, opened.st_ino) == (linked.st_dev, linked.st_ino)


@contextmanager
def exclusive_file_lock(
    path: str | os.PathLike[str],
    *,
    timeout_seconds: float = 0.0,
    poll_interval_seconds: float = 0.1,
) -> Iterator[Path]:
    """Own ``path`` until the context exits, optionally waiting for it.

    Lock files intentionally remain on disk. Removing an advisory lock file can
    create two independently lockable inodes during a hand-off race.
    """

    for value, label, allow_zero in (
        (timeout_seconds, "timeout_seconds", True),
        (poll_interval_seconds, "poll_interval_seconds", False),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{label} must be a finite number")
        if value < 0 or (not allow_zero and value == 0):
            comparator = ">= 0" if allow_zero else "> 0"
            raise ValueError(f"{label} must be {comparator}")

    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        stream = _open_lock_file(lock_path)
    except OSError as exc:
        raise FileLockUnavailableError(f"could not open cache lock {lock_path}: {exc}") from exc
    try:
        deadline = time.monotonic() + float(timeout_seconds)
        while True:
            try:
                _try_lock(stream)
                break
            except FileLockUnavailableError as exc:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise FileLockUnavailableError(
                        f"cache lock remained busy for {float(timeout_seconds):g}s: {lock_path}"
                    ) from exc
                time.sleep(min(float(poll_interval_seconds), remaining))
        try:
            if not _lock_path_is_stable(stream, lock_path):
                raise FileLockUnavailableError(f"cache lock path changed while waiting: {lock_path}")
            yield lock_path
        finally:
            _unlock(stream)
    finally:
        stream.close()


__all__ = ["FileLockUnavailableError", "exclusive_file_lock"]
