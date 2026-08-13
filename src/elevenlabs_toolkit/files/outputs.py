from __future__ import annotations

import os
import tempfile
from pathlib import Path

from elevenlabs_toolkit.models.jobs import ArtifactStatus, ConflictPolicy


class OutputConflictError(FileExistsError):
    """Raised when an output already exists under the error policy."""

    def __init__(self, path: Path) -> None:
        self.path = path
        super().__init__(f"output already exists: {path}")


class AtomicPublishError(OSError):
    """Raised when a filesystem cannot provide atomic no-clobber publication."""

    def __init__(self, path: Path, cause: OSError) -> None:
        self.path = path
        self.cause = cause
        super().__init__(
            f"cannot publish '{path}' without overwriting another process: {cause}. "
            "This filesystem may not support hard links; choose --on-conflict replace only when overwriting is intended."
        )


def ensure_atomic_no_clobber_supported(directory: str | os.PathLike[str]) -> None:
    """Verify that ``directory`` supports the no-clobber publish primitive.

    This probe runs before a paid request so a successful provider response is
    never discarded merely because the destination cannot create hard links.
    """

    parent = Path(directory)
    parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    probe: Path | None = None
    probe_created = False
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=parent,
            prefix=".elevenlabs-toolkit.atomic-probe.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        probe = temporary.with_name(f"{temporary.name}.link")
        try:
            os.link(temporary, probe)
            probe_created = True
        except OSError as exc:
            raise AtomicPublishError(probe, exc) from exc
    finally:
        if probe_created and probe is not None:
            probe.unlink(missing_ok=True)
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _path_exists(path: Path) -> bool:
    """Treat broken symlinks as occupied output names too."""

    return path.exists() or path.is_symlink()


def resolve_conflict_target(
    path: str | os.PathLike[str],
    policy: ConflictPolicy = ConflictPolicy.ERROR,
) -> Path | None:
    """Return the path to write, or ``None`` when an existing output is skipped."""

    target = Path(path)
    if not _path_exists(target):
        return target

    if policy is ConflictPolicy.ERROR:
        raise OutputConflictError(target)
    if policy is ConflictPolicy.SKIP:
        return None
    if policy is ConflictPolicy.REPLACE:
        return target
    if policy is ConflictPolicy.RENAME:
        counter = 2
        while True:
            candidate = target.with_name(f"{target.stem} ({counter}){target.suffix}")
            if not _path_exists(candidate):
                return candidate
            counter += 1

    raise ValueError(f"unsupported conflict policy: {policy!r}")


def _commit_temporary(
    temporary: Path,
    target: Path,
    original: Path,
    policy: ConflictPolicy,
) -> tuple[Path, ArtifactStatus]:
    if policy is ConflictPolicy.REPLACE:
        os.replace(temporary, target)
        return target, ArtifactStatus.WRITTEN

    candidate = target
    while True:
        try:
            # A same-directory hard link publishes the fully flushed file and
            # fails atomically if another process claimed the target first.
            os.link(temporary, candidate)
        except FileExistsError:
            if policy is ConflictPolicy.SKIP:
                temporary.unlink(missing_ok=True)
                return original, ArtifactStatus.SKIPPED
            if policy is ConflictPolicy.ERROR:
                raise OutputConflictError(candidate) from None
            renamed = resolve_conflict_target(original, ConflictPolicy.RENAME)
            if renamed is None:  # pragma: no cover - rename never skips
                raise RuntimeError("rename did not produce an output target") from None
            candidate = renamed
            continue
        except OSError as exc:
            raise AtomicPublishError(candidate, exc) from exc
        temporary.unlink(missing_ok=True)
        return candidate, ArtifactStatus.WRITTEN


def atomic_write_text(
    path: str | os.PathLike[str],
    text: str,
    policy: ConflictPolicy = ConflictPolicy.ERROR,
) -> tuple[Path, ArtifactStatus]:
    """Atomically write UTF-8 text according to the requested conflict policy."""

    original = Path(path)
    target = resolve_conflict_target(original, policy)
    if target is None:
        return original, ArtifactStatus.SKIPPED

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        target, status = _commit_temporary(temporary, target, original, policy)
        temporary = None
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise

    return target, status


def atomic_write_bytes(
    path: str | os.PathLike[str],
    data: bytes,
    policy: ConflictPolicy = ConflictPolicy.ERROR,
) -> tuple[Path, ArtifactStatus]:
    """Atomically write bytes according to the requested conflict policy."""

    original = Path(path)
    target = resolve_conflict_target(original, policy)
    if target is None:
        return original, ArtifactStatus.SKIPPED

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        target, status = _commit_temporary(temporary, target, original, policy)
        temporary = None
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise

    return target, status
