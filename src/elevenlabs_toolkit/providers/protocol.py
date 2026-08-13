"""Provider contracts and provider-neutral failures."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..models import TranscriptionOptions


class ProviderError(RuntimeError):
    """Base class for errors raised at a provider boundary."""


class ProviderCredentialError(ProviderError):
    """Credentials are missing, invalid, or rejected by the provider."""


class ProviderDependencyError(ProviderError):
    """A provider's optional runtime dependency is unavailable or incompatible."""


class ProviderTransientError(ProviderError):
    """A request may succeed if it is retried later."""

    def __init__(self, message: str, *, retry_after_seconds: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class ProviderResponseError(ProviderError):
    """A provider returned a response that cannot be used safely."""


@runtime_checkable
class SpeechToTextProvider(Protocol):
    """A service capable of turning a local media file into a transcript."""

    def transcribe(
        self,
        path: str | Path,
        options: TranscriptionOptions,
    ) -> dict[str, Any]:
        """Transcribe ``path`` and return a provider payload as a plain dict."""


__all__ = [
    "ProviderCredentialError",
    "ProviderDependencyError",
    "ProviderError",
    "ProviderResponseError",
    "ProviderTransientError",
    "SpeechToTextProvider",
]
