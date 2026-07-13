"""Speech-to-text provider boundaries."""

from .elevenlabs import (
    ElevenLabsProvider,
    build_request_kwargs,
    decode_additional_formats,
    normalize_response,
    resolve_api_key,
)
from .protocol import (
    ProviderCredentialError,
    ProviderDependencyError,
    ProviderError,
    ProviderResponseError,
    ProviderTransientError,
    SpeechToTextProvider,
)

__all__ = [
    "ElevenLabsProvider",
    "ProviderCredentialError",
    "ProviderDependencyError",
    "ProviderError",
    "ProviderResponseError",
    "ProviderTransientError",
    "SpeechToTextProvider",
    "build_request_kwargs",
    "decode_additional_formats",
    "normalize_response",
    "resolve_api_key",
]
