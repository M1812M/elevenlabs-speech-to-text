"""Typed building blocks for the ElevenLabs post-production toolkit."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("elevenlabs-toolkit")
except PackageNotFoundError:  # Running directly from an uninstalled source tree.
    __version__ = "0.0.0+local"


__all__ = ["__version__"]
