"""Configuration loading and named workflow profiles.

Configuration files use a deliberately small TOML shape::

    profile = "social"

    [segmentation]
    max_lines = 2

    [text]
    script = "source"

    [profiles.my-profile.segmentation]
    max_duration = 4.0

    [profiles.my-profile.text]
    cleanup = "my-cleanup"

The same keys may live below ``[tool.elevenlabs-toolkit]`` in
``pyproject.toml``.  Configuration is read only; this module never creates a
user or project configuration file.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from dataclasses import fields
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 only
    import tomli as tomllib

from .models import ScriptMode, SegmentationOptions, TextOptions


class ConfigurationError(ValueError):
    """Raised when toolkit configuration cannot be loaded or validated."""


_DEFAULTS: dict[str, dict[str, Any]] = {
    "segmentation": {
        "preset": "standard",
        "max_chars_per_line": 42,
        "max_lines": 2,
        "max_duration": 5.5,
        "min_duration": 1.0,
        "gap_seconds": 0.9,
        "hard_gap_seconds": 1.8,
        "pause_detection": False,
        "srt_fps": 30.0,
        "srt_padding_frames": 2,
        "srt_gap_milliseconds": 80,
    },
    "text": {
        "script": "source",
        "cleanup": None,
        "replacements": (),
    },
}


_SOCIAL_SEGMENTATION: dict[str, Any] = {
    "preset": "social",
    "max_chars_per_line": 30,
    "max_lines": 2,
    "max_duration": 2.6,
    "min_duration": 0.9,
    "gap_seconds": 0.75,
    "hard_gap_seconds": 1.5,
    "pause_detection": True,
    "max_words": 9,
}


# Profiles contain only their intentional changes from the neutral defaults.
# This keeps ordinary user/project settings useful while still making the
# selected profile a later, higher-precedence configuration layer.
_BUILTIN_PROFILES: dict[str, dict[str, Any]] = {
    "standard": {"segmentation": {"preset": "standard"}},
    "social": {"segmentation": _SOCIAL_SEGMENTATION},
    "broadcast": {
        "segmentation": {
            "preset": "broadcast",
            "max_chars_per_line": 42,
            "max_lines": 2,
            "max_duration": 6.0,
            "min_duration": 1.2,
            "gap_seconds": 0.8,
            "hard_gap_seconds": 1.6,
            "pause_detection": True,
        }
    },
}


_SEGMENTATION_KEYS = frozenset(field.name for field in fields(SegmentationOptions))
_TEXT_KEYS = frozenset(field.name for field in fields(TextOptions))
_ROOT_KEYS = frozenset({"profile", "segmentation", "text", "profiles"})


def user_config_path() -> Path:
    """Return the platform-native per-user configuration file location."""

    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "elevenlabs-toolkit" / "config.toml"


def discover_project_config(cwd: str | Path | None = None) -> Path | None:
    """Find the nearest project configuration while walking towards root.

    A dedicated ``elevenlabs-toolkit.toml`` takes precedence over a
    ``pyproject.toml`` at the same directory level. A pyproject is considered
    a toolkit configuration only when it contains ``[tool.elevenlabs-toolkit]``.
    """

    current = Path.cwd() if cwd is None else Path(cwd)
    current = current.expanduser().resolve()
    if current.is_file():
        current = current.parent

    for directory in (current, *current.parents):
        dedicated = directory / "elevenlabs-toolkit.toml"
        if dedicated.is_file():
            return dedicated

        pyproject = directory / "pyproject.toml"
        if pyproject.is_file() and _pyproject_section(pyproject) is not None:
            return pyproject
    return None


def available_profiles(
    config: Mapping[str, Any] | None = None,
    *,
    cwd: str | Path | None = None,
) -> tuple[str, ...]:
    """Return built-in profile names followed by configured custom profiles."""

    if config is None and cwd is not None:
        user_layer = _resolve_user_source(None, None)
        project_layer = _resolve_project_source(None, None, cwd)
        _validate_config_table(user_layer, "user configuration")
        _validate_config_table(project_layer, "project configuration")
        return tuple(_profile_definitions(user_layer, project_layer))

    names = list(_BUILTIN_PROFILES)
    if config is not None:
        profiles = config.get("profiles", {})
        if profiles is None:
            profiles = {}
        if not isinstance(profiles, Mapping):
            raise ConfigurationError("profiles must be a table")
        names.extend(sorted(str(name) for name in profiles if str(name) not in names))
        selected = config.get("profile")
        if isinstance(selected, str) and selected and selected not in names:
            names.append(selected)
    return tuple(names)


def effective_config(
    profile: str | None = None,
    *,
    overrides: Mapping[str, Any] | None = None,
    explicit_overrides: Mapping[str, Any] | None = None,
    cwd: str | Path | None = None,
    user_config: Mapping[str, Any] | str | Path | None = None,
    project_config: Mapping[str, Any] | str | Path | None = None,
    user_path: str | Path | None = None,
    project_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build the effective configuration with deterministic precedence.

    Precedence is: built-in defaults, user configuration, project
    configuration, the selected named profile, then explicit overrides.

    Passing a mapping for ``user_config`` or ``project_config`` is convenient
    for embedding and tests. Passing a path loads that file. When omitted, the
    platform user path and nearest discovered project file are optional.
    ``user_path`` and ``project_path`` provide explicit path-only aliases.
    """

    if overrides is not None and explicit_overrides is not None:
        raise ConfigurationError("pass either overrides or explicit_overrides, not both")
    if user_config is not None and user_path is not None:
        raise ConfigurationError("pass either user_config or user_path, not both")
    if project_config is not None and project_path is not None:
        raise ConfigurationError("pass either project_config or project_path, not both")

    user_layer = _resolve_user_source(user_config, user_path)
    project_layer = _resolve_project_source(project_config, project_path, cwd)
    override_layer = _normalise_overrides(overrides if overrides is not None else explicit_overrides)

    _validate_config_table(user_layer, "user configuration")
    _validate_config_table(project_layer, "project configuration")

    selected = _selected_profile(profile, user_layer, project_layer, override_layer)
    profiles = _profile_definitions(user_layer, project_layer)
    if selected not in profiles:
        choices = ", ".join(profiles)
        raise ConfigurationError(f"unknown profile {selected!r}; available profiles: {choices}")

    result: dict[str, Any] = _copy_mapping(_DEFAULTS)
    _deep_merge(result, _root_options(user_layer))
    _deep_merge(result, _root_options(project_layer))
    _deep_merge(result, profiles[selected])
    _deep_merge(result, _root_options(override_layer))
    result["profile"] = selected

    # Constructing the typed options here makes effective_config the single
    # validation boundary and also normalizes enums to stable string values.
    segmentation, text = _options_from_mapping(result, context=f"profile {selected!r}")
    return {
        "profile": selected,
        "segmentation": {field.name: getattr(segmentation, field.name) for field in fields(SegmentationOptions)},
        "text": {
            field.name: (
                getattr(text, field.name).value
                if isinstance(getattr(text, field.name), ScriptMode)
                else list(getattr(text, field.name))
                if isinstance(getattr(text, field.name), tuple)
                else getattr(text, field.name)
            )
            for field in fields(TextOptions)
        },
    }


def profile_options(
    name: str,
    config: Mapping[str, Any] | None = None,
) -> tuple[SegmentationOptions, TextOptions]:
    """Create typed segmentation/text options for ``name``.

    ``config`` may be an effective configuration returned by
    :func:`effective_config`, or a raw configuration mapping containing custom
    profiles. With no configuration, only built-in defaults/profiles are used
    and no files are read.
    """

    if config is None:
        effective = effective_config(name, user_config={}, project_config={})
    elif config.get("profile") == name and "profiles" not in config:
        effective = _copy_mapping(config)
    else:
        effective = effective_config(name, user_config={}, project_config=config)
    return _options_from_mapping(effective, context=f"profile {name!r}")


def _resolve_user_source(
    source: Mapping[str, Any] | str | Path | None,
    explicit_path: str | Path | None,
) -> dict[str, Any]:
    if source is not None:
        return _source_mapping(source, kind="user", optional=False)
    path = Path(explicit_path).expanduser() if explicit_path is not None else user_config_path()
    return _source_mapping(path, kind="user", optional=explicit_path is None)


def _resolve_project_source(
    source: Mapping[str, Any] | str | Path | None,
    explicit_path: str | Path | None,
    cwd: str | Path | None,
) -> dict[str, Any]:
    if source is not None:
        return _source_mapping(source, kind="project", optional=False)
    if explicit_path is not None:
        return _source_mapping(Path(explicit_path).expanduser(), kind="project", optional=False)
    discovered = discover_project_config(cwd)
    return {} if discovered is None else _source_mapping(discovered, kind="project", optional=False)


def _source_mapping(
    source: Mapping[str, Any] | str | Path,
    *,
    kind: str,
    optional: bool,
) -> dict[str, Any]:
    if isinstance(source, Mapping):
        return _copy_mapping(source)

    path = Path(source).expanduser()
    if not path.is_file():
        if optional:
            return {}
        raise ConfigurationError(f"{kind} configuration file does not exist: {path}")
    payload = _read_toml(path)
    if path.name.casefold() == "pyproject.toml":
        section = _tool_section(payload)
        if section is None:
            raise ConfigurationError(f"{path} does not contain a [tool.elevenlabs-toolkit] table")
        return _copy_mapping(section)

    # Accept the pyproject-style wrapper in a dedicated file as a convenience,
    # while keeping root-level toolkit settings the normal representation.
    section = _tool_section(payload)
    if section is not None and not any(key in payload for key in _ROOT_KEYS):
        return _copy_mapping(section)
    return payload


def _read_toml(path: Path) -> dict[str, Any]:
    if tomllib is None:  # Defensive for embedded environments with a broken install.
        raise ConfigurationError(f"tomllib-compatible TOML parser is unavailable; install tomli to read {path}")
    try:
        with path.open("rb") as stream:
            payload = tomllib.load(stream)
    except (OSError, ValueError) as exc:
        raise ConfigurationError(f"could not read TOML configuration {path}: {exc}") from exc
    if not isinstance(payload, dict):  # tomllib currently always returns dict
        raise ConfigurationError(f"TOML configuration {path} must contain a table")
    return payload


def _pyproject_section(path: Path) -> Mapping[str, Any] | None:
    return _tool_section(_read_toml(path))


def _tool_section(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    tool = payload.get("tool")
    if not isinstance(tool, Mapping):
        return None
    section = tool.get("elevenlabs-toolkit")
    return section if isinstance(section, Mapping) else None


def _selected_profile(
    requested: str | None,
    user: Mapping[str, Any],
    project: Mapping[str, Any],
    overrides: Mapping[str, Any],
) -> str:
    value: Any = "standard"
    for layer in (user, project, overrides):
        if "profile" in layer:
            value = layer["profile"]
    if requested is not None:
        value = requested
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError("profile must be a non-empty string")
    return value.strip()


def _profile_definitions(user: Mapping[str, Any], project: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result = _copy_mapping(_BUILTIN_PROFILES)
    for label, layer in (("user configuration", user), ("project configuration", project)):
        configured = layer.get("profiles", {})
        if configured is None:
            continue
        if not isinstance(configured, Mapping):
            raise ConfigurationError(f"{label}.profiles must be a table")
        for raw_name, definition in configured.items():
            name = str(raw_name)
            if not name:
                raise ConfigurationError(f"{label} contains an empty profile name")
            if not isinstance(definition, Mapping):
                raise ConfigurationError(f"{label}.profiles.{name} must be a table")
            _validate_options_table(definition, f"{label}.profiles.{name}")
            current = result.setdefault(name, {"segmentation": {"preset": name}})
            _deep_merge(current, definition)
            current.setdefault("segmentation", {}).setdefault("preset", name)
    return result


def _normalise_overrides(overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    if overrides is None:
        return {}
    if not isinstance(overrides, Mapping):
        raise ConfigurationError("explicit overrides must be a mapping")

    result: dict[str, Any] = {}
    for raw_key, value in overrides.items():
        key = str(raw_key)
        if key in _ROOT_KEYS:
            if key in {"segmentation", "text", "profiles"} and isinstance(value, Mapping):
                result[key] = _copy_mapping(value)
            else:
                result[key] = value
            continue
        if "." in key:
            section, option = key.split(".", 1)
            valid = _SEGMENTATION_KEYS if section == "segmentation" else _TEXT_KEYS if section == "text" else ()
            if option not in valid:
                raise ConfigurationError(f"unknown explicit override {key!r}")
            result.setdefault(section, {})[option] = value
            continue
        if key in _SEGMENTATION_KEYS:
            result.setdefault("segmentation", {})[key] = value
        elif key in _TEXT_KEYS:
            result.setdefault("text", {})[key] = value
        else:
            raise ConfigurationError(f"unknown explicit override {key!r}")

    _validate_options_table(result, "explicit overrides", allow_profile=True)
    return result


def _validate_config_table(config: Mapping[str, Any], context: str) -> None:
    unknown = set(config) - _ROOT_KEYS
    if unknown:
        names = ", ".join(sorted(str(name) for name in unknown))
        raise ConfigurationError(f"{context} contains unknown key(s): {names}")
    _validate_options_table(config, context, allow_profile=True, allow_profiles=True)


def _validate_options_table(
    config: Mapping[str, Any],
    context: str,
    *,
    allow_profile: bool = False,
    allow_profiles: bool = False,
) -> None:
    allowed = {"segmentation", "text"}
    if allow_profile:
        allowed.add("profile")
    if allow_profiles:
        allowed.add("profiles")
    unknown_sections = set(config) - allowed
    if unknown_sections:
        names = ", ".join(sorted(str(name) for name in unknown_sections))
        raise ConfigurationError(f"{context} contains unknown key(s): {names}")

    for section, valid_keys in (("segmentation", _SEGMENTATION_KEYS), ("text", _TEXT_KEYS)):
        options = config.get(section)
        if options is None:
            continue
        if not isinstance(options, Mapping):
            raise ConfigurationError(f"{context}.{section} must be a table")
        unknown = set(options) - valid_keys
        if unknown:
            names = ", ".join(sorted(str(name) for name in unknown))
            raise ConfigurationError(f"{context}.{section} contains unknown option(s): {names}")


def _root_options(config: Mapping[str, Any]) -> dict[str, Any]:
    return {section: _copy_mapping(config[section]) for section in ("segmentation", "text") if section in config}


def _options_from_mapping(config: Mapping[str, Any], *, context: str) -> tuple[SegmentationOptions, TextOptions]:
    _validate_options_table(config, context, allow_profile=True)
    segmentation_values = dict(config.get("segmentation", {}))
    text_values = dict(config.get("text", {}))
    try:
        if "script" in text_values:
            text_values["script"] = ScriptMode(text_values["script"])
        if "replacements" in text_values:
            raw_replacements = text_values["replacements"]
            if not isinstance(raw_replacements, (list, tuple)):
                raise ValueError("text.replacements must be an array of FROM=TO strings")
            text_values["replacements"] = tuple(str(item) for item in raw_replacements)
        segmentation = SegmentationOptions(**segmentation_values)
        text = TextOptions(**text_values)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"invalid {context}: {exc}") from exc
    return segmentation, text


def _copy_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _copy_mapping(item) if isinstance(item, Mapping) else item for key, item in value.items()}


def _deep_merge(target: dict[str, Any], incoming: Mapping[str, Any]) -> None:
    for raw_key, value in incoming.items():
        key = str(raw_key)
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = _copy_mapping(value) if isinstance(value, Mapping) else value


__all__ = [
    "ConfigurationError",
    "available_profiles",
    "discover_project_config",
    "effective_config",
    "profile_options",
    "user_config_path",
]
