from __future__ import annotations

from pathlib import Path

import pytest

from elevenlabs_toolkit import config
from elevenlabs_toolkit.config import ConfigurationError
from elevenlabs_toolkit.models import ScriptMode, SegmentationOptions, TextOptions


def test_builtin_profiles_produce_typed_options_without_reading_files() -> None:
    assert config.available_profiles() == (
        "standard",
        "social",
        "broadcast",
    )

    standard_segmentation, standard_text = config.profile_options("standard")
    social_segmentation, social_text = config.profile_options("social")
    broadcast_segmentation, _ = config.profile_options("broadcast")

    assert isinstance(standard_segmentation, SegmentationOptions)
    assert isinstance(standard_text, TextOptions)
    assert standard_text.script is ScriptMode.SOURCE
    assert social_segmentation.max_chars_per_line < standard_segmentation.max_chars_per_line
    assert social_segmentation.max_duration < standard_segmentation.max_duration
    assert social_segmentation.max_words == 9
    assert social_text.cleanup is None
    assert broadcast_segmentation.max_duration > standard_segmentation.max_duration


def test_effective_config_merges_all_layers_in_documented_order() -> None:
    user = {
        "segmentation": {"max_lines": 3, "gap_seconds": 1.0},
        "text": {"replacements": ["foo=bar"]},
        "profiles": {"social": {"segmentation": {"max_duration": 3.1}}},
    }
    project = {
        "segmentation": {"max_lines": 4, "gap_seconds": 1.1},
        "text": {"cleanup": "uzbek"},
        "profiles": {"social": {"segmentation": {"max_duration": 3.2}}},
    }

    effective = config.effective_config(
        "social",
        user_config=user,
        project_config=project,
        overrides={
            "segmentation.max_lines": 5,
            "gap_seconds": 0.7,
            "script": "latin",
        },
    )

    assert effective["profile"] == "social"
    assert effective["segmentation"]["max_lines"] == 5  # explicit override
    assert effective["segmentation"]["gap_seconds"] == 0.7  # explicit override
    assert effective["segmentation"]["max_duration"] == 3.2  # project profile
    assert effective["segmentation"]["max_chars_per_line"] == 30  # selected profile
    assert effective["text"] == {
        "script": "latin",
        "cleanup": "uzbek",
        "replacements": ["foo=bar"],
    }


def test_project_profile_selection_and_custom_profile() -> None:
    raw = {
        "profile": "interview",
        "profiles": {
            "interview": {
                "segmentation": {"max_duration": 4.0},
                "text": {"script": "cyrillic"},
            }
        },
    }

    effective = config.effective_config(user_config={}, project_config=raw)
    segmentation, text = config.profile_options("interview", effective)

    assert effective["profile"] == "interview"
    assert segmentation.preset == "interview"
    assert segmentation.max_duration == 4.0
    assert text.script is ScriptMode.CYRILLIC
    assert config.available_profiles(raw)[-1] == "interview"


def test_discovers_nearest_dedicated_project_config_upward(tmp_path: Path) -> None:
    outer = tmp_path / "project"
    nested = outer / "src" / "package"
    nested.mkdir(parents=True)
    config_file = outer / "elevenlabs-toolkit.toml"
    config_file.write_text(
        "[segmentation]\nmax_chars_per_line = 37\n[text]\nscript = 'cyrillic'\n",
        encoding="utf-8",
    )

    assert config.discover_project_config(nested) == config_file
    effective = config.effective_config(cwd=nested, user_config={})
    assert effective["segmentation"]["max_chars_per_line"] == 37
    assert effective["text"]["script"] == "cyrillic"


def test_reads_tool_table_from_pyproject(tmp_path: Path) -> None:
    project = tmp_path / "project"
    nested = project / "nested"
    nested.mkdir(parents=True)
    pyproject = project / "pyproject.toml"
    pyproject.write_text(
        "[project]\nname = 'example'\n"
        "[tool.elevenlabs-toolkit.segmentation]\nmax_lines = 3\n"
        "[tool.elevenlabs-toolkit.text]\ncleanup = 'uzbek'\n",
        encoding="utf-8",
    )

    assert config.discover_project_config(nested) == pyproject
    effective = config.effective_config(cwd=nested, user_config={})
    assert effective["segmentation"]["max_lines"] == 3
    assert effective["text"]["cleanup"] == "uzbek"


def test_nearest_project_config_wins_instead_of_merging_ancestors(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    child = parent / "child"
    nested = child / "nested"
    nested.mkdir(parents=True)
    (parent / "elevenlabs-toolkit.toml").write_text("[segmentation]\nmax_lines = 4\n", encoding="utf-8")
    (child / "elevenlabs-toolkit.toml").write_text("[segmentation]\nmax_lines = 3\n", encoding="utf-8")

    effective = config.effective_config(cwd=nested, user_config={})
    assert effective["segmentation"]["max_lines"] == 3


def test_user_config_path_uses_platform_location(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(config.sys, "platform", "win32")
    monkeypatch.setenv("APPDATA", str(tmp_path))

    path = config.user_config_path()
    path.parent.mkdir(parents=True)
    path.write_text("[text]\ncleanup = 'uzbek'\n", encoding="utf-8")

    effective = config.effective_config(project_config={})
    assert path == tmp_path / "elevenlabs-toolkit" / "config.toml"
    assert effective["text"]["cleanup"] == "uzbek"


def test_missing_optional_config_files_leave_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "user_config_path", lambda: tmp_path / "missing.toml")

    effective = config.effective_config(cwd=tmp_path)

    assert effective["profile"] == "standard"
    assert effective["segmentation"]["max_chars_per_line"] == 42
    assert effective["text"]["script"] == "source"


def test_project_config_can_override_srt_frame_padding_and_gap() -> None:
    effective = config.effective_config(
        user_config={},
        project_config={
            "segmentation": {
                "srt_fps": 60,
                "srt_padding_frames": 3,
                "srt_gap_milliseconds": 40,
            }
        },
    )

    assert effective["segmentation"]["srt_fps"] == 60.0
    assert effective["segmentation"]["srt_padding_frames"] == 3
    assert effective["segmentation"]["srt_gap_milliseconds"] == 40


def test_tomllib_unavailable_has_clear_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_file = tmp_path / "elevenlabs-toolkit.toml"
    config_file.write_text("[text]\nscript = 'source'\n", encoding="utf-8")
    monkeypatch.setattr(config, "tomllib", None)

    with pytest.raises(ConfigurationError, match=r"tomllib.*unavailable"):
        config.effective_config(user_config={}, project_config=config_file)


@pytest.mark.parametrize(
    ("project", "message"),
    [
        ({"unknown": 1}, "unknown key"),
        ({"segmentation": {"max_lines": 0}}, "line limits"),
        ({"segmentation": {"max_lines": True}}, "integer"),
        ({"segmentation": {"gap_seconds": float("nan")}}, "finite"),
        ({"text": {"script": "runic"}}, "runic"),
        ({"text": {"cleanup": "typo"}}, "cleanup"),
        ({"text": {"speaker_labels": "all"}}, "unknown option"),
        ({"profiles": []}, "profiles must be a table"),
    ],
)
def test_invalid_configuration_has_context(project: dict, message: str) -> None:
    with pytest.raises(ConfigurationError, match=message):
        config.effective_config(user_config={}, project_config=project)


def test_explicit_nonexistent_config_path_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="does not exist"):
        config.effective_config(user_config={}, project_path=tmp_path / "missing.toml")


def test_available_profiles_loads_custom_project_profiles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "user_config_path", lambda: tmp_path / "missing-user.toml")
    (tmp_path / "elevenlabs-toolkit.toml").write_text(
        "[profiles.custom.segmentation]\nmax_duration = 4.0\n",
        encoding="utf-8",
    )

    assert "custom" in config.available_profiles(cwd=tmp_path)
