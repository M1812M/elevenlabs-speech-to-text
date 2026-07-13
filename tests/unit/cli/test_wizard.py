from __future__ import annotations

import io

import pytest

from elevenlabs_toolkit.cli import wizard
from elevenlabs_toolkit.cli.context import CliContext


def test_wizard_previews_before_confirmation_and_cancels_safely(monkeypatch: pytest.MonkeyPatch) -> None:
    answers = iter(("transcribe", "clip.wav", "", "", "n"))
    calls: list[list[str]] = []
    monkeypatch.setattr(wizard.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))

    result = wizard.run(
        object(),  # type: ignore[arg-type]
        CliContext(stderr=io.StringIO()),
        dispatch=lambda argv: calls.append(argv) or 0,
    )

    assert result == 0
    assert calls == [["transcribe", "clip.wav", "--output-dir", "artifacts", "--format", "json", "--dry-run"]]


def test_wizard_runs_only_after_a_successful_preview_and_confirmation(monkeypatch: pytest.MonkeyPatch) -> None:
    answers = iter(("export", "sample.json", "", "srt,txt", "yes"))
    calls: list[list[str]] = []
    monkeypatch.setattr(wizard.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))

    result = wizard.run(
        object(),  # type: ignore[arg-type]
        CliContext(stderr=io.StringIO()),
        dispatch=lambda argv: calls.append(argv) or 0,
    )

    base = ["export", "sample.json", "--output-dir", "exports", "--format", "srt", "--format", "txt"]
    assert result == 0
    assert calls == [[*base, "--dry-run"], base]
