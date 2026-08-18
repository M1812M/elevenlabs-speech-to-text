import io

import pytest

from elevenlabs_toolkit.cli.context import CliContext


def test_live_progress_emits_stable_phase_lines_when_not_attached_to_a_terminal() -> None:
    stderr = io.StringIO()
    context = CliContext(stderr=stderr)

    with context.live_progress() as update:
        update("[1/1] clip.wav - uploading + transcribing")
        update("[1/1] clip.wav - complete")

    assert stderr.getvalue().splitlines() == [
        "[1/1] clip.wav - uploading + transcribing",
        "[1/1] clip.wav - complete",
    ]


@pytest.mark.parametrize("mode", ["quiet", "json"])
def test_live_progress_does_not_pollute_quiet_or_json_output(mode: str) -> None:
    stderr = io.StringIO()
    context = CliContext(
        quiet=mode == "quiet",
        json_output=mode == "json",
        stderr=stderr,
    )

    with context.live_progress() as update:
        update("must remain hidden")

    assert stderr.getvalue() == ""
