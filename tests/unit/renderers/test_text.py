from elevenlabs_toolkit.models import SpeakerLabels
from elevenlabs_toolkit.renderers import render_txt


def test_render_txt_without_labels() -> None:
    items = (("First sentence.", "speaker_0"), ("Second sentence.", "speaker_1"))

    assert render_txt(items) == "First sentence.\nSecond sentence.\n"


def test_render_txt_secondary_labels_infers_first_most_frequent_speaker() -> None:
    items = (
        ("Main one.", "speaker_0"),
        ("Guest.", "speaker_1"),
        ("Main two.", "speaker_0"),
        ("Unknown.", None),
    )

    assert render_txt(items, SpeakerLabels.SECONDARY) == ("Main one.\n[speaker_1] Guest.\nMain two.\nUnknown.\n")


def test_render_txt_secondary_labels_respects_explicit_main_speaker() -> None:
    items = (("Alpha", "speaker_0"), ("Beta", "speaker_1"))

    assert render_txt(items, SpeakerLabels.SECONDARY, main_speaker="speaker_1") == ("[speaker_0] Alpha\nBeta\n")


def test_render_txt_all_labels_and_source_heading() -> None:
    items = (("First\n sentence.", "speaker_0"), ("Unattributed.", None))

    assert render_txt(items, SpeakerLabels.ALL, include_source="episode  1.json") == (
        "# episode 1.json\n\n[speaker_0] First sentence.\nUnattributed.\n"
    )


def test_render_txt_empty_input_is_empty() -> None:
    assert render_txt(()) == ""
    assert render_txt((), include_source="empty.json") == "# empty.json\n\n"
