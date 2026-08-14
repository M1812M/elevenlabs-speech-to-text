from elevenlabs_toolkit.renderers import render_txt


def test_render_txt_without_labels() -> None:
    items = (("First sentence.", "speaker_0"), ("Second sentence.", "speaker_1"))

    assert render_txt(items) == "First sentence.\nSecond sentence.\n"


def test_render_txt_ignores_speaker_metadata_and_keeps_source_heading() -> None:
    items = (("First\n sentence.", "speaker_0"), ("Unattributed.", None))

    assert render_txt(items, include_source="episode  1.json") == "# episode 1.json\n\nFirst sentence.\nUnattributed.\n"


def test_render_txt_empty_input_is_empty() -> None:
    assert render_txt(()) == ""
    assert render_txt((), include_source="empty.json") == "# empty.json\n\n"
