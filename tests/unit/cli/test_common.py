from pathlib import Path

from elevenlabs_toolkit.cli.common import input_spec
from elevenlabs_toolkit.cli.main import build_parser


def test_transcribe_uses_simple_project_defaults() -> None:
    args = build_parser().parse_args(["transcribe"])

    assert input_spec(args).paths == (Path("media"),)
    assert args.recursive is True
    assert args.output_dir == Path("media")
    assert args.env_file == Path(".env")
    assert args.on_conflict == "replace"
    assert args.formats is None


def test_folder_followed_by_positional_glob_is_supported() -> None:
    args = build_parser().parse_args(["transcribe", "recordings", "*.wav"])

    spec = input_spec(args)

    assert spec.paths == (Path("recordings"),)
    assert spec.glob == "*.wav"
    assert spec.recursive is True


def test_single_glob_path_is_split_into_folder_and_pattern() -> None:
    args = build_parser().parse_args(["transcribe", str(Path("recordings") / "*.flac")])

    spec = input_spec(args)

    assert spec.paths == (Path("recordings"),)
    assert spec.glob == "*.flac"


def test_bare_glob_uses_default_media_folder() -> None:
    args = build_parser().parse_args(["transcribe", "*.wav"])

    spec = input_spec(args)

    assert spec.paths == (Path("media"),)
    assert spec.glob == "*.wav"


def test_other_file_commands_also_default_to_media() -> None:
    parser = build_parser()
    commands = (["export"], ["clean"], ["inspect"], ["transliterate", "--to", "latin"])

    for command in commands:
        args = parser.parse_args(command)
        assert input_spec(args).paths == (Path("media"),)
        if hasattr(args, "output_dir"):
            assert args.output_dir == Path("media")
