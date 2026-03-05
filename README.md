# ElevenLabs Subtitle Toolkit

This project uses a `src/` layout with reusable core modules and thin CLI wrappers.

## Structure

```text
project/
  pyproject.toml
  README.md
  .env.example

  src/
    elevenlabs_toolkit/
      __init__.py
      cli/
        transcribe.py
        transform.py
      core/
        stt_client.py
        srt_builder.py
      translit.py
      uzbek_cleanup.py
      io_paths.py
      selectors.py
      timecode.py
      transcript_utils.py

  scripts/
    transcribe.py
    transform.py

  tests/
    test_cli_uzbek_clean_flow.py
    test_html_conversion_flow.py
    test_translit.py
    test_srt_split.py
    test_path_selectors.py
    test_timing_sentence_split.py
    test_uzbek_cleanup.py
```

## Requirements

- Python 3.10+
- `elevenlabs`
- `tqdm`
- Optional: `python-dotenv`

```powershell
pip install -e .
```

or:

```powershell
pip install elevenlabs tqdm python-dotenv
```

## Environment Setup

```powershell
Copy-Item .\.env.example .\.env
```

```env
ELEVENLABS_API_KEY=your_key_here
```

## 1) Transcribe With ElevenLabs

```powershell
python .\scripts\transcribe.py --help
```

Behavior:
- If no arguments are passed, it starts interactive prompts.
- `--path` accepts one file, one folder, or a regex path expression.
- `--json-out-dir` defaults to `media/JSON`.
- Optional extras: `--create-srt`, `--create-txt`, `--language-code`, `--api-formats`.

Examples:

```powershell
python .\scripts\transcribe.py
python .\scripts\transcribe.py --path .\media\REC --create-srt --create-txt
python .\scripts\transcribe.py --path .\media\REC\sample.mp3 --language-code deu
python .\scripts\transcribe.py --path .\media\REC --json-out-dir .\media\JSON-override
```

## 2) Transform Existing JSON/SRT

```powershell
python .\scripts\transform.py --help
```

Behavior:
- If no arguments are passed, it prints help and exits.
- `--path` accepts file, folder, or regex expression path.
- You must select at least one action:
  - `--create-srt`
  - `--create-sentence-srt`
  - `--create-txt`
  - `--create-txt-combined`
  - `--create-clean-json`
  - `--create-social-srt-latin`
  - `--create-social-srt-cyrillic`
  - `--create-social-srt-raw`
  - `--convert-latin-srt-to-cyrillic`
- Uzbek readability options:
  - `--uzbek-clean` applies orthography/casing cleanup to generated outputs.
  - `--sentence-gap-seconds` and `--sentence-hard-gap-seconds` control pause-based sentence splitting for TXT.
  - `--create-clean-json` creates `_uz_clean.json` files and keeps originals untouched.

Examples:

```powershell
python .\scripts\transform.py --path .\media\JSON --create-srt --create-txt
python .\scripts\transform.py --path .\media\JSON --create-txt-combined
python .\scripts\transform.py --path .\media\JSON --create-clean-json --uzbek-clean
python .\scripts\transform.py --path .\media\JSON --create-social-srt-latin --create-social-srt-cyrillic
python .\scripts\transform.py --path .\media\SRT-social --convert-latin-srt-to-cyrillic
```

Combined TXT naming (`--create-txt-combined`):
- one source file -> `<basename>_comb.txt`
- multiple source files with shared prefix -> `<prefix>_comb.txt`
- no clear shared prefix -> `combined.txt`
