# ElevenLabs Subtitle Toolkit

Two main CLIs:

- `elevenlabs_transcribe.py`: sends recordings to ElevenLabs and stores transcript outputs.
- `transcript_transform.py`: local JSON/SRT transformations (no ElevenLabs API call).

## Requirements

- Python 3.10+
- `elevenlabs`
- `tqdm`
- Optional: `python-dotenv`

```powershell
pip install elevenlabs tqdm python-dotenv
```

## Environment Setup

Copy template and set your key locally:

```powershell
Copy-Item .\.env.example .\.env
```

```env
ELEVENLABS_API_KEY=your_key_here
```

## 1) Transcribe With ElevenLabs

```powershell
python .\elevenlabs_transcribe.py --help
```

Behavior:
- If no arguments are passed, it starts interactive prompts.
- `--path` accepts either a file or a folder.
- `--json-out-dir` accepts only a folder.
- Optional extras: `--create-srt`, `--create-txt`, `--language-code`, `--api-formats`.

Examples:

```powershell
python .\elevenlabs_transcribe.py
python .\elevenlabs_transcribe.py --path .\media\REC --json-out-dir .\media\JSON --create-srt --create-txt
python .\elevenlabs_transcribe.py --path .\media\REC\sample.mp3 --json-out-dir .\media\JSON --language-code deu
```

## 2) Transform Existing JSON/SRT

```powershell
python .\transcript_transform.py --help
```

Behavior:
- If no arguments are passed, it prints help and exits.
- `--path` accepts file or folder.
- You must select at least one action:
  - `--create-srt`
  - `--create-sentence-srt`
  - `--create-txt`
  - `--create-txt-combined`
  - `--create-social-srt-latin`
  - `--create-social-srt-cyrillic`
  - `--convert-latin-srt-to-cyrillic`

Examples:

```powershell
python .\transcript_transform.py --path .\media\JSON --create-srt --create-txt
python .\transcript_transform.py --path .\media\JSON --create-txt-combined
python .\transcript_transform.py --path .\media\JSON --create-social-srt-latin --create-social-srt-cyrillic
python .\transcript_transform.py --path .\media\SRT-social --convert-latin-srt-to-cyrillic
```

Combined TXT naming (`--create-txt-combined`):
- one source file -> `<basename>_comb.txt`
- multiple source files with shared prefix -> `<prefix>_comb.txt`
- no clear shared prefix -> `combined.txt`
