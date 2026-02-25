# ElevenLabs Subtitle Toolkit

Small local toolkit for:
- transcribing audio/video with ElevenLabs,
- generating standard subtitle/text files,
- creating social subtitle variants in Uzbek Latin and Cyrillic,
- converting edited Latin SRT files back to Cyrillic.

## Folder Layout

The scripts use this structure:

```text
elevenlabs_toolkit/  # shared internal modules used by all scripts
  paths.py
  timecode.py
  transcript_utils.py

media/
  REC/         # input audio/video files for transcription
  JSON/        # transcription JSON outputs
  SRT/         # regular SRT outputs
  TXT/         # regular TXT outputs
  SRT-social/  # social subtitle variants (_social_latin/_social_cyrillic)
```

Top-level script files remain entrypoints:
- `elevenlabs-speech-text.py`
- `json_to_social_srt.py`
- `parse_json.py`

## Requirements

- Python 3.10+
- `elevenlabs` SDK
- `tqdm`
- Optional: `python-dotenv` (for `.env` loading)

Install packages:

```powershell
pip install elevenlabs tqdm python-dotenv
```

Create `.env` in project root:

```env
ELEVENLABS_API_KEY=your_key_here
```

## Main Scripts

### 1) `elevenlabs-speech-text.py`

Batch transcribes all files from `media/REC` and writes:
- JSON to `media/JSON`
- SRT to `media/SRT`
- TXT to `media/TXT`

Run:

```powershell
python .\elevenlabs-speech-text.py
```

Useful examples:

```powershell
python .\elevenlabs-speech-text.py --timestamps-granularity character
python .\elevenlabs-speech-text.py --additional-formats srt txt segmented_json
python .\elevenlabs-speech-text.py --additional-formats --no-include-speakers
```

### 2) `json_to_social_srt.py`

Creates social subtitle versions from JSON:
- `*_social_cyrillic.srt`
- `*_social_latin.srt`

Input: `media/JSON`  
Output: `media/SRT-social`

Run:

```powershell
python .\json_to_social_srt.py
```

Also supports converting edited Latin SRT to Cyrillic while keeping timings:

```powershell
python .\json_to_social_srt.py --mode latin-srt-to-cyrillic
python .\json_to_social_srt.py --latin-srt ".\media\SRT-social\myfile_social_latin.srt"
```

Notes:
- If `--latin-srt` is provided, SRT conversion mode is auto-selected.
- Converted file naming:
  - `*_latin.srt` -> `*_cyrillic.srt`
  - otherwise `*.srt` -> `*_cyrillic.srt`

### 3) `parse_json.py`

Utility converter for existing JSON files:
- single-file JSON -> SRT/TXT
- directory JSON -> per-file SRT/TXT
- directory JSON -> combined TXT

Examples:

```powershell
python .\parse_json.py --input .\media\JSON\file.json --out-srt .\media\SRT\file.srt --out-txt .\media\TXT\file.txt
python .\parse_json.py --srt-dir .\media\JSON
python .\parse_json.py --srt-dir .\media\JSON --sentence-srt
python .\parse_json.py --combine-dir .\media\JSON --out-txt .\media\TXT\combined.txt --only-combine
```

## Typical Workflow

1. Put source media files into `media/REC`.
2. Run `elevenlabs-speech-text.py`.
3. Run `json_to_social_srt.py`.
4. Optionally edit `*_social_latin.srt`.
5. Convert edited Latin back to Cyrillic with `json_to_social_srt.py --latin-srt ...`.

## Troubleshooting

- `Missing ELEVENLABS_API_KEY`: add key to environment or `.env`.
- `No audio files found`: check files are in `media/REC` and have supported extensions.
- `No JSON files found`: ensure transcription step already created JSON in `media/JSON`.
