# ElevenLabs Toolkit

A small Windows-friendly CLI for sending local audio or video to ElevenLabs
Speech to Text and writing ready-to-use subtitle and transcript files.

`transcribe` saves the complete provider response as the reusable transcript
JSON. It does not create a separate cache manifest or persistent lock file.

## Quick start

Create `.env` in the project root:

```dotenv
ELEVENLABS_API_KEY=your_key_here
```

Then put media in `media` and run:

```powershell
python .\run_toolkit.py transcribe
```

That command recursively finds supported media below `media`, transcribes every
selected file, and writes one reusable JSON transcript per source to `media`.
Existing JSON transcripts are replaced because `transcribe` means “request a
fresh transcription.”

Preview the exact files, outputs, and maximum API attempts without contacting
ElevenLabs:

```powershell
python .\run_toolkit.py --json transcribe --dry-run
```

## Defaults

| Setting | Default |
| --- | --- |
| Input when omitted | `./media` |
| Output when omitted | `./media` |
| Directory scanning | Recursive |
| Transcription output | JSON only |
| Credentials file | `./.env` |
| Model | `scribe_v2` |
| Timestamps | Character |
| Existing transcription outputs | Replace |

An `ELEVENLABS_API_KEY` already present in the process environment takes
precedence over `.env`. Use `--env-file PATH` only when a different dotenv file
is intentional.

## Selecting media

No selector is needed for normal use:

```powershell
# Everything supported below ./media
python .\run_toolkit.py transcribe

# Everything supported below another folder
python .\run_toolkit.py transcribe "D:\Audio\Project"
```

A wildcard such as `*.wav` is a **glob**, not a regular expression. Both simple
forms below work and remain recursive:

```powershell
python .\run_toolkit.py transcribe "D:\Audio\Project" "*.wav"
python .\run_toolkit.py transcribe "D:\Audio\Project\*.wav"
```

A bare pattern uses the default `media` folder:

```powershell
python .\run_toolkit.py transcribe "*.flac"
```

Use `--regex` only when glob syntax is not expressive enough. Regex is matched
against each path relative to the selected folder:

```powershell
python .\run_toolkit.py transcribe .\media `
  --regex '(?i)^day-\d+/.+\.(wav|flac)$'
```

`--glob` remains available for explicit scripts, and `--no-recursive` limits a
directory to its top level.

The automatic selector follows ElevenLabs' documented Speech-to-Text inputs:
AAC, AIFF, OGG, MP3, OPUS, WAV, FLAC, M4A, WebM, MP4, AVI, MKV, MOV, WMV, FLV,
MPEG, and 3GPP. See the
[ElevenLabs transcription documentation](https://elevenlabs.io/docs/overview/capabilities/speech-to-text).

## Transcription outputs

JSON is always written. Add derived local formats by repeating `--format`:

```powershell
python .\run_toolkit.py transcribe --format txt
python .\run_toolkit.py transcribe --format srt --format resolve-edl
```

The first command writes JSON + TXT. The second writes JSON + SRT + Resolve EDL.
Without `--format`, only JSON is written.

Additional local choices are `srt`, `txt`, `srt-mini`, `resolve-edl`, and
`cue-index-srt`. Provider-generated choices are requested with
`--remote-format`: `pdf`, `docx`, `html`, and `segmented-json`.

```powershell
python .\run_toolkit.py transcribe .\media\interview.wav `
  --format srt --remote-format docx
```

Use `-o` only to override the default `media` output folder:

```powershell
python .\run_toolkit.py transcribe .\incoming -o .\finished
```

The provider response JSON is retained as the reusable transcript. No
`.manifest.json` or `.transcription.lock` sidecar is written. Temporary files
used for atomic output publication are removed before the command returns,
including on handled failures.

SRT cue text is written on one line by default so DaVinci Resolve can handle
the visual wrapping. To insert balanced line breaks into the SRT itself, add
`--srt-smart-line-breaks`. With `export`, `--max-chars-per-line` and
`--max-lines` control that optional wrapping:

```powershell
python .\run_toolkit.py export .\media\interview.json --format srt `
  --srt-smart-line-breaks --max-chars-per-line 42 --max-lines 2
```

### Sentence-driven `srt-mini`

`srt-mini` uses only the reusable JSON word timings. No marker file is needed:

```powershell
python .\run_toolkit.py export .\media\interview.json --format srt-mini
```

This writes `interview.mini.srt` using deterministic rules without AI. The
JSON `language_code` selects one shared set of structural phrases:

| Language | Codes | Structural split phrases |
| --- | --- | --- |
| English | `en`, `eng`, locale variants such as `en-US` | `and`, `or`, `but`, `because`, `therefore`, `however`, `otherwise`, `then`, `while`, `although`, `whereas`, `so`, `yet`, `if`, `now`, `in short`, `in general`, `finally` |
| Uzbek | `uz`, `uzb` | `va`, `yoki`, `hamda`, `lekin`, `ammo`, `biroq`, `chunki`, `shuning uchun`, `aks holda`, `keyin`, `shunda`, `demak`, `garchi`, `esa`, `agar`, `hozir`, `xullas`, `umuman`, `nihoyat`, `mana`; the corresponding Uzbek Cyrillic spellings are supported too |
| Kyrgyz | `ky`, `kir` | `жана`, `же`, `же болбосо`, `бирок`, `анткени`, `ошондуктан`, `болбосо`, `андан кийин`, `анда`, `демек`, `ошентсе да`, `ал эми`, `эгерде`, `азыр`, `кыскасы`, `жалпысынан`, `акыры` |
| Russian | `ru`, `rus`, locale variants such as `ru-RU` | `и`, `или`, `либо`, `но`, `а`, `потому что`, `поэтому`, `однако`, `иначе`, `затем`, `тогда`, `хотя`, `так что`, `при этом`, `если`, `теперь`, `короче`, `в общем`, `вообще`, `наконец` |

If an older transcript has no `language_code`, the four sets are combined. A
present but unsupported language code gets no language-specific split rules.
The phrase remains at the beginning of the new cue.

1. Put each complete sentence in its own cue.
2. Split a sentence at semicolons, commas, or before a matching structural
   phrase only when every resulting clause contains at least three spoken words
   and at least 0.8 seconds of natural JSON timing. Semicolons take precedence
   over competing comma splits; sentence-ending punctuation remains a hard
   boundary.
3. Keep short introductions and enumeration fragments together, so isolated
   words do not become subtitle cues.
4. Keep at least 100 milliseconds between adjacent cues. If the JSON timing is
   too tight to fit that gap, join the affected cues instead of losing text or
   producing an invalid timestamp.

The same `srt-mini` format can be used with `transcribe`, in which case the
canonical JSON is still written too. Script conversion belongs to `export`;
add `--script latin` or `--script cyrillic` when desired. Cue text remains on
one line unless `--srt-smart-line-breaks` is supplied.

Uzbek Latin/Cyrillic conversion and `--clean uzbek` remain explicitly
Uzbek-only. English, Kyrgyz, and Russian use the same subtitle structure rules,
but are not passed through Uzbek editorial cleanup or transliteration.

## Paid-call and conflict behavior

Every selected source normally causes a fresh ElevenLabs request. Retries
default to zero because another attempt can incur another charge.

- `--on-conflict replace` is the transcription default and overwrites final
  outputs after a fresh request.
- `--on-conflict error` stops in preflight when an output already exists.
- `--on-conflict skip` avoids the API request when all requested outputs for a
  source already exist. If only some outputs exist, one fresh request creates
  the missing outputs and preserves the existing ones.
- `--dry-run` never writes files or calls ElevenLabs.
- `--retries N` permits `N` additional attempts for transient provider errors.
- `--request-delay SECONDS` spaces requests in a batch.

## Useful transcription options

```text
--model MODEL
--language-code CODE
--timestamps none|word|character
--[no-]diarize
--num-speakers N
--[no-]audio-events
--keyterm TERM
--no-verbatim
--profile NAME
--pause-detection
--srt-smart-line-breaks
--script source|latin|cyrillic
--clean none|uzbek
--speaker-labels none|secondary|all
--replace TOKEN=TOKEN
```

Timed outputs require word or character timestamps. Character timestamps are
the default and give generated SRT cue boundaries the precise first and last
spoken-character times. Word times remain the fallback for older JSON files.
Adjacent SRT cues retain a minimum gap of 100 milliseconds. Pause detection
also requires character timestamps.

## Other commands

All file-oriented commands use `media` as their default input and output folder.
They also scan directories recursively unless `--no-recursive` is supplied.

`export` renders an existing ElevenLabs-style transcript JSON offline:

```powershell
python .\run_toolkit.py export .\media\legacy.json --format srt --format txt
```

`inspect` validates transcript JSON without writing:

```powershell
python .\run_toolkit.py --json inspect .\media\legacy.json
```

`clean` creates an explicitly cleaned JSON derivative, while `transliterate`
converts SRT subtitle text between Uzbek scripts:

```powershell
python .\run_toolkit.py clean .\media\legacy.json
python .\run_toolkit.py transliterate .\media\subtitles.srt --to cyrillic
```

These legacy/offline JSON tools do not change the direct behavior of
`transcribe`.

## Configuration profiles

Optional project configuration is loaded from `elevenlabs-toolkit.toml`.
Start from `elevenlabs-toolkit.toml.example`, then inspect the merged result:

```powershell
python .\run_toolkit.py config show
python .\run_toolkit.py config show --profile short-form
```

Credentials never belong in TOML. Keep `.env` uncommitted.

## Development

The project supports Python 3.10 through 3.14. With the pinned uv version:

```powershell
uv sync --locked --extra stt --extra dev
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

The local launcher also works without installing the package:

```powershell
python .\run_toolkit.py --help
```
