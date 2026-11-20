# ElevenLabs Toolkit

An English desktop app and CLI for transcribing local audio or video with
ElevenLabs, exporting subtitles and transcripts, and converting Uzbek text
between Latin and Cyrillic scripts.

For the desktop app, see [Desktop setup and use](#desktop-setup-and-use).
The command-line quick start follows below.

`transcribe` saves the complete provider response as the reusable transcript
JSON. It does not create a separate cache manifest or persistent lock file.

## Quick start

Create `.env` in the project root:

```dotenv
ELEVENLABS_API_KEY=your_secret_key_here
```

Use the secret starting with `sk_` shown when you create or rotate an
ElevenLabs API key.

Then put media in `media` and run this from the project folder. uv installs
the locked ElevenLabs SDK in the project environment on first use.

```powershell
uv run --locked --extra stt python run_toolkit.py transcribe
```

That command recursively finds supported media below `media`, transcribes every
selected file, and writes one reusable JSON transcript per source to `media`.
Existing JSON transcripts are replaced because `transcribe` means “request a
fresh transcription.”

Once the SDK is installed in the project `.venv`, `python run_toolkit.py transcribe`
also works. The launcher uses that environment if the current Python cannot
find ElevenLabs. To keep both the CLI and desktop dependencies available when
syncing with uv, use `uv sync --locked --extra stt --extra gui`.

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
| Credentials file | `./.env`, with the project’s `.env` as a fallback when launched via `run_toolkit.py` |
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
2. Within a sentence, split at a natural spoken pause longer than 1.0 second,
   using character timings when present and word timings as the fallback. A
   pause split must leave at least two spoken words on both sides.
3. Leave the resulting cue intact when it contains at most 80 characters.
4. For a cue over 80 characters, split first at a semicolon, otherwise at a
   comma, otherwise before a matching structural phrase. Both sides must retain
   at least two spoken words. Within the same priority, prefer the most balanced
   split and repeat the check for any still-overlong result.
5. If no safe boundary exists, leave the overlong cue intact for manual review
   instead of creating an isolated word or cutting at an arbitrary position.
6. Keep at least 80 milliseconds between adjacent cues. Reduce frame padding
   first when space is tight, then adjust both cue boundaries evenly if the
   spoken edges themselves are closer. Join only when no valid cue pair fits.

The same `srt-mini` format can be used with `transcribe`, in which case the
canonical JSON is still written too. Script conversion belongs to `export`;
add `--script latin` or `--script cyrillic` when desired. Cue text remains on
one line unless `--srt-smart-line-breaks` is supplied.

### Live transcription status

During a normal interactive `transcribe` run, one animated status line shows
the current file number, filename, size, phase, and elapsed time. The provider
request is labelled `uploading + transcribing` because the synchronous
ElevenLabs request does not expose separate upload or server-side percentage
progress. Once the response arrives, the status changes to output preparation
and writing.

When output is redirected, the same phase changes are emitted as ordinary
lines without terminal control characters. `--quiet` suppresses them, and
`--json` remains a single clean machine-readable result.

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
--srt-fps FPS
--srt-padding-frames FRAMES
--srt-gap-ms MILLISECONDS
--script source|latin|cyrillic
--clean none|uzbek
--replace TOKEN=TOKEN
```

Timed outputs require word or character timestamps. Character timestamps are
the default. Generated SRT cues start two frames before the first spoken
character and end two frames after the last spoken character; word times remain
the fallback for older JSON files. The default timing frame rate is 30 fps, so
two frames add about 67 milliseconds on each side. At 60 fps, the same setting
adds about 33 milliseconds. Adjacent SRT cues retain a minimum gap of 80
milliseconds, reducing the available padding when necessary. Pause detection
also requires character timestamps.

The defaults can be changed per command, for example:

```powershell
python .\run_toolkit.py export .\media --format srt-mini --srt-fps 60 --srt-padding-frames 2 --srt-gap-ms 80
```

They can also be stored under `[segmentation]` in
`elevenlabs-toolkit.toml` as `srt_fps`, `srt_padding_frames`, and
`srt_gap_milliseconds`.

Speaker metadata remains available in the JSON, but SRT and TXT exports never
add speaker prefixes. Consequently, speaker identifiers are not part of any
subtitle character limit.

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

Converted files use ` LTN` or ` CYR` before the extension. A trailing script
suffix is replaced, so `04 T LTN.srt` becomes `04 T CYR.srt`. This naming
also applies to TXT files converted in the desktop app.

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

The project supports Python 3.10 through 3.15. The local environment uses the
version pinned in `.python-version`. With the pinned uv version:

```powershell
uv sync --locked --extra stt --extra dev
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

## Desktop setup and use

To run the desktop app from the project, include the GUI extra and launch the window:

```sh
uv sync --locked --extra gui --extra stt --extra dev
uv run --locked --extra gui --extra stt --extra dev python run_gui.py
```

After installing the GUI extra, `python run_gui.py` also works from the
project folder: it uses the project `.venv` if the current Python does not
have PySide6 installed.

Choose Transcribe, Convert Script, Export, Clean, Inspect, or Configuration
in the sidebar. Add files or folders with the buttons or drag and drop.
The selected project configuration folder is remembered across launches and
is included in named presets. If a preset's project folder is unavailable,
choose or restore the folder before loading it.
Folder filters apply within selected folders. Preview lists planned destinations
and API requests, including existing-file conflicts under the error policy.
For Transcribe and Export, **Review before running** adds a
confirmation step before the planned job starts.
Changing inputs or settings marks a displayed preview as out of date; run
Preview again to check the new plan. Starting a batch always plans it again.

The batch table lists discovered inputs, output paths, statuses, and errors.
The preview, stop, and start buttons stay visible at the bottom while the main
panel scrolls in a smaller window.
Use **Open Output Folder** to locate the shared output folder, or select a row
first when the batch has several destinations. Use **Save Batch Report** to save the displayed rows as JSON. The report
does not overwrite an existing file. **Stop After File** stops before
the next file; rows not processed are marked **Not run**.

The **Preferences** menu saves, loads and deletes named presets, and resets settings.
Presets contain workflow settings and the selected project folder, but never
the credentials-file field or API key. **Reset to Defaults** clears the current
credentials-file selection as well as workflow settings; saved presets remain.
**Show Effective Settings** displays the
selected profile's settings, workflow overrides, effective values and the source
of each value (default, user, project, profile or workflow override).

For Convert Script and Export, **Preview Content** renders the first selected
SRT, SRT-mini, TXT, or combined TXT output in the details area without writing a file. Long
previews are truncated; use **Preview** to inspect all planned destinations.
For an existing destination, the content preview indicates whether the current
conflict policy would skip or replace it.

After a batch, **Retry Unfinished** retries failed, cancelled and not-run items
with the original settings and destinations. It preserves successful outputs
and skips outputs created or changed since the failed attempt. If the original
batch selected Replace, retry can replace an unchanged existing output for an
unfinished item. The table and saved report retain
successful results from the original batch. Transcription retries require review
because they may make a new paid API request. They use the currently selected
credentials file, so you can correct it before retrying; other processing settings
and destinations remain those of the original batch. A combined export stopped before
publication can be retried without publishing a partial file. Previewing another
plan keeps the prior batch retryable; the button then reads **Retry Previous Batch**.
Switching workflows preserves each workflow's batch results and retry state until
the app closes or a new batch replaces them.

If transcription outputs change after planning, existing outputs are skipped
without connecting to the provider. If a file disappears after a zero-request
plan, the batch reports the change and asks you to start a new plan before any
paid request can occur.

By default, outputs are saved in an `Output` subfolder beside each source.
Clear **Use Output subfolder** and leave the output folder blank to save beside
each source, or choose a custom folder. Transcription writes
`filename.ext.transcript.json` by default. Select additional formats in the
transcription panel, or open an existing transcript in Export to render it
offline. Select **JSON** in the result selector to copy a machine-readable
preview or result from the details area. Batch results include overall status,
written/skipped/failed artifact counts and individual results.

The custom output-folder field is temporarily disabled while **Use Output
subfolder** is selected. Its previous value is restored when the option is turned off. Recursive
folder scans skip nested `Output` folders while it is enabled; files or folders
selected directly can still be used as inputs.
When using a custom output folder inside a recursively scanned input folder,
matching supported files in that folder are included in the scan. The app warns
when the selected inputs and output location can lead to this. To avoid reprocessing
them, choose an output folder outside the input tree or select the input files
explicitly.

The local launcher also works without installing the package:

```powershell
python .\run_toolkit.py --help
```

### Build the desktop application

On macOS, a local build produces `dist/ElevenLabs Toolkit.app`. Use a separate
build environment so packaging dependencies do not change the development
virtual environment. Python 3.13 is used for the packaging toolchain:

```sh
UV_PROJECT_ENVIRONMENT=.venv-build uv sync --locked --extra gui --extra stt --python 3.13
uv pip install --python .venv-build/bin/python pyinstaller==6.22.3
.venv-build/bin/python scripts/build_gui.py
```

To verify the packaged macOS executable without contacting ElevenLabs, run
the following from the project folder. It opens the window offscreen, checks
the available workflows, and converts a temporary SRT file:

```sh
QT_QPA_PLATFORM=offscreen 'dist/ElevenLabs Toolkit.app/Contents/MacOS/ElevenLabs Toolkit' --self-test
```

Open the resulting app from Finder. The app defaults to your home directory as
its configuration context; use **Configuration → Choose Project Folder** to
select a project folder. To use a different credentials file, choose
**Configuration → Choose Credentials File…** and select the existing `.env`
file. The same selection is also available in Transcribe’s advanced recognition
settings.
The app's **Help → Credentials and API Costs** menu also explains this setup.
The file should contain `ELEVENLABS_API_KEY=your-key` on its own line. Use the
secret starting with `sk_` shown when you create or rotate the key. Leave
the field empty to use `.env` in the selected project folder. An API key in the
process environment takes precedence. The GUI does not save the selected
credentials-file path in preferences or presets, so select it again after
restarting if it differs from the project default. On macOS, press
Command-Shift-Period in the file picker to show hidden `.env` files.
The local macOS build is an unnotarized development build. Build on the
destination platform.

The [combined desktop and credential run](https://github.com/M1812M/elevenlabs-speech-to-text/actions/runs/44539457902)
passed Linux and Windows GUI checks and tested the extracted macOS and Windows
packages offline. The local macOS app also opens, resizes, and keeps its bottom
buttons visible in a small window.
Artifacts from successful jobs are available on the corresponding
[GitHub Actions run](https://github.com/M1812M/elevenlabs-speech-to-text/actions).
A [manual desktop checklist](docs/desktop-manual-checks.md) covers visible
window behavior and an offline conversion on a real display.
