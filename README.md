# ElevenLabs Toolkit

An alpha-stage, CLI-first post-production toolkit for transcribing media and
turning transcript JSON into subtitles, text, and DaVinci Resolve artifacts.
ElevenLabs is isolated behind a provider adapter; inspection, export, Uzbek
cleanup, and transliteration work locally without the ElevenLabs SDK.

The project is intentionally free to make breaking changes while its workflows
and data model settle.

## Installation

Python 3.10 or newer is required. Create and activate a virtual environment,
then install only the dependencies needed for the workflow:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Only needed for transcription
python -m pip install "elevenlabs>=2.53,<3"

# Optional development tools
python -m pip install pytest ruff mypy
```

The project is run directly as a Python script; it does not build an EXE.
Run `python .\run_toolkit.py --help` or
`python .\run_toolkit.py COMMAND --help` for the full option set.

## Credentials

Only `transcribe` needs an API key. Set it in the process environment:

```powershell
$env:ELEVENLABS_API_KEY = "your-key"
python .\run_toolkit.py transcribe .\media\clip.wav
```

Or pass a dotenv file explicitly:

```powershell
Copy-Item .\.env.example .\.env
python .\run_toolkit.py transcribe .\media --env-file .\.env
```

The toolkit does not search the package directory or current directory for a
dotenv file. If both are supplied, `ELEVENLABS_API_KEY` from the environment
takes precedence over the explicit file. Keep credentials out of toolkit TOML
configuration and version control.

## Core workflows

Inputs are positional files or directories. For directories, use `--glob` or
`--regex` to select names and `--recursive` to descend into subdirectories.
Selection is explicit: a nonexistent path is an error, not an inferred regex.
Transcription source stems must also be portable output names; unsafe Windows
device names, path punctuation, control characters, and trailing dots or spaces
are rejected during planning before any upload.

### Transcribe media

The default output root is `artifacts`, and the default requested local format is
the canonical provider JSON. A cache manifest is always maintained alongside
it.

```powershell
# JSON transcript and manifest
python .\run_toolkit.py transcribe .\media\clip.wav --env-file .\.env

# Batch transcript plus locally rendered subtitles and text
python .\run_toolkit.py transcribe .\media -o .\artifacts --recursive `
  --glob "*.wav" --format json --format srt --format txt --env-file .\.env

# Request provider-generated formats as well
python .\run_toolkit.py transcribe .\media\interview.mp3 `
  --remote-format pdf --remote-format docx --env-file .\.env
```

Provider options include language, word or character timestamps, diarization,
speaker count, keyterms, audio-event tagging, no-verbatim mode, explicit
retries, and request pacing. Automatic retries default to zero because a retry
may incur another provider charge. `--dry-run` reports both planned requests
and the maximum attempt count after `--retries`.

`no-verbatim` is limited to `scribe_v2`. Keyterms are validated locally against
the provider limits before upload; ElevenLabs currently applies a surcharge to
keyterm-enabled transcription. Timed exports require word or character
timestamps, and pause detection requires character timestamps. The synchronous
CLI deliberately does not expose webhook mode, which returns before a
transcript is available, or separate multichannel response shapes.

### Export transcript JSON

`export` is offline and defaults to SRT. Repeat `--format` to create several
artifacts from one validated transcript.

```powershell
python .\run_toolkit.py export .\artifacts -o .\exports `
  --format srt --format txt --format resolve-edl

python .\run_toolkit.py export .\artifacts -o .\exports `
  --profile social --format social-srt

python .\run_toolkit.py export .\artifacts -o .\exports `
  --format combined-txt --combined-name production.txt
```

Available export formats are `srt`, `txt`, `combined-txt`, `social-srt`,
`resolve-edl`, `cue-index-srt`, and `clean-json`. Generated transcript JSON
such as cleaned derivatives and segmented JSON is excluded from input discovery
by default; use `--include-generated` only when that is intentional. Cache
manifests are metadata, not transcripts, and are always excluded.

Text changes are opt-in and independent of rendering:

```powershell
# Preserve source script but apply the named editorial cleanup
python .\run_toolkit.py export .\artifacts\clip.json --clean uzbek --format srt

# Convert output text and apply a project-specific replacement
python .\run_toolkit.py export .\artifacts\clip.json --script cyrillic `
  --replace "Acme=ACME" --format srt
```

`--script source` is the neutral default. Replacements are literal,
case-insensitive, single-token `SOURCE=TARGET` mappings; the target is emitted
exactly as written and is not transliterated. Uzbek replacement sources may be
written in Latin or Cyrillic script. Cleanup, script conversion, and
replacements never modify the input transcript.

### Inspect, clean, and transliterate

```powershell
# Validate and summarize without writing
python .\run_toolkit.py inspect .\artifacts\clip.json

# Write an explicitly cleaned transcript derivative
python .\run_toolkit.py clean .\artifacts\clip.json --language uzbek -o .\exports

# Convert only SRT text; cue numbers, timing, and HTML tags are preserved
python .\run_toolkit.py transliterate .\exports\clip.srt `
  --to cyrillic -o .\exports\cyrillic

# Use the interactive front end explicitly
python .\run_toolkit.py wizard
```

Clean JSON keeps changed source text and character-timing provenance alongside
the derivative. The wizard always shows a dry-run plan and asks for
confirmation before it dispatches a mutating workflow.

Put global output controls before the command. For example,
`python .\run_toolkit.py --json inspect .\artifacts\clip.json` emits a machine-readable
result, while `-q` and `-v` select quiet and verbose operation.

## Profiles and configuration

Built-in profiles are:

- `standard`: neutral general-purpose segmentation.
- `social`: shorter, denser cues for short-form video.
- `broadcast`: slightly longer cues and broadcast-oriented timing.
- `social-uzbek`: social segmentation plus explicit Uzbek cleanup; it still
  preserves the source script unless another script is requested.

Select one with `--profile`, or set a default in configuration:

```powershell
python .\run_toolkit.py export .\artifacts --profile broadcast --format srt
python .\run_toolkit.py config show
python .\run_toolkit.py config show --profile social
```

Copy [`elevenlabs-toolkit.toml.example`](elevenlabs-toolkit.toml.example) to
`elevenlabs-toolkit.toml` to create a project configuration. The toolkit walks
from the current directory towards the filesystem root and uses the nearest
project configuration. At the same directory level,
`elevenlabs-toolkit.toml` wins over a `pyproject.toml` containing
`[tool.elevenlabs-toolkit]`.

User configuration is optional and lives at:

- Windows: `%APPDATA%\elevenlabs-toolkit\config.toml`
- macOS: `~/Library/Application Support/elevenlabs-toolkit/config.toml`
- Linux: `${XDG_CONFIG_HOME:-~/.config}/elevenlabs-toolkit/config.toml`

Effective values are merged in this order, from lowest to highest precedence:

1. Built-in defaults.
2. User configuration.
3. Nearest project configuration.
4. The selected built-in or custom profile.
5. Explicit command-line options.

This makes project defaults reusable while keeping every one-off CLI decision
authoritative. `config show` prints the resolved values and all available
built-in and custom profiles. The `clean` command uses the same effective
configuration, script, and replacement rules as `export --format clean-json`.

## Safe planning, conflicts, and resume

Every mutating command plans the complete batch before writing. Use `--dry-run`
to print sources, targets, conflicts, required API calls, and the maximum number
of attempts; a dry run never writes files or contacts ElevenLabs. It validates
the path and option plan, while provider response and transcript renderability
are checked immediately before any output is published during execution.

The default `--on-conflict error` stops before work begins. Other policies are:

- `skip`: preserve existing artifacts and report them as skipped without
  parsing or rendering them again.
- `replace`: replace each target atomically through a temporary file in the
  target directory.
- `rename`: choose `name (2).ext`, then `name (3).ext`, and so on.

```powershell
python .\run_toolkit.py export .\artifacts --format srt --dry-run
python .\run_toolkit.py export .\artifacts --format srt --on-conflict rename
```

`rename` is available for deterministic local export, cleanup, and
transliteration jobs. Transcription cache names must remain stable, so
`transcribe` offers only `error`, `skip`, and `replace`. A stale or incomplete
cache combined with `skip` is a preflight conflict; the toolkit will not make a
paid request it cannot cache.

Transcription resumes by default. A cached JSON transcript is reused only when
its manifest matches the provider identity, source name, size and SHA-256,
canonical transcription options, transcript filename, and transcript SHA-256.
Missing local outputs can then be rendered without another paid API request.

Each cache stem has a persistent dot-prefixed advisory lock such as
`.interview.transcription.lock`. Execution rechecks the cache while holding the
lock, preventing two toolkit processes from paying for the same missing cache.
A contender waits up to 300 seconds by default and then reuses the cache created
by the owner; change that bound with `--lock-timeout`. JSON and manifest
publication is rolled back if the pair cannot be completed.

Use `--force-transcribe` to ignore a cache and replace all planned outputs. The
equivalent explicit form is `--no-resume --on-conflict replace`; `--no-resume`
alone will correctly conflict with an existing cache under the default `error`
policy.

The non-overwriting policies use same-directory hard links to close the usual
check/write race. If the destination filesystem cannot support atomic
no-clobber publication, the command fails with a capability error instead of
silently overwriting another process. Transcription probes this capability
under its cache lock before contacting the provider. Use `replace` only when
overwriting is actually intended.

## Output layout and names

Batch output preserves source-relative directories below the common selected
input parent. This prevents files in different source subdirectories from
colliding. Ambiguous mappings, including two files with the same stem in the
same relative directory, are reported during planning.

For a source named `interview.wav` or transcript named `interview.json`, the
standard artifact names are:

| Format | Output name |
| --- | --- |
| Cached transcript | `interview.json` |
| Cache manifest | `interview.manifest.json` |
| Cache lock | `.interview.transcription.lock` |
| SRT | `interview.srt` |
| TXT | `interview.txt` |
| Social SRT | `interview.social.<script>.srt` |
| Resolve markers | `interview.resolve.edl` |
| Cue-index SRT | `interview.cue-index.srt` |
| Clean transcript | `interview.clean.json` |
| Provider segmented JSON | `interview.segmented.json` |
| Transliteration | `interview.latin.srt` or `interview.cyrillic.srt` |

Combined text defaults to `combined.txt`. Provider PDF, DOCX, and HTML outputs
use the source stem and their normal extensions.

## Architecture

```text
CLI
  -> input discovery and configuration
  -> complete JobPlan
  -> application service
  -> canonical Transcript -> language processing -> timed Cue objects
  -> format renderer
  -> atomic output store and JobResult

ElevenLabs adapter -> validated provider payload -> canonical Transcript
```

The main package boundaries are:

- `cli/`: argument parsing, terminal/JSON reporting, and exit codes.
- `application/`: planning, caching, transcription, and export orchestration.
- `models/`: provider-neutral transcripts, timed cues, options, plans, and
  results.
- `providers/`: speech-to-text protocol and the lazy ElevenLabs adapter.
- `segmentation/`: reusable cue construction and preservation rules.
- `renderers/`: SRT, text, and Resolve EDL serialization.
- `languages/`: explicit cleanup and script-conversion profiles.
- `files/`: deterministic discovery and safe atomic output storage.
- `config.py`: layered TOML configuration and named profiles.

Provider responses are normalized once at the adapter boundary. Segmentation
keeps timed words attached to cues so cue timing is derived from the words it
contains, and renderers do not own filesystem policy.

## Development

The committed `uv.lock` is the reproducible development and CI environment.
With uv installed, sync it and run the complete quality gate with locked
dependencies:

```powershell
uv sync --locked --extra stt --extra dev
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest
uv run python run_toolkit.py --help
```

For a pip-only environment, `python -m pip install -e ".[stt,dev]"` remains
supported, but it uses the compatible dependency ranges from `pyproject.toml`
rather than the exact lock.

The direct-script smoke test verifies the launcher can find the local source
tree without creating a distribution or executable.
