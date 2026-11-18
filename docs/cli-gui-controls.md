# CLI and desktop control map

This inventory maps CLI capabilities to desktop controls. A mapping establishes
where to find a capability; it does not by itself prove every behavior or option
combination equivalent. See the [parity audit](superpowers/plans/2026-11-15-desktop-parity-audit.md)
for current test evidence and remaining verification.
An automated check compares the long options in the CLI parser with this map;
it checks coverage, not the accuracy of each mapping.

## General and shared controls

| CLI | Desktop control or equivalent |
| --- | --- |
| `--help`, `--version` | Help → Using the Toolkit; Help → About ElevenLabs Toolkit. The GUI help is a summary, not the CLI help text. |
| `--json`, `--quiet`, `--verbose` | Result selector: JSON, Quiet, Verbose. Quiet suppresses progress updates but keeps the final GUI result visible. |
| Positional files and folders | Add Files, Add Folder, or drag and drop. The CLI defaults to `media` when omitted; the GUI requires a selection. |
| Positional glob; `--glob` | Choose Glob in the folder filter and enter the pattern. Directly selected files are not filtered. |
| `--regex` | Choose Regular expression and enter the pattern matched against paths relative to each selected folder. |
| `--recursive`, `--no-recursive` | Include subfolders. |
| `--output-dir` | Clear Use Output subfolder, then enter or browse to an output folder. Leave it blank to save beside sources. New GUI workflows default to an `Output` subfolder; the CLI defaults to `media`. |
| `--on-conflict` | Conflict selector: Skip existing, Replace existing, Report conflict, or Choose a new name. Rename is unavailable for transcription, matching the CLI's supported policies. The GUI defaults to Skip. |
| `--dry-run` | Preview. It plans paths, conflicts and API requests without writing outputs or making requests. Preview Content separately renders the first supported text or subtitle output without writing. |
| `--fail-fast` | Stop on error. |
| `--include-generated` | Include generated JSON in Export, Clean and Inspect. |

## Transcribe

Choose **Transcribe**. JSON is always produced; additional formats are optional.

| CLI | Desktop control |
| --- | --- |
| `--format` | Local output checkboxes: SRT, SRT-mini, TXT, Resolve EDL and cue-index SRT. |
| `--remote-format` | Provider output checkboxes: PDF, DOCX, HTML and segmented JSON. |
| `--model` | Model. |
| `--language-code` | Language code; blank means automatic detection. |
| `--timestamps` | Timestamps: Character, Word or None. |
| `--diarize`, `--no-diarize` | Identify speakers. |
| `--audio-events`, `--no-audio-events` | Tag audio events. |
| `--num-speakers` | Speaker count. |
| `--keyterm` | Keyterms, one per line. |
| `--no-verbatim` | Remove fillers and false starts. |
| `--seed` | Seed. |
| `--temperature` | Temperature. |
| `--env-file` | Credentials file. The selection is not saved in settings or presets. |
| `--retries` | Retries. |
| `--retry-backoff` | Retry backoff (seconds). |
| `--request-delay` | Request delay (seconds). |

## Text settings shared by Transcribe and Export

| CLI | Desktop control |
| --- | --- |
| `--profile` | Profile. Project default follows the selected configuration context. |
| `--script` | Output script (Uzbek): profile default, source, Latin or Cyrillic. |
| `--clean` | Cleanup: profile default, None or Uzbek editorial cleanup. |
| `--replace` | Override profile replacements and enter one `TOKEN=TOKEN` pair per line. |
| `--pause-detection`, `--no-pause-detection` | Advanced subtitle and Resolve settings → Pause detection: profile default, On or Off. |
| `--srt-smart-line-breaks` | Advanced → Balance subtitle lines. |
| `--srt-fps` | Advanced → Subtitle FPS. |
| `--srt-padding-frames` | Advanced → Subtitle padding (frames). |
| `--srt-gap-ms` | Advanced → Subtitle minimum gap (ms). |

## Export

Choose **Export** and select one or more format checkboxes.

| CLI | Desktop control |
| --- | --- |
| `--format` | SRT, SRT-mini, TXT, combined TXT, Resolve EDL, cue-index SRT and cleaned JSON checkboxes. |
| `--max-chars-per-line` | Advanced → Maximum characters per line. |
| `--max-lines` | Advanced → Maximum lines. |
| `--max-duration` | Advanced → Maximum duration (seconds). |
| `--marker-fps` | Advanced → Resolve marker FPS. |
| `--marker-color` | Advanced → Resolve marker colour. |
| `--marker-prefix` | Advanced → Resolve marker prefix. |
| `--combined-name` | Advanced → Combined text filename. |

Export also provides the shared text settings above. When saving beside sources
or in `Output` subfolders, combined text is grouped per source directory; when
using a custom folder, the selected inputs share one combined file.

## Other workflows

| CLI | Desktop control or equivalent |
| --- | --- |
| `transliterate --to latin` / `--to cyrillic` | Convert Script → Latin / Cyrillic. The GUI accepts TXT as well as SRT. |
| `clean --language uzbek` | Clean uses Uzbek editorial rules; it also offers Profile, Output script and token replacements. |
| `inspect` | Inspect displays read-only transcript summaries and validation errors. |
| `config show --profile` | Configuration → Profile and effective configuration; Choose Project Folder sets the configuration context. |
| `wizard` | Transcribe or Export with Review before running provides the preview-and-confirm flow. Its interaction and defaults differ from the terminal wizard. |

The GUI also offers batch status, retry, saved reports, persistent preferences,
named presets and per-workflow selections. See the desktop section of the
[README](../README.md#desktop-setup-and-use) for use and packaging details.
