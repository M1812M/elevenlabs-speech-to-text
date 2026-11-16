# Desktop parity audit and next iterations

This is an initial control-level inventory, not proof of behavioral parity. The
CLI parser and desktop panels were inspected on 2026-11-15. Unchecked items have
some automated coverage described below, but still need the remaining cases verified.

## CLI controls and desktop mapping

- [ ] Global `--help`, `--version`: Help and About menus. `--json`, `--quiet`,
  `--verbose`: result selector; compare behavior across all workflows.
- [ ] Shared positional inputs: Add Files/Add Folder/drop. CLI defaults to
  `media` and accepts a positional glob; the GUI requires explicit selection and
  offers glob through its filter selector. Check folder selection differences.
- [ ] `--glob`, `--regex`, `--recursive`/`--no-recursive`: filter selector,
  pattern, Include subfolders. Compare nested matching and empty results.
- [ ] `--output-dir`, `--on-conflict`, `--dry-run`, `--fail-fast`: Output
  folder, conflict selector, Preview, Stop on error. Check policies per workflow,
  collisions, writes over inputs, and multi-directory plans.
- [ ] Transcribe `--format`, `--remote-format`: local and provider format
  checkboxes; JSON is mandatory. Verify request counts and partial outputs.
- [ ] Transcribe `--model`, `--language-code`, `--timestamps`, `--diarize`,
  `--audio-events`, `--num-speakers`, `--keyterm`, `--no-verbatim`, `--seed`,
  `--temperature`, `--env-file`, `--retries`, `--retry-backoff`,
  `--request-delay`: Recognition and additional outputs panel. Compare request
  payloads, validation, credential precedence and retry behavior using a fake provider.
- [ ] Transcribe and export `--profile`, `--script`, `--clean`, `--replace`,
  `--pause-detection`, `--srt-smart-line-breaks`, `--srt-fps`,
  `--srt-padding-frames`, `--srt-gap-ms`: text/profile and advanced controls.
  Compare profile precedence and byte-for-byte output with equivalent CLI settings.
- [ ] Export `--format`, `--max-chars-per-line`, `--max-lines`,
  `--max-duration`, `--marker-fps`, `--marker-color`, `--marker-prefix`,
  `--combined-name`, `--include-generated`: Export format, advanced settings,
  and Include generated JSON controls. Compare combined exports and errors.
- [ ] Transliterate `--to`: Convert Script target selector. Verify SRT structure,
  TXT conversion, LTN/CYR names, selection of legacy suffixes and collisions.
  Desktop folder discovery now includes legacy `.latin`/`.cyrillic` inputs,
  matching CLI selection; covered by a worker regression test. Collision
  handling has been compared against the CLI for skip, replace, rename and
  error policies. Other cases above still require parity verification.
  An invalid UTF-8 SRT with an existing destination now fails in both the CLI
  and the GUI under the skip policy, preserving both files. GUI/CLI comparisons
  also verify byte-identical conversions in both directions, replacing LTN/CYR
  suffixes or appending them when absent, while preserving each source.
- [ ] Clean `--language`: GUI is explicitly Uzbek-only. `--profile`, `--script`,
  `--replace`, `--include-generated`: Clean controls. Compare derivatives and errors.
- [ ] Inspect `--include-generated`: Inspect and Include generated JSON. Compare
  malformed and legacy transcript reports.
- [ ] Config `show --profile`: Configuration profile and effective configuration
  display. Equivalent CLI/GUI tests cover project and user precedence, a custom
  profile, available profiles, omission of an environment API key, invalid
  project configuration and an unknown profile. The configuration display also
  refreshes when its profile or project changes. Additional error cases remain.
- [ ] Wizard: Transcribe/Export plus Review before running provides a related
  workflow; defaults and interaction differ. Test preview and confirmation safety.

## Remaining iterations

1. Add behavioral parity tests for each unchecked mapping, fix differences,
   and update each item with evidence. The original roadmap contains historical
   intermediate status and must not be read as the current checklist.
2. Batch table, output-folder access, retry, saved reports and cancellation are
   implemented. Tests cover retrying not-run files after stopping or fail-fast,
   and retaining successful rows in the table and report after retry; verify
   remaining failure and cancellation combinations.
3. Window and workflow settings, recent folders, named presets and reset are
   implemented; credentials are excluded from presets and persisted settings.
4. Advanced sections, tooltips, inline numeric and replacement validation,
   and effective settings with per-value source information are implemented;
   review remaining controls for clarity and validation coverage.
5. Text/subtitle content preview, source/Output-subfolder/custom destinations
   and LTN/CYR naming are implemented; verify remaining combinations.
6. A macOS development app, setup and credential guidance, offscreen keyboard
   and resize checks, and CI jobs exist. Verify native interaction, actual GitHub
   runs, distribution and other platforms.

A separate CI job now installs the GUI extra and runs the desktop tests offscreen;
its first GitHub run remains unverified. Each CI `uv run` explicitly selects its
matrix Python; otherwise the project's Python 3.15 pin replaces the environment
created for another matrix version. An isolated Python 3.13 environment ran all
79 GUI/launcher tests locally. A local macOS bundle builds and starts
offscreen and its self-test runs all six panels plus an SRT conversion in the
packaged process, including an SDK import. The current bundle was rebuilt from the locked GUI/STT dependencies
in an isolated Python 3.13 environment, with PyInstaller installed separately.
Native interaction, distribution and other platforms remain unverified.
An attempted native Cocoa window test on 2026-11-16 reported "no screens
available" in this session; offscreen tests cannot establish visible behavior.
The CI configuration now includes a macOS bundle build, offline self-test and
zipped artifact upload; the job has not yet run on GitHub.
It also includes a Windows GUI test, bundle build, offline self-test and zip
upload; the Windows job has not yet run and its packaging remains unverified.

Recent automated parity evidence: byte-identical SRT, SRT-mini, TXT, Resolve EDL,
cue-index SRT, cleaned JSON and combined text exports; GUI Resolve marker options;
project script override; glob and regex conversion with and without recursion,
including invalid and empty matches; matching transcription
provider options and JSON/SRT bytes using a fake provider; rejection of timed
outputs without timestamps before a provider is constructed; all four export
conflict policies and skipped transcription without a provider; and matching
PDF, DOCX, HTML and segmented JSON outputs and request options with a fake
provider. These cover only the tested combinations, not all unchecked options
above. Additional GUI/CLI tests cover one transient retry and a partial output
set where an existing JSON file is preserved while a missing PDF is requested.
Window-level comparisons also cover profile and pause-detection overrides,
SRT timing and smart line breaks, and clean JSON with script and replacements.
Inspection comparisons cover mixed valid/malformed inputs, read-only behavior,
and the generated-file toggle; export also checks generated-file selection.
Mixed-file export comparisons cover both continuing after an error and stopping
on the first error, including unchanged source files.
Fake-provider transcription comparisons also verify request counts and later
file processing with and without stop-on-error, without sending paid requests.
GUI transcription tests exercise project and explicitly selected credential
files and environment-variable precedence with a fake SDK client; test keys
are absent from the displayed result and persisted settings. A missing project
credentials file can be fixed and retried with a new review, without a network call.
