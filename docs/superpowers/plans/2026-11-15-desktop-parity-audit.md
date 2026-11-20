# Desktop parity audit and next iterations

This is an initial control-level inventory, not proof of behavioral parity. The
CLI parser and desktop panels were inspected on 2026-11-15. Unchecked items have
some automated coverage described below, but still need the remaining cases verified.
For an option-by-option control map, see [CLI and desktop control map](../../cli-gui-controls.md).

## CLI controls and desktop mapping

- [ ] Global `--help`, `--version`: Help and About menus. `--json`, `--quiet`,
  `--verbose`: result selector. Conversion tests verify errors remain visible
  in quiet mode and verbose mode includes a traceback. Inline preflight errors
  are structured JSON when JSON mode is selected. An About-dialog test compares
  its version with the CLI, and the local bundle contains the package metadata;
  compare remaining behavior across workflows.
- [ ] Shared positional inputs: Add Files/Add Folder/drop. CLI defaults to
  `media` and accepts a positional glob; the GUI requires explicit selection and
  offers glob through its filter selector. Tests drop the same FLAC file and its
  parent folder onto both input areas and verify one fake-provider request and
  an unchanged source. Check other folder selection differences.
- [ ] `--glob`, `--regex`, `--recursive`/`--no-recursive`: filter selector,
  pattern, Include subfolders. Window-level GUI/CLI tests cover nested regex
  matching with and without recursion and generated JSON in Inspect and Clean;
  compare remaining filter and empty-result combinations.
- [ ] `--output-dir`, `--on-conflict`, `--dry-run`, `--fail-fast`: Output
  folder, conflict selector, Preview, Stop on error. Check policies per workflow,
  collisions, writes over inputs, and multi-directory plans.
- [ ] Transcribe `--format`, `--remote-format`: local and provider format
  checkboxes; JSON is mandatory. Fake-provider tests compare all five local
  output formats and all provider formats with the CLI; verify remaining request
  and partial-output combinations.
- [ ] Transcribe `--model`, `--language-code`, `--timestamps`, `--diarize`,
  `--audio-events`, `--num-speakers`, `--keyterm`, `--no-verbatim`, `--seed`,
  `--temperature`, `--env-file`, `--retries`, `--retry-backoff`,
  `--request-delay`: Recognition and additional outputs panel. Compare request
  payloads, validation, credential precedence and retry behavior using a fake provider.
  GUI and CLI boundary tests cover speaker count, seed and temperature values,
  including non-finite temperature, as well as retry counts, backoff and delay.
- [ ] Transcribe and export `--profile`, `--script`, `--clean`, `--replace`,
  `--pause-detection`, `--srt-smart-line-breaks`, `--srt-fps`,
  `--srt-padding-frames`, `--srt-gap-ms`: text/profile and advanced controls.
  Compare profile precedence and byte-for-byte output with equivalent CLI settings.
- [ ] Export `--format`, `--max-chars-per-line`, `--max-lines`,
  `--max-duration`, `--marker-fps`, `--marker-color`, `--marker-prefix`,
  `--combined-name`, `--include-generated`: Export format, advanced settings,
  and Include generated JSON controls. A GUI/CLI regression verifies combined
  exports use the same natural order for numbered filenames, including separate
  default Output folders compared with CLI exports per folder. A window-level
  comparison also covers a custom combined filename under skip, replace, error
  and rename conflict policies, preserving both inputs. Compare remaining
  combined-export cases and errors. When combined text and cleaned JSON are
  selected together, a GUI/CLI regression compares both output files with
  automatic and explicit Uzbek cleanup; the effective-settings display also
  identifies the automatic cleanup. GUI and CLI both reject cleaned JSON with
  cleanup explicitly disabled before writing any selected outputs; the GUI
  displays this conflict beside Cleanup before launching the worker.
- [ ] Transliterate `--to`: Convert Script target selector. Verify SRT structure,
  TXT conversion, LTN/CYR names, selection of legacy suffixes and collisions.
  Desktop folder discovery now includes legacy `.latin`/`.cyrillic` inputs,
  matching CLI selection; covered by a worker regression test. Collision
  handling has been compared against the CLI for skip, replace, rename and
  error policies. Other cases above still require parity verification.
  An invalid UTF-8 SRT with an existing destination now fails in both the CLI
  and the GUI under the skip policy, preserving both files. GUI/CLI comparisons
  also verify byte-identical conversions in both directions, replacing LTN/CYR
  suffixes or appending them when absent, while preserving each source. A
  multi-folder test also verifies distinct same-named inputs retain their folder
  structure in a chosen output directory, and that preview makes no writes.
  A window-level test also verifies preview and execution reject an output hard
  linked to its input, including under Replace, without modifying either path.
- [ ] Clean `--language`: GUI is explicitly Uzbek-only. `--profile`, `--script`,
  `--replace`, `--include-generated`: Clean controls. GUI/CLI comparisons verify
  byte-identical output for Uzbek and unspecified language, and rejection of
  English before writing a derivative; sources remain unchanged. Window-level
  comparisons also cover skip, replace, error and rename when a cleaned output
  already exists. Compare other errors.
- [ ] Inspect `--include-generated`: Inspect and Include generated JSON. Compare
  malformed and legacy transcript reports. GUI/CLI tests compare malformed,
  generated, segment-only, untimed-word and punctuation-token inputs, including
  read-only behavior; further input variants remain unchecked.
- [ ] Config `show --profile`: Configuration profile and effective configuration
  display. Equivalent CLI/GUI tests cover project and user precedence, a custom
  profile, available profiles, omission of an environment API key, invalid
  project configuration and an unknown profile. The configuration display also
  refreshes when its profile or project changes. Additional error cases remain.
- [ ] Wizard: Transcribe/Export plus Review before running provides a related
  workflow; defaults and interaction differ. Tests verify the prepared export
  plan is executed only after approval, and declining the GUI transcription
  review defaults to Cancel, makes no provider request and leaves the batch retryable.
  Visible native dialog interaction remains unverified.

## Remaining iterations

1. Add behavioral parity tests for each unchecked mapping, fix differences,
   and update each item with evidence. The original roadmap contains historical
   intermediate status and must not be read as the current checklist.
2. Batch table, output-folder access, retry, saved reports and cancellation are
   implemented. Tests cover retrying not-run files after stopping or fail-fast,
   retaining successful rows in the table and report after retry, and declining
   the review without writing while reporting cancellation in JSON. A fake-provider
   test confirms stopping during a request finishes that file, makes no request
   for the next file, and a reviewed retry requests only the unfinished file.
   A combined export that fails on malformed JSON can be retried after the input
   is repaired; the combined output and both batch rows then reflect success.
   For script conversion, retrying a failed Replace operation replaces an
   unchanged pre-existing regular file; a created or changed target is skipped.
   Tests cover all three cases. Export retries also preserve successful outputs
   and replace unchanged failed targets; tests cover targets changed after failure.
   Fake-provider tests also verify transcription retry replaces an unchanged failed
   target after review, and makes no further request when that target was changed.
   Retry review reports whether each existing output will be replaced or skipped,
   updates the request count, and cancels if the output plan changes during review.
   If a destination changes after review and would require an extra request,
   the affected transcription is stopped for a fresh review before contacting the provider.
   A window-level fake-provider test also covers a stopped file whose previously
   skipped output was removed: retry reviews the newly required request count.
   Partial-output tests verify that retry preserves an already-written JSON
   transcript while replacing a failed subtitle or provider-generated PDF.
   Stopped-before-start conversion, export and fake-provider transcription
   retries replace unchanged targets and preserve targets changed after stopping.
   Conversion and export tests also verify retry does not replace an output that
   another selected input was changed to link to after the original failure.
   Additional partial transcription output combinations need verification.
   Saved reports before and after retry retain both file statuses; verify
   remaining failure and cancellation combinations.
3. Window and workflow settings, recent folders, named presets and reset are
   implemented; credentials are excluded from presets and persisted settings.
   Invalid saved workflow controls are rejected before applying a preset and
   fall back to defaults on launch; tests cover both recovery paths.
4. Advanced sections, tooltips, inline numeric and replacement validation,
   inline Resolve colour, combined-filename and transcription timestamp validation, and effective settings
   with per-value source information are implemented;
   review remaining controls for clarity and validation coverage.
5. Text/subtitle content preview, source/Output-subfolder/custom destinations
   and LTN/CYR naming are implemented. New GUI workflows default to an Output
   subfolder; saved source-adjacent preferences remain in effect. Combined TXT
   content preview uses the export renderer and identifies its source files;
   tests compare preview and output in text and JSON modes with automatic cleanup.
   Window tests
   cover the default for conversion, fake-provider transcription, export and
   cleaning, including unchanged inputs. Content preview reports output
   conflicts even when the selected format has no text preview. Verify remaining combinations.
6. A macOS development app, setup and credential guidance, offscreen keyboard
   and resize checks, and CI jobs exist. On the local macOS offscreen platform,
   workflows fit an 800-pixel-wide window without horizontal scrolling; Windows
   CI reported wider controls and can require horizontal scrolling. The bottom action buttons stay
   visible while the main panel scrolls. An offscreen check requesting a 320×200
   window also verifies the buttons remain inside the actual window, which Qt
   may enlarge to its minimum size. The user confirmed the packaged macOS window opens
   and resizes on 2026-11-17, and confirmed on 2026-11-18 that the bottom buttons
   remain visible in a small window. The user reported that opening the output folder did
   not work. The button now handles an unselected shared destination; native
   file-manager behavior still needs verification. Offscreen tests also check
   that a path containing a newline remains intact after switching workflows.
   Verify the remaining native interactions, actual GitHub runs,
   distribution and other platforms.

Offscreen keyboard tests also traverse from the navigation to the Start button
with Tab in every workflow, including in a 600×300 window. Multiline fields move
focus on Tab rather than trapping keyboard navigation, and focused controls are
checked against both the main and nested scrollable viewports. Separate tests
cover traversal through expanded advanced settings in Transcribe and Export.

A separate CI job installs the GUI extra and runs the desktop tests offscreen.
Each CI `uv run` explicitly selects its
matrix Python; otherwise the project's Python 3.15 pin replaces the environment
created for another matrix version. The full suite, including GUI tests, passed
locally on Python 3.13 and the development environment on 2026-11-16 (521 tests
on each at that point). On 2026-11-18, 703 tests passed on Python 3.13.14
and on the Python 3.15.2 development environment,
and the rebuilt local macOS bundle passes its offline self-test. A local macOS bundle builds and starts
offscreen and its self-test runs all six panels, verifies a read-only JSON
content preview, and performs an SRT conversion in the packaged process,
including an SDK import. The current bundle was rebuilt from the locked GUI/STT dependencies
in an isolated Python 3.13 environment, with PyInstaller installed separately.
The user reported that the packaged macOS app opens and its window resizes on
2026-11-17. Other native interaction, distribution and other platforms remain unverified.
The latest local macOS app was also archived with `ditto`, extracted into a
temporary directory, and passed its offline self-test from the extracted copy.
The current archive passed the same extraction and offline self-test again on
2026-11-17. This verifies archive contents, not native launch or platform security approval.
An attempted native Cocoa window test on 2026-11-16 reported "no screens
available" in this session; offscreen tests cannot establish visible behavior.
GitHub CI run 44220137184 passed all Python 3.10–3.15 quality jobs, Linux GUI tests, and the
macOS bundle build, offline self-test and artifact upload. The downloaded CI
artifact from an earlier run matched GitHub's SHA-256 digest and its unpacked
macOS app passed the offline self-test locally; this is not a visible native-interaction test.
An earlier Windows job failed in GUI tests before packaging: three JSON-path
assertions assumed unescaped path separators, while layout checks encountered
larger Windows minimum widths and offscreen focus. Local changes parse JSON paths,
check actual button visibility at platform minimum sizes, and reveal the focused
control's center in scroll areas. The subsequent
[desktop fix branch run](https://github.com/M1812M/elevenlabs-speech-to-text/actions/runs/44241772427)
passed Windows and Linux GUI tests, the Python 3.10–3.15 quality matrix,
and offline self-tests of both extracted desktop archives. The desktop fix
and credential changes pass 708 tests locally. The later
[combined changes run](https://github.com/M1812M/elevenlabs-speech-to-text/actions/runs/44539457902)
also passed all nine jobs, including both extracted package self-tests.
The clean-export parity check also now reads Cyrillic JSON explicitly as UTF-8
and verifies the converted text, avoiding reliance on the Windows default encoding.
All 313 current desktop and launcher tests passed locally on Python 3.13 with
unspecified `Path.read_text` and `Path.write_text` encodings forced to Windows-1252.
An earlier run also passed on Python 3.15 with the same simulation. This
simulation does not cover other file APIs or replace a Windows CI run.
In a separate Python 3.13 environment without PySide6, the CLI help command
worked and the current suite passed 367 tests with four GUI-related skips.

Recent automated parity evidence: byte-identical SRT, SRT-mini, TXT, Resolve EDL,
cue-index SRT, cleaned JSON and combined text exports; GUI Resolve marker options;
project script override; glob and regex conversion with and without recursion,
including invalid and empty matches; matching transcription
provider options and JSON/SRT bytes using a fake provider; rejection of timed
outputs without timestamps before a provider is constructed, and matching
untimed plain-text transcription when pause detection is disabled; all four export
conflict policies and skipped transcription without a provider; and matching
PDF, DOCX, HTML and segmented JSON outputs and request options with a fake
provider. These cover only the tested combinations, not all unchecked options
above. Additional GUI/CLI tests cover one transient retry and a partial output
set where an existing JSON file is preserved while a missing PDF is requested.
Window-level comparisons also cover profile and pause-detection overrides,
SRT timing and smart line breaks, and clean JSON with script and replacements.
GUI/CLI boundary comparisons cover line limits, duration, subtitle FPS, frame
padding and subtitle gap, including an inherited minimum-duration conflict.
Subtitle timing overrides are validated against values inherited from the selected
profile before starting, with errors shown beside the relevant control.
Inspection comparisons cover mixed valid/malformed inputs, read-only behavior,
and the generated-file toggle; export and clean also check generated-file selection.
Mixed-file export comparisons cover both continuing after an error and stopping
on the first error, including unchanged source files. With combined text also
selected, both GUI and CLI leave the combined output unpublished when a source
is malformed, under both stop-on-error settings.
Fake-provider transcription comparisons also verify request counts and later
file processing with and without stop-on-error, without sending paid requests.
Additional fake-provider comparisons check FLAC glob and regex filtering with and without
recursive folder scanning, matching selected inputs and output bytes with the CLI.
GUI transcription tests exercise project and explicitly selected credential
files and environment-variable precedence with a fake SDK client; test keys
are absent from the displayed result and persisted settings. A missing project
credentials file can be fixed and retried with a new review, without a network call.

On 2026-11-20, Configuration gained a credentials-file picker synchronized
with Transcribe; automated tests check both controls, confirm the path is not
persisted, and run a fake-SDK transcription after selecting the file through
Configuration, including the environment-variable precedence case. Transcription settings were split into recognition, output,
and advanced groups. An offscreen main-window test checks the recognition and
output groups switch between two columns and one column as the window is
resized. The 711-test suite and the rebuilt macOS app's offline self-test passed
locally. Native file-manager behavior and native verification of the new layout
remain outstanding; the earlier report that the output folder did not open is
not considered resolved solely by mocked tests.
A subsequent offscreen visual inspection showed the enlarged maximum height
alone did not allocate enough space to transcription settings. The panel now
receives at least 350 pixels inside the main scrolling area, with the processing
buttons outside it; a regression test and the existing small-window and keyboard
checks pass. The full suite passed 712 tests, and the rebuilt Mac bundle passed
its offline self-test. Native visual verification remains outstanding.
The 2026-11-20 rebuilt macOS archive was also extracted into a temporary
folder and its packaged application passed the offline self-test. This does not
verify Finder integration or visible desktop interaction.
The five local transcription output comparisons now also exercise disabling
speaker identification, enabling filler removal, and multiple keyterms; they
compare the full fake-provider request options and resulting bytes with the CLI.
Windows CI for commit 0513007 revealed that a scroll area's content may retain
its wide minimum width after the window narrows, preventing the transcription
layout from returning to one column. The layout now uses the visible main
viewport width. A regression simulates content wider than the viewport; local
keyboard and small-window checks pass. The [subsequent CI run](https://github.com/M1812M/elevenlabs-speech-to-text/actions/runs/44720692490)
passed all nine jobs, including Windows GUI tests and the extracted Windows
package self-test.

The batch workspace redesign was exercised on a native Cocoa display on
macOS 27.2.1. At the requested narrow and wide sizes (620 × 390 and
1250 × 820), the primary action stayed inside the window and the batch table
had no horizontal overflow. Keyboard workflow navigation, a no-write SRT
preview, conversion with unchanged input, persistence of a disposable preset,
and visibility of the Configuration credential picker passed. The native
file-manager API accepted the selected output folder URL. The packaged app
also passed its native offline self-test. These checks used temporary inputs
and made no API requests.
