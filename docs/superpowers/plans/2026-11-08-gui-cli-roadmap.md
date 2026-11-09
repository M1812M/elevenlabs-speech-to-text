# Iterative GUI Expansion Implementation Plan

> For agentic workers: use superpowers:executing-plans to implement one milestone at a time. Each milestone ends with a usable application and verification.

**Goal** — Expose the toolkit's existing CLI workflows through the English desktop interface in manageable iterations.

**Architecture** — Keep the sidebar, workflow options, input table, details area, and execution bar. Reuse application planners, executors, models, discovery, configuration, and atomic writes. Extract CLI-owned processing into shared application functions when required.

**Tech stack** — Python 3.15.2, PySide6 Essentials, existing ElevenLabs SDK and pytest/Ruff/mypy tooling.

**Spec** — The approved English GUI roadmap in this chat, refined against the current GUI and CLI implementation on 2026-11-08. This document is a planning deliverable.

## Constraints and design decisions

- English labels throughout. Preserve the existing CLI contract.
- Audio transcription defaults to JSON only.
- GUI outputs default to the source directory. Folder selection is recursive.
- Default conflict policy is Skip. Replacement is explicit.
- Uzbek cleanup and script conversion remain explicitly Uzbek-only.
- One immutable batch runs at a time. Settings and selected sources are captured before execution.
- Offline workflows do not require credentials or instantiate the provider.
- Existing configuration precedence remains authoritative. Explicit GUI options become overrides.
- Each workflow owns its session selection and options. The shared table and execution bar display the selected workflow.
- Additional controls go into the relevant workflow panel or its Advanced section.

## Current implementation gaps

The current GUI is a preview. Stop is disabled, the progress bar is never updated, and Preview only lists source files. Placeholder navigation falls through to Transcribe. Replace is ignored during transcription. Conversion writes directly and aborts the whole batch on one exception. Inputs and settings can change during a running job.

These gaps must be addressed before expanding the number of workflows.

## Code organization

- Keep run_gui.py as the entry point. Convert src/elevenlabs_toolkit/gui.py into a gui package with __init__.py retaining main, window.py for the shell, jobs.py for worker lifecycle, and workflows.py for explicit workflow dispatch and state.
- Add workflow panels under gui only when implemented. Shared input-table and options controls remain reusable.
- Keep processing in application. Extract conversion from cli/transliterate.py and inspection from cli/inspect.py into shared application modules, with CLI handlers delegating to them.
- Reuse JobPlan, PlannedArtifact, ArtifactResult, JobResult, InputSpec, ExportOptions, and existing configuration types. Avoid parsing CLI output or building a generic plugin system.

## Milestone 1 — Complete the shared execution workflow

**Outcome** — Transcribe and Convert Script become dependable foundations for all later workflows.

- [ ] Add explicit workflow identifiers and dispatch. Disable unavailable navigation entries. Capture an immutable job request before starting.
- [ ] Use a table with Source, Output, and Status columns. Preview displays actual planned destinations, conflicts, skipped inputs, and API request counts without writes or API calls.
- [ ] Use shared discovery, normalize and deduplicate paths, exclude generated conversion outputs from folder scans, and keep direct selection available. Run large scans in the background.
- [ ] Extract conversion into the application layer. Use existing atomic writes and source-overwrite checks. Continue after per-file errors and report each outcome.
- [ ] Implement Skip and Replace consistently for both active workflows. Replace the radio button with a conflict-policy selector. Create the provider only when a planned request is required.
- [ ] Resolve output targets before execution. Use stem.latin.srt, stem.cyrillic.txt, and original-audio-filename.transcript.json. Extend planner support for explicit destinations while keeping CLI naming defaults. Preview existing legacy JSON outputs as distinct from the new GUI target.
- [ ] Wire completed-file progress and activity indication during provider requests. Implement stop after the current file, controlled window closing, and worker cleanup before another job starts.
- [ ] Add per-workflow state and freeze editing during execution. Show readable per-file errors in the details area.
- [ ] Verify unsupported navigation cannot trigger an API call, all-skipped jobs need no credentials, replacement applies correctly, sources cannot be overwritten, and a failed middle file does not prevent later files.
- [ ] Verify stopping starts no further file and closing cannot destroy a running worker. Manually check the Mac window.
- [ ] Commit the milestone after its checks pass.

## Milestone 2 — Offline JSON export

**Outcome** — A saved transcript can be turned into useful subtitles and text directly from the GUI.

**Existing CLI source** — export, plan_exports, execute_export, ExportOptions.

- [ ] Enable Export in the sidebar with JSON file/folder selection.
- [ ] Add format checkboxes for SRT, SRT-mini, and TXT. Default to SRT, allow multiple outputs, and require at least one.
- [ ] Add the existing profile selector and Output Script selector with Source, Latin, and Cyrillic. Default to the effective configured profile and Source.
- [ ] Group source-adjacent export plans by source directory and combine them for a full destination/conflict preview. Preserve existing renderer suffixes.
- [ ] Use the shared execution controller and per-file results. Preserve existing timestamp validation and legacy JSON handling.
- [ ] Compare GUI and CLI exports with equivalent inputs and options. Verify multiple formats, nested directories, invalid JSON, legacy word timings, and no provider access.
- [ ] Commit the milestone after its checks pass.

## Milestone 3 — Inspect and Clean

**Outcome** — Validate transcripts and prepare cleaned derivatives within the same application.

**Existing CLI sources** — inspect and clean.

- [ ] Extract inspection into a shared service. Enable Inspect with language, validity, duration, word and segment counts, speakers, and character-timing counts in the details area.
- [ ] Enable Clean for JSON inputs. Use existing Uzbek cleanup with optional output script and a token-replacement editor.
- [ ] Reuse the current single-token replacement validation. Save cleaned JSON derivatives beside sources using the existing .clean.json convention and conflict policy.
- [ ] Add Include Generated Files to JSON discovery controls so cleaned derivatives can be explicitly inspected or exported.
- [ ] Verify inspection writes nothing, malformed files show per-file errors, cleanup preserves usable timing metadata, and unsupported language handling matches the CLI.
- [ ] Commit the milestone after its checks pass.

## Milestone 4 — Subtitle and editing exports

**Outcome** — Expose the toolkit's subtitle tuning and DaVinci Resolve outputs.

- [ ] Extend Export formats with Resolve EDL, cue-index SRT, combined TXT, and cleaned JSON.
- [ ] Add Advanced groups for Segmentation, Timing, Text Processing, and Resolve.
- [ ] Expose smart line breaks, character/line limits, minimum/maximum duration, pause and gap settings, word limits, and speaker-change splitting from the existing options/configuration.
- [ ] Expose SRT FPS, frame padding, minimum cue gap, and Resolve marker FPS, colour, and prefix.
- [ ] Reuse cleanup and replacements controls where supported. Validate option combinations through the application models.
- [ ] Produce one combined TXT per source directory, sorted by source path. Show member files and the configurable combined filename in Preview.
- [ ] Verify configured settings reach the renderers, invalid timing combinations fail before writes, and combined output membership is deterministic.
- [ ] Commit the milestone after its checks pass.

## Milestone 5 — Advanced transcription

**Outcome** — Make existing provider options available through the Transcribe panel.

- [ ] Add Advanced groups for Recognition, Speakers, and Request Behavior.
- [ ] Expose model, language, timestamp granularity, diarization, speaker count, audio events, keyterms, verbatim behavior, seed, and temperature using current CLI defaults and validation.
- [ ] Expose retries, backoff, and request delay. Keep retries at zero by default and show maximum planned API attempts before Start.
- [ ] Add Additional Outputs for local SRT, SRT-mini, TXT, Resolve EDL, and cue-index SRT, plus provider PDF, DOCX, HTML, and segmented JSON.
- [ ] Keep JSON mandatory and the sole default output. Clarify in the interface which additional formats require a fresh provider request.
- [ ] Reuse applicable export settings. Preserve the CLI's timestamp dependencies, conflict rules, and partial-existing-output behavior.
- [ ] Verify request options with a fake provider. Test invalid combinations, no credentials, transient failures, and output conflicts without paid calls.
- [ ] Commit the milestone after its checks pass.

## Milestone 6 — Shared advanced controls and configuration

**Outcome** — Complete the remaining CLI controls without changing the window structure.

- [ ] Add mutually exclusive glob/regex filters, recursion toggle, and generated-file inclusion where supported.
- [ ] Expose Error and supported Rename conflict policies, optional output directory, and fail-fast behavior. Preserve source-adjacent output as the default.
- [ ] Add Settings for credential-file selection, project configuration context, available profiles, and the effective merged configuration.
- [ ] Keep credentials out of configuration displays and diagnostics. Provide availability status without exposing key contents.
- [ ] Reuse the CLI wizard's preview-before-run flow as the existing GUI workflow. Use the details area for verbose diagnostics and structured result information.
- [ ] Verify configuration precedence, explicit overrides, directory context, filter validation, and output paths across multiple source folders.
- [ ] Commit the milestone after its checks pass.

## Verification and review focus

For each milestone, run targeted behavior tests, the existing suite, Ruff, formatting checks, and mypy. Exercise normal CLI entry points after shared-service changes. Use automated Qt checks for lifecycle and dispatch, plus a manual Mac check for file dialogs, drag-and-drop, resizing, and keyboard use.

Pay particular attention to these cases in their owning milestone:

1. Selecting an unavailable workflow must never submit audio (Milestone 1).
2. Input aliases, generated files, and colliding targets must not duplicate requests or overwrite sources (Milestone 1).
3. Failed input between valid files must retain an accurate batch result (Milestones 1–3).
4. Missing legacy timing data must follow existing parser/renderer behavior (Milestones 2–4).
5. Partial output sets and retry settings must match previewed provider-request behavior (Milestone 5).

## Recommended next implementation unit

Implement Milestones 1 and 2 as separate commits and reviewable deliverables. Together they provide a stable desktop workflow and the most useful missing capability: exporting existing JSON into SRT, SRT-mini, and TXT.

Later milestones can be selected individually. This plan does not start their implementation.

## Implementation progress — 2026-11-08

The first goal continuation added stop-between-files handling, worker lifecycle controls, disabled placeholder navigation, atomic conversion writes, per-file exception handling, consistent transcription replacement, lazy provider construction, and destination previews. Offline SRT/SRT-mini/TXT export is now available in the Export panel.

Verification: the existing 347 tests plus three worker regressions passed together (350 tests). A fourth targeted regression verifies multi-format offline export. Ruff and mypy pass. The Export panel initialized successfully using Qt's offscreen platform.

Still pending in the shared workflow: immutable preflight plans with complete conflict checking, table-based per-file state, background discovery, generated-file exclusion, per-workflow session state, shared conversion extraction, and the agreed suffixed transcription filenames. The preview currently derives filenames from existing planner naming. Export still needs profile/script controls. Clean, Inspect, configuration, advanced exports, transcription options, and advanced selection/execution controls remain pending. The overall goal is active.

### Goal continuation 2

Implemented Clean, Inspect, and Configuration navigation. Export and Clean now expose profile, output script, and token replacement controls; Export also exposes optional cleanup. Configuration displays effective merged values and profiles, and selects the project context used by jobs. Each workflow retains its own file selection.

Extracted SRT conversion and transcript inspection from CLI handlers into shared application services. Added immutable DesktopSettings and PreparedFile records, and shared preparation for preview and execution. Full-selection preflight now rejects duplicate destinations and outputs that overwrite any selected input. TXT conversion planning uses the same suffix normalization as SRT.

Verification: 359 tests pass, Ruff passes, formatting passes for 84 files, and mypy passes for 52 source files. New cases cover GUI/CLI cleanup parity, read-only inspection, malformed-input continuation, profile override precedence, destination collisions, and workflow selection persistence. Reviewed an offscreen Clean-panel render.

Remaining: advanced export formats/options, combined exports with proper grouping and cancellation, advanced transcription options and provider formats, all conflict/filter controls, background discovery, per-file status table, credential-file selection, full per-workflow option state, and the previously agreed suffixed audio JSON naming. Current JSON naming remains the CLI-compatible stem.json until the shared planner supports explicit GUI naming. Full goal completion is unproven and the goal remains active.

### Final verification

The desktop exposes Transcribe, Convert Script, Export, Clean, Inspect, and
Configuration. The CLI wizard maps to Review before running on Transcribe and
Export. The Help menu and report selector expose help, version, JSON, Quiet,
and Verbose behavior. The source list and scrollable details area show the
selection, planned destinations, and per-file results.

Folder filters, generated-file inclusion, output locations, conflict choices,
and stop-on-error are available in the shared controls. File discovery and
planning run in a background worker. The workflow keeps its own selection and
common options. Audio defaults to JSON beside the source and offline export
uses existing JSON without contacting ElevenLabs.

Reviewed the CLI help for every workflow against the desktop controls. A
simulated provider test exercises drag and drop, read-only preview, JSON-only
transcription, and offline export through the window. Other desktop tests
cover cancellation, confirmation, conflict naming, filtering, and reporting.
The native macOS window launched and all six panels initialized. The full
suite passed with 372 tests, along with Ruff, formatting, mypy, and the
offline lockfile check.
