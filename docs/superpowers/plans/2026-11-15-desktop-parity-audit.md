# Desktop parity audit and next iterations

This is an initial control-level inventory, not proof of behavioral parity. The
CLI parser and desktop panels were inspected on 2026-11-15. Each mapped item
still needs equivalent-input tests, including output bytes and failure cases.

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
  Desktop folder discovery excludes legacy `.latin`/`.cyrillic` outputs whereas
  the CLI includes them; establish and implement consistent intended behavior.
- [ ] Clean `--language`: GUI is explicitly Uzbek-only. `--profile`, `--script`,
  `--replace`, `--include-generated`: Clean controls. Compare derivatives and errors.
- [ ] Inspect `--include-generated`: Inspect and Include generated JSON. Compare
  malformed and legacy transcript reports.
- [ ] Config `show --profile`, `profiles`: Configuration profile and effective
  configuration display. Verify project context, overrides and secret redaction.
- [ ] Wizard: Transcribe/Export plus Review before running provides a related
  workflow; defaults and interaction differ. Test preview and confirmation safety.

## Remaining iterations

1. Add behavioral parity tests for each unchecked mapping, fix differences,
   and update each item with evidence. The original roadmap contains historical
   intermediate status and must not be read as the current checklist.
2. Introduce a per-file table, output-folder access, retry of failed files,
   saved batch reports and explicit stop-after-current-file behavior.
3. Persist window and per-workflow settings, recent folders and named presets;
   exclude credentials and provide reset.
4. Improve option grouping, inline validation, tooltips and profile provenance.
5. Add content previews and explicit source/Output-subfolder/custom destinations;
   reconcile generated-file discovery with LTN/CYR naming.
6. Build and verify a distributable desktop app, document setup and credentials,
   exercise keyboard/resizing/platform behavior, and run GUI tests in CI.

GUI tests are currently skipped in CI because the quality job does not install
the GUI extra. Desktop packaging and native platform checks remain unverified.
