# Desktop app manual checks

Use a disposable folder for these checks. Record the app version, operating
system and version, display scaling, and whether each step passes. No API key
or paid transcription request is needed.

1. Open the packaged app from the file manager. Confirm a window appears without
   a terminal and the workflow names and controls are readable.
2. Select **Convert Script**. Add an SRT file and a folder by drag and drop;
   verify both appear as inputs. Use the keyboard to select and remove an input,
   then add it back.
3. Select Cyrillic, choose **Use Output subfolder**, and press **Preview**.
   Check destinations and, with **Preview Content**, check subtitle text and
   timestamps. Confirm neither preview creates or modifies files.
4. Start the conversion. Check the batch statuses, open an output folder from a
   selected row, and save a batch report. Compare the generated subtitles with
   the original; the original must still be present and unchanged.
5. With a batch selected, use the keyboard to move between the navigation,
   options and buttons. Resize the window to fit the available display, scroll
   the main panel, and verify that the bottom processing controls remain visible.
6. Change a setting, save a named preset, close and reopen the app, and load
   the preset. Verify the value and window size are restored. Test **Reset to
   Defaults**, then delete the disposable preset.
7. In **Configuration**, select a disposable credentials file using **Choose
   Credentials File…**. Confirm its path appears there and in Transcribe’s
   advanced recognition settings. Close and reopen the app and check that the
   selected path was not saved. Do not use a real key for this check.
8. In **Transcribe**, widen and narrow the window. Confirm recognition and
   additional outputs appear side by side when there is room, return to a
   single column at narrower widths, and remain usable by keyboard. Open
   **Help → Credentials and API Costs** and check that credential instructions
   and the charge notice are legible. Do not start a transcription unless you
   intend to make a paid request.

For Windows, repeat after extracting the entire packaged application folder;
the executable depends on files in that folder. For macOS, open the extracted
`.app` in Finder. Record any launch or platform security dialog separately.
