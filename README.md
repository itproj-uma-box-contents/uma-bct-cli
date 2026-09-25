# UMA Box Contents Transcription

This tool turns archive-box photos into a reviewable catalogue on your
computer. The supplied launcher keeps photos, crops, model weights, and results
local; it does not upload them.

## Run the tool

1. Download and unzip `archive-box-transcription.zip` into a private folder
   that is not synced to iCloud, OneDrive, Dropbox, or a network drive.
2. Start UMA BCT:

   - **Windows:** double-click `Run UMA BCT.cmd`.
   - **macOS or Linux:** open Terminal in the unzipped folder and run
     `sh run-uma-bct.sh`.

3. Put the photos directly in `Add Photos Here`, then return to the launcher
   and press Enter if prompted.

The launcher prints the photo folder's full path. If it does not open
automatically, open that path yourself before pressing Enter.

The terminal updates one live progress display with the current photo and box,
completed photo count, and elapsed time. Captured logs show short milestones
instead. When the run ends, a short report shows photo outcomes, catalogue
rows, elapsed time, and the Results location. Keep the terminal open until that
report appears.

During model setup and photo processing, the CLI requests protection against
automatic idle sleep and releases it when the CLI exits. macOS confirms the
request before work starts. If the computer rejects it, the terminal warns you
and the run continues. Keep the laptop lid open: manual sleep or low battery
can still interrupt work. Some screen saver or lock policies may still turn
off the display while processing continues. If a warning appears, adjust the
computer's power settings before a long batch.

The first run needs an internet connection because it installs the verified
private runtime and downloads the large model. The launcher finishes all online
preparation before asking for photographs. Later runs reuse that runtime and
can work offline.

Supported computers are Windows 10 22H2 or newer on x86-64, macOS 14 or newer
on Intel or Apple silicon, and Linux with glibc 2.28 or newer on x86-64 or
arm64. Windows ARM, musl-based Linux, and older operating systems are not
supported. Allow at least 16 GB of memory and use a trusted, single-user
workstation with enough free disk space.

Each launcher starts a new, launcher-owned Ollama service on a temporary
numeric loopback port. It never reuses or terminates another folder's service
or an unrelated service on port 11434. The foreground CLI is interrupted first
when the launcher is stopped; cleanup is bounded and targets only processes
that launcher started.

## Photos and results

Supported input types are `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`, `.webp`,
`.heic`, `.heif`, `.dng`, `.avif`, `.bmp`, and `.gif`. Extensions are matched
case-insensitively. HEIC and HEIF containers use their primary image; animated
or multi-page formats use the first frame or page. DNG files are developed from
the full RAW image rather than a lower-resolution embedded preview. Adding a
photo to `Add Photos Here` is a local file copy, not an internet upload.

Every run creates or resumes a readable batch:

```text
Archive Box Transcription/
  Add Photos Here/
  Results/
    Batch 2026-09-11 14-30-05/
      Catalogue.csv
      Run Details.json
      Box Images/
        PXL_230150462 - Box 01.png
        PXL_230150462 - Box 02.png
```

An unfinished batch instead contains `Run Progress.json` and has no catalogue
until every source photo has a terminal outcome. Start the same launcher again
to resume it. Photos already committed as complete, partial, or failed are
reused; the photo active at interruption is processed again. Do not add,
remove, rename, reorder, or edit source photos while a batch is unfinished.
The launcher reports the affected batch if its input snapshot no longer
matches.

A partial terminal batch publishes `Catalogue.csv` when it has at least one
usable row. A batch with no usable rows publishes only failed `Run Details.json`.
Leaving an already completed photo set in place does not repeat model work.
Advanced direct-CLI users may pass `--reprocess` with `--results-directory` to
create a separate new batch intentionally.

The CSV columns are:

```text
Unit Type
Unit Number
Description
Date Range
Transcription Confidence (%)
Review Flags
Cropped Box Image Link
```

### Unit Type and Unit Number

`Unit Type` contains all printed text visible on the box label in reading
order, except for the unit-number phrase. Printed names, headings, logos,
labels such as `Creator:`, and record identifiers are included. Visual line
breaks and repeated whitespace are converted to single spaces in the CSV cell.

`Unit Number` contains only the detected box or unit number, normalized to four
digits. The tool recognizes `Unit`, `Box`, `Box No.`, `Box Number`, and `No.`
labels. A sequence total in a phrase such as `Box 7 of 9` is not part of the
unit number.

For example:

| Printed label text | Unit Type | Unit Number |
| --- | --- | --- |
| `HOME, Roderick Weir` / `2008.0036` / `Unit 7` | `HOME, Roderick Weir 2008.0036` | `0007` |
| `BURNET, Sir Frank Macfarlane` / `1986.0107 (1989.0034)` / `Unit 22` | `BURNET, Sir Frank Macfarlane 1986.0107 (1989.0034)` | `0022` |
| `Printing & Allied Trades` / `Employers' Association` / `of Victoria` / `1975.0064 Unit 8` | `Printing & Allied Trades Employers' Association of Victoria 1975.0064` | `0008` |
| `Victorian Chamber of Mines` / `107/56` / `Box 7 of 9` | `Victorian Chamber of Mines 107/56` | `0007` |

When the unit phrase shares a line with other text, only that phrase is
removed. For example, `1975.0064 Unit 8` contributes `1975.0064` to `Unit Type`.
A label containing only `Unit 8` has a blank `Unit Type` and is flagged for
review.

The model's `typed` transcription is the source for `Unit Type`. Handwriting
and explicit content or condition annotations are instead considered for
`Description`; identifier-only and unit-only text is not repeated there.
Malformed dotted identifiers such as `1960.009` remain visible in `Unit Type`
but receive `Ambiguous identifier` and `Transcription mismatch` review flags.
If more than one valid identifier is visible, all printed text remains in
`Unit Type` and the row receives `Multiple identifiers`.

### Missing-box detection

Missing-box detection is enabled by default. After the photographed boxes are
transcribed, rows are sorted case-insensitively by `Unit Type` and then
numerically by `Unit Number`. If two consecutive rows have the same complete
`Unit Type` and their unit numbers contain an internal gap, the tool inserts a
blank review row for each missing number.

For example, observed rows `HOME, Roderick Weir 2008.0036 / 0001` and
`HOME, Roderick Weir 2008.0036 / 0003` produce an inserted `0002` row. The
inserted row keeps the shared `Unit Type`, has `Missing Box` in `Review Flags`,
and leaves transcription confidence, description, and cropped-image link
blank.

Detection is deliberately conservative:

- It only fills gaps between observed unit numbers; it does not infer missing
  boxes before the lowest or after the highest observed number.
- Both surrounding rows must have the same complete `Unit Type`, ignoring
  letter case, and valid numeric unit numbers.
- Gaps larger than 100 units are not filled.
- Differing printed headings, identifiers, or other label text create different
  `Unit Type` groups, so no missing row is inferred between them.
- Inserted rows are review suggestions, not proof that a physical box is
  missing.

Advanced CLI users can disable this behavior with
`--no-missing-box-detection` (or its shorter alias `--no-missing-box`).

The last column is a controlled relative `HYPERLINK` formula to its file under
`Box Images`. Keep and move the whole batch folder together. Excel,
LibreOffice, and Apple's spreadsheet engine document this function, but CSV
import can vary by application: [Excel](https://support.microsoft.com/en-us/excel/functions/hyperlink-function),
[LibreOffice Calc](https://help.libreoffice.org/latest/en-US/text/scalc/01/04060109.html), and
[Apple spreadsheet functions](https://support.apple.com/guide/functions/functions-overview-ffa083bef0c/web).
Google Sheets cannot open these private local file links and is unsupported.
Opening a link uses the operating system's registered application; revealing
and selecting the file in Finder or Explorer is not promised.

Before importing the catalogue, review each row against its crop and source
photo. The CSV is sorted by `Unit Type` and `Unit Number`:

- Compare every row with its `Cropped Box Image Link` evidence.
- Check every non-empty `Review Flags` value and the
  `Transcription Confidence (%)`. Every transcription still needs human review.
- Keep separate rows for separate physical boxes, even when their unit type and
  number match.

## Compatibility and privacy

Older extracted bundles with private data only in `inputs` and `outputs` remain
supported in place. New legacy runs create `outputs/Batch ...`; existing
top-level catalogues, manifests, and crops are not changed. If both the friendly
and legacy directory pairs contain data, the launcher stops and names both; it
never merges, moves, or deletes private files.

An older offline runtime without the live display dependency still processes
photos and shows short milestone lines. A fresh extracted bundle installs the
live display before photographs are added.

Only numeric loopback Ollama endpoints are accepted by default. Direct CLI use
still defaults to `127.0.0.1:11434`. A developer may use an approved managed
endpoint only with HTTPS and `UMA_BCT_ALLOW_REMOTE_OLLAMA=1`; image pixels then
leave the workstation, so written approval is required. The bundled Compose
service opts into only its exact internal `ollama` hostname and disables
Ollama Cloud. Cloud-tagged models are unsupported. Anyone managing a separate
local Ollama service for direct CLI use must disable its cloud features so a
loopback API cannot proxy private images off the workstation.

Advanced users can prepare custom locations with:

```bash
uma-bct setup --path-to-inputs "/private/photos" --results-directory "/private/results"
uma-bct --path-to-inputs "/private/photos" --results-directory "/private/results"
```

Direct CLI runs show the same readable final report as the launcher. Add
`--json` when a script needs one JSON summary on stdout; progress and warnings
still go to stderr. The `--human` flag remains accepted for older commands.

`--results-directory` cannot be combined with `-o`, `-b`, or `--force`.
Without it, every existing direct-CLI default, alias, and Docker workflow stays
available.

Current limitations:

- Results depend on Qwen accurately transcribing all printed text and
  preserving its reading order.
- Labels such as `Carton 7`, `Container 7`, and `Package 7` are not recognized
  as unit numbers.
- Values such as `Unit 8A`, ranges, and unclear or crossed-out unit numbers are
  not accepted as ordinary unit numbers.
- Only malformed dotted standard-looking identifiers are explicitly detected.
  Malformed UMA-style or slash-style values may still be accepted as arbitrary
  identifiers.
- The parser cannot semantically distinguish incidental text such as
  `Research Unit 8` from an actual unit claim; `Unit 8` would be removed from
  `Unit Type`.
- Printed logos, headings, and labels such as `Creator:` are included because
  the rule is all printed label text.
