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

The terminal shows the current photo, completed counts, and a "still working"
update every 30 seconds during long model requests. When the run ends, a short
report shows photo outcomes, catalogue rows, elapsed time, and the Results
location. Keep the terminal open until that report appears.

During model setup and photo processing, the CLI asks the operating system to
prevent automatic idle sleep. It releases that request when the CLI exits. If
the computer cannot provide this protection, the terminal warns you and the
run continues. Keep the laptop lid open: manual sleep or low battery can still
interrupt work. Some screen saver or lock policies may still turn off the
display while processing continues. If the warning appears, adjust the
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
Node ID
Node Type
Node Title
Description
Date Range
Transcription Confidence (%)
Review Flags
Cropped Box Image Link
```

The last column is a controlled relative `HYPERLINK` formula to its file under
`Box Images`. Keep and move the whole batch folder together. Excel,
LibreOffice, and Apple's spreadsheet engine document this function, but CSV
import can vary by application: [Excel](https://support.microsoft.com/en-us/excel/functions/hyperlink-function),
[LibreOffice Calc](https://help.libreoffice.org/latest/en-US/text/scalc/01/04060109.html), and
[Apple spreadsheet functions](https://support.apple.com/guide/functions/functions-overview-ffa083bef0c/web).
Google Sheets cannot open these private local file links and is unsupported.
Opening a link uses the operating system's registered application; revealing
and selecting the file in Finder or Explorer is not promised.

Before importing the catalogue, review rows in their original photo and
left-to-right/top-to-bottom box order:

- Compare every row with its `Cropped Box Image Link` evidence.
- Check every non-empty `Review Flags` value and the
  `Transcription Confidence (%)`. Every transcription still needs human review.
- Keep separate rows for separate physical boxes, even when titles match.

## Compatibility and privacy

Older extracted bundles with private data only in `inputs` and `outputs` remain
supported in place. New legacy runs create `outputs/Batch ...`; existing
top-level catalogues, manifests, and crops are not changed. If both the friendly
and legacy directory pairs contain data, the launcher stops and names both; it
never merges, moves, or deletes private files.

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

Direct CLI runs continue to write one JSON summary to stdout, with live
progress on stderr. Add `--human` to show the same readable final report as
the launcher instead of JSON.

`--results-directory` cannot be combined with `-o`, `-b`, or `--force`.
Without it, every existing direct-CLI default, alias, and Docker workflow stays
available.
