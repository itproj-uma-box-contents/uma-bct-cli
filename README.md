# UMA Box Contents Transcription

This tool turns archive-box photos into a reviewable catalogue on your
computer. By default, the supplied launcher keeps photos, crops, model weights,
and results local. Hosted mode sends sanitized image pixels to the selected API.

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
request before work starts. If the computer rejects it, the terminal shows the
failing operation and the available operating-system error, and the run continues.
On Linux, helper error output and exit codes explain problems such as denied
permissions, a missing system bus, or an unavailable `systemd-inhibit`. A helper
that exits during processing also produces a warning. Windows reports a native
error when supplied, or states that no extended error was provided. These
diagnostics go to stderr and remain visible in captured terminal logs. Keep the laptop lid open: manual sleep or low battery
can still interrupt work. Some screen saver or lock policies may still turn
off the display while processing continues. If a warning appears, adjust the
computer's power settings before a long batch.

The first run needs an internet connection because it installs the verified
private runtime and, in local mode, downloads the large model. The launcher
finishes online preparation before asking for photographs. Later local runs
reuse that runtime and can work offline; hosted runs need internet access.

Supported computers are Windows 10 22H2 or newer on x86-64, macOS 14 or newer
on Intel or Apple silicon, and Linux with glibc 2.28 or newer on x86-64 or
arm64. Windows ARM, musl-based Linux, and older operating systems are not
supported. Allow at least 16 GB of memory and use a trusted, single-user
workstation with enough free disk space.

In local mode, each launcher starts a new, launcher-owned Ollama service on a
temporary numeric loopback port. It never reuses or terminates another
folder's service or an unrelated service on port 11434. The foreground CLI is interrupted first
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
until every source photo and any requested verification have a terminal outcome. Start the same launcher again
to resume it. Photos already committed as complete, partial, or failed are
reused; the photo active at interruption is processed again. Photos added while a batch runs are left for the next launch. A later launch processes
only new or changed photos, using the saved run details to identify earlier work.
Each batch has its own catalogue; older catalogues are not combined or overwritten.
If the current photos span several completed batches, the report lists all relevant
results and includes failures from earlier batches. Recorded catalogue row counts
refer to the original published files, which can also contain photos since removed
from the input folder; the report separately counts transcribed boxes for current
photos. Later spreadsheet edits are not recounted.
Do not remove, rename, or edit an original photo while its batch is unfinished;
if that happens, the CLI preserves progress and explains which inputs to restore.

An unfinished batch also requires its original processing options. If it was
started with the former 80% transcription review threshold, resume it with
`--min-transcription-confidence 80`; the current default is 50%.

You may edit a completed catalogue or save corrected copies and notes beside
it. Later launches preserve those files and report a changed catalogue. Keep the
run details and crop files intact: they record which photographs were processed.

A partial terminal batch publishes its selected catalogue format when it has at least one
usable row. A batch with no usable rows publishes only failed `Run Details.json`.
Leaving an already completed photo set in place does not repeat model work.
Advanced direct-CLI users may pass `--reprocess` with `--results-directory` to
create a separate new batch intentionally.

CSV is the default output. Use `--file-type excel` or explicitly pass
`-o catalogue.xlsx` to select Excel. `--file-type csv` selects CSV explicitly.
Format names are case-insensitive; a format flag and an explicit filename must
agree. Direct runs default to `outputs/catalogue.csv`. For example, to select
Excel and enable verification:

```bash
uma-bct --file-type excel --verification always
```

Existing catalogues are preserved, and unfinished batches keep their original
format when resumed. Choosing a different format does not convert completed
batches; use `--reprocess` to create a new managed batch if needed.

Both formats use these columns:

```text
Unit Type
Unit Number
Description
Date Range
Transcription Confidence (%)
Review Flags
Cropped Box Image Link
Dates
Earliest Date
Latest Date
```

The first four import columns keep their existing order; `Date Range` remains
blank. Date extraction from handwriting adds `Dates`, `Earliest Date`, and
`Latest Date` after the existing review columns. The default is regex extraction;
use `--date-detector none` to disable it or `--ai-dates` (equivalently,
`--date-detector qwen`) to use the selected model: local Qwen normally, or the
configured provider when `--hosted` is present. No dates are shown as `No Dates`,
with `None` for the earliest and latest date. Check extracted dates against the
crop during human review.

An unfinished managed batch keeps its original date extraction method when the
option is omitted. Explicitly selecting a different method requires a new batch.

### Unit Type and Unit Number

`Unit Type` contains all printed text visible on the box label in reading
order, except for the unit-number phrase. Printed names, headings, logos,
labels such as `Creator:`, and record identifiers are included. Visual line
breaks and repeated whitespace are converted to single spaces in the catalogue cell.

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

### Review results and open crops

The catalogue adds `Low confidence` to scores below 50% by default;
a score of exactly 50% does not receive that flag. Set
`--min-transcription-confidence INTEGER` (or `-c INTEGER`) to choose a review
threshold from 0 to 100. This flags rows for review and keeps every transcription
in the catalogue. The separate `--min-detector-confidence` option controls the
OpenCV detection threshold on a 0-to-1 scale and defaults to 0.60.

Excel workbooks contain clickable crop filenames in `Cropped Box Image Link`.
Each link opens the image file using your system's associated application in
desktop Excel or LibreOffice. Keep the workbook and crop folder together when
moving results. If your spreadsheet application cannot open a local link, open
the named crop directly from the batch's `Box Images` folder. Apple Numbers and
Google Sheets cannot open these private local-file links. CSV cannot save column
widths or text wrapping; managed CSV batches retain their controlled relative
`HYPERLINK` formulas for compatible Excel and LibreOffice imports.

### Optional transcription verification

Verification is disabled by default. Use `--verification always` to add a
second reading of the same images within the same run using the same Qwen model.
Use `--verification ask` to request an interactive choice after the first pass:
choose **y** for this run or Enter to finish. The model receives
a distinct checking prompt without seeing the first answer. Only crops with
substantive differences receive a third request. Detection is not repeated.

A correction requires the third reading to corroborate an earlier candidate.
Unresolved differences, whole or partial content deletions, and ambiguous matches between boxes
keep the original text and receive review flags. Printed text, identifiers, and
unit number are reconciled together, including handwritten identity claims. Every row still needs human review:
agreement between model readings does not prove correctness or raise confidence.

Run details preserve the original and comparison readings. Managed runs checkpoint
completed readings; restart the launcher to resume an interrupted check. A failed
verification request preserves the first
reading and produces a partial result with a review flag.

Users can select `--verification always`, `--verification never`, or
`--verification ask`. Only the CLI option enables verification for a new run;
`UMA_BCT_VERIFICATION` and saved `.uma-bct-settings.json` preferences do not enable it.
An unfinished batch keeps its recorded verification choice; finish it
before changing that option. Older unfinished
batches continue without verification. Already completed batches are reused;
use `--reprocess` to intentionally start another complete run.

`--json`, redirected stdin/stderr, and unattended runs never prompt. They verify
only when verification was requested with `always`. Verification improves the opportunity
to find discrepancies; measured accuracy gains require human-reviewed references.

Before importing the catalogue, review each row against its crop and source
photo. The catalogue is sorted by `Unit Type` and `Unit Number`:

- Compare every row with its `Cropped Box Image Link` evidence.
- Check every non-empty `Review Flags` value and the
  `Transcription Confidence (%)`. Every transcription still needs human review.
- Keep separate rows for separate physical boxes, even when their unit type and
  number match.

## Compatibility and privacy

### Hosted model opt-in

Set `UMA_BCT_API_KEY` and `UMA_BCT_API_MODEL` in the process environment and
pass `--hosted` to use a hosted vision model. The flag is the explicit privacy
opt-in: sanitized detection and crop PNGs leave the workstation over HTTPS.
Original photo files and EXIF metadata are not sent. Only direct OpenAI
(`openai/`), Claude (`anthropic/`), and Gemini (`gemini/`) API origins are
accepted. Confirm your organisation permits this disclosure and review the
chosen provider's data handling and charges before adding private photos.
Never put a key in command arguments, a committed file, or a shell history line.

For a direct CLI install, use `uv sync --locked --extra hosted` (or install
`uma-bct-cli[hosted]`), then set the environment variables and run
`uv run --locked uma-bct --hosted --json --force`. The `uma-bct --hosted`
flag selects the model; without it, local Qwen runs even when API variables
are set.
For the supplied launcher, its verified dependencies already include the hosted
client; setting the key in the launcher's environment makes it pass `--hosted`
to the CLI. For example, in Bash:

```bash
export UMA_BCT_API_MODEL=anthropic/your-vision-model
printf "API key: "
read -r -s UMA_BCT_API_KEY
printf "\n"
export UMA_BCT_API_KEY
sh run-uma-bct.sh
```

On Windows PowerShell, set `$env:UMA_BCT_API_MODEL` and
`$env:UMA_BCT_API_KEY` in the same terminal before running
`./run-uma-bct.ps1`. An empty key keeps the local Qwen workflow even when a
hosted model name is set. In direct CLI use, `--hosted` requires a nonempty key
and a provider-qualified model; `--model` remains a local Ollama option.
Hosted setup checks the installed
client without making a paid API call, and the launcher does not start Ollama.
The first image request verifies the key and model with the provider.

Hosted requests report the exception type and HTTP status, with fixed diagnostic
hints that do not expose provider reply text or credentials. Temporary HTTP
500/502/503/504 failures receive at most two retries with exponential backoff and
jitter. Authentication, model-selection, and quota errors are not retried.
Gemini requests use the versioned `/v1beta` API base; the manifest retains the
provider's HTTPS origin. LiteLLM's feedback banner is suppressed to keep JSON
stdout clean.

After model preflight, the CLI prints `Selected model: <model> (<backend>)` on
stderr before processing photos. For hosted models, preflight checks configuration
and dependencies; the first API request still determines model access.

The same variables work with the opt-in Compose service:
`docker compose --profile hosted run --rm hosted`. That service has no Ollama
dependency. Hosted APIs may omit token probabilities; the corresponding
transcription confidence is treated as unavailable and sent for review. A
Gemini request that explicitly rejects optional log probabilities with HTTP 400
is retried once without that parameter. Unavailable confidence is currently
exported as `0` with review flags; it does not prevent CSV output or mean that
the transcription has been measured as completely wrong. Available scores use
the least probable content token, and are not calibrated OCR accuracy scores
that can be compared directly between different model families. A
managed batch resumes only with the same backend, provider, model, and HTTPS
origin. Changing models after completion creates a separate batch.

The shared detector uses each model family's coordinate convention: Qwen uses
`[xmin, ymin, xmax, ymax]`, while Gemini uses `[ymin, xmin, ymax, xmax]`, both
normalized to 0-1000. Gemini coordinates are converted to X-first corners before
pixel cropping; see Google's [bounding-box guide](https://ai.google.dev/gemini-api/docs/image-understanding#bounding-box-detection).
The `qwen` stage name also appears in hosted manifests. Its detector confidence
of `1.0` is a placeholder for accepted geometry, not a measured probability of
correct detection. `--min-detector-confidence` filters OpenCV proposals only.
Thin fragments and empty or unreadable crop responses still require review.

Hosted-model prompts request one rectangle per substantially visible physical
box front and exclude shelves, side panels, labels, and thin edge fragments.
Their edge rules omit narrow, textless box fragments; the local Qwen prompts
retain their original wording. OpenAI and Anthropic use X-first coordinates,
while Gemini uses Y-first coordinates with the same hosted selection rules.
Transcription can return `{"boxes": []}` for a non-box crop; this produces a
processing warning without a catalogue row. A genuine box with unreadable text
retains its row for review. These prompts do not impose a fixed box count or
guarantee correct localization; assess the latest run's crops and manifest rather
than a crop directory from an earlier run.

The current `Transcription mismatch` flag also covers `needs_review`, including
unavailable token confidence. It can therefore appear alongside `Low confidence`
on a correctly transcribed synthetic label. Review flags and successful API
access do not establish transcription accuracy. `partial` status means some
processing work remained unresolved; confidence review flags alone do not make
an image partial.

### Test Gemini without paying

Use `gemini/gemini-3.5-flash-lite` for the verified Free Tier smoke-test example.
On 2026-10-07, a Windows run using a Google AI Studio key without billing
successfully generated a text reply and processed one invented box image through
detection, transcription, and CSV export. The exported Unit Type was `1960.0009`
and Unit Number was `0004`, with unavailable confidence exported as `0` and
review flags. This verifies that workflow on that project; it does not establish
reliable handwriting recognition or guarantee future service availability.
That smoke test predates the Gemini coordinate-order correction; rerun it after
updating, and use a non-square image with several boxes to assess localization.

Google currently lists [`gemini-3.5-flash-lite`](https://ai.google.dev/gemini-api/docs/pricing#gemini-3.5-flash-lite)
with free input and output tokens. It accepts image input and returns text.
Free-tier access and quotas depend on your project; check its **Billing Tier**
and [active rate limits](https://ai.google.dev/gemini-api/docs/rate-limits)
in Google AI Studio before testing. Keep the project on the Free Tier to avoid
paid API usage. The model here reads images; it is not the separate image
*generation* model. See the [model capabilities](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite/)
and [billing guide](https://ai.google.dev/gemini-api/docs/billing).

Google also lists `gemini-3.8-flash` with a free tier, but the Windows generation
probes timed out in both PowerShell and Python while model lookup and token
counting succeeded. The same key and network generated successfully with
`gemini-3.5-flash-lite`. Use the verified model for this smoke test rather than
treating the 3.8 free-tier listing as a successful test result.

Google states that Free Tier content may be used to improve its products.
**Use only invented test images with a Free Tier key. Never use UMA client photos
or other confidential material for this test.** Obtain a Gemini API key in
[Google AI Studio](https://aistudio.google.com/apikey). The key belongs to a
project, so confirm that project's tier before using it. See Google's
[API key guide](https://ai.google.dev/gemini-api/docs/api-key).

From the repository root, this Bash example makes one synthetic archive-box
image in a private temporary directory, then runs the full detection and
transcription workflow. Enter the key at the hidden prompt; it will not appear
in the shell history:

```bash
uv sync --locked --extra hosted
test_root="$(mktemp -d)"
export UMA_BCT_API_MODEL=gemini/gemini-3.5-flash-lite
printf 'Gemini API key: '
read -r -s UMA_BCT_API_KEY
printf '\n'
export UMA_BCT_API_KEY

uv run --locked python - "$test_root" <<'PY'
from pathlib import Path
import sys
from PIL import Image, ImageDraw, ImageFont

root = Path(sys.argv[1])
image = Image.new("RGB", (1200, 800), "#e8e4db")
draw = ImageDraw.Draw(image)
draw.rectangle((120, 100, 1080, 700), fill="#ad9270", outline="#574936", width=10)
draw.rectangle((240, 255, 960, 565), fill="white", outline="black", width=5)
font = ImageFont.load_default(size=58)
draw.text((280, 315), "1960.0009", font=font, fill="black")
draw.text((280, 425), "Unit 4", font=font, fill="black")
image.save(root / "synthetic-box.png")
PY

uv run --locked uma-bct --hosted --json \
  --path-to-inputs "$test_root/synthetic-box.png" \
  --path-to-outputs "$test_root/catalogue.csv" \
  --path-to-box-images "$test_root/crops"
```

The JSON summary names the actual model and the run manifest. Open
`$test_root/catalogue.run.json` to check `model.backend` is `hosted` and
`model.model` is `gemini/gemini-3.5-flash-lite`; review `catalogue.csv` and its crop
if a row was produced. A synthetic drawing may be missed or transcribed
imperfectly, so inspect `status`, `failures`, and `images` in the manifest
rather than treating catalogue accuracy as the API connection check. If a
warning says `Hosted model request failed`, check the key, model name, and quota
in AI Studio. One photo can make several API requests through detection,
transcription, and fallback. When finished, run
`unset UMA_BCT_API_KEY UMA_BCT_API_MODEL` in Bash.

On Windows, create an invented PNG in Paint and save it in a new temporary
folder. In PowerShell, set the model and enter the key without putting it in
command history, then point the same CLI at that PNG:

```powershell
uv sync --locked --extra hosted
$env:UMA_BCT_API_MODEL = 'gemini/gemini-3.5-flash-lite'
$secret = Read-Host 'Gemini API key' -AsSecureString
$env:UMA_BCT_API_KEY = [System.Net.NetworkCredential]::new('', $secret).Password
uv run --locked uma-bct --hosted --json `
  --path-to-inputs 'C:\path\to\invented-box.png' `
  --path-to-outputs 'C:\path\to\gemini-test.csv'
```

Replace both paths with your own temporary paths. The CSV and its run manifest
must not already exist, unless you use `--force`. The
[direct CLI options](#compatibility-and-privacy) are the same on every platform.

If the API is failing, first isolate access from image processing. With the key
still set in the same PowerShell terminal, these requests send only an invented
text prompt to Google and bypass LiteLLM and Ollama:

```powershell
$headers = @{ 'x-goog-api-key' = $env:UMA_BCT_API_KEY }
$models = Invoke-RestMethod `
  -Uri 'https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000' `
  -Headers $headers
$models.models | Where-Object {
  $_.name -match 'flash' -and $_.supportedGenerationMethods -contains 'generateContent'
} | Select-Object name, displayName

$modelId = $env:UMA_BCT_API_MODEL -replace '^gemini/', ''
$body = @{
  contents = @(@{ parts = @(@{ text = 'Reply with the word OK.' }) })
  generationConfig = @{ thinkingConfig = @{ thinkingLevel = 'LOW' } }
} | ConvertTo-Json -Depth 10
$reply = Invoke-RestMethod -Method Post `
  -Uri "https://generativelanguage.googleapis.com/v1beta/models/${modelId}:generateContent" `
  -Headers $headers -ContentType 'application/json' -Body $body `
  -TimeoutSec 60 -ErrorAction Stop
$reply.candidates.content.parts.text
```

The model list establishes model names, not free-tier quota eligibility. Confirm
the selected model's active quota in AI Studio. A 404 can indicate an unavailable
model or an incorrect endpoint; 401/403 requires checking the key and access;
429 requires checking the project's minute, token, and daily limits; and 503
requires waiting or trying another available free-tier vision model. A project
with no free quota for the selected model cannot be repaired by rapid retries.
Each photo normally needs one detection call plus one call per crop, with
additional calls for fallbacks and retries.

Successful API calls can still produce bad crops or incorrect unit readings.
Missing-box suggestions use internal gaps between the transcribed numbers in
each matching Unit Type group. For example, readings `0001`, `0003`, and `0005`
produce suggestions `0002` and `0004`; incorrect readings `0002` and `0003`
contain no gap and cannot recover the original sequence. Correct localization
and review of visible unit numbers come first. Use `--no-missing-box-detection`
when inspecting raw transcription rows separately from gap suggestions.

After a successful direct request, rerun the full CLI on one invented image and
inspect the JSON summary and CSV. The `qwen` detector/transcriber names refer to
the shared components; a selected hosted backend sends both stages
to Gemini without contacting Ollama. Do not enable LiteLLM debug dumps for
private photographs or keys. When finished:

```powershell
Remove-Item Env:UMA_BCT_API_KEY
Remove-Item Env:UMA_BCT_API_MODEL
Remove-Variable headers, secret -ErrorAction SilentlyContinue
```

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

Direct CLI runs show the same readable final report as the launcher. Interrupted
direct runs exit cleanly (130 for Ctrl+C); use managed batches for resumable work. Add
`--json` when a script needs one JSON summary on stdout; progress and warnings
still go to stderr. The `--human` flag remains accepted for older commands.

`--results-directory` cannot be combined with `-o`, `-b`, or `--force`.
Without it, direct output paths and existing aliases remain available. The bundled
Docker workflow also produces Excel by default.

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
