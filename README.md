# UMA Box Contents Transcription

Turn archive-box photos into a catalogue for human review. The default launcher keeps
photos, crops, model weights, and results on your computer. Optional hosted mode sends
sanitized image pixels to the selected API.

[Run the ZIP](#run-the-tool) · [Review results](#catalogue-and-human-review) · [Hosted
models](#hosted-model-opt-in) · [Develop](#development) · [Build the
ZIP](#build-the-client-deliverables)

## CLI arguments

Add these options after `uma-bct`, `sh run-uma-bct.sh`, or the Windows launcher. For
example, `sh run-uma-bct.sh --format excel` creates an Excel catalogue for a new
batch. Format names ignore case: `--format Excel` works too.

| Option | What it does | Default / example |
| --- | --- | --- |
| `--format csv` / `--format excel` | Choose CSV or an Excel `.xlsx` workbook for a new batch. `--file-type` is the same option. | CSV; `--format excel` |
| `--verification never\|always\|ask` | Skip verification, run a blind second reading, or ask after transcription. A third reading is used only for substantive differences. | `never`; `ask` prompts only interactively |
| `--ai-dates` | Extract dates with the selected local or hosted model. | Local regex extraction |
| `-c`, `--min-transcription-confidence INTEGER` | Flag scores below this percentage; keep the rows. Accepts 0–100. | `50`; `-c 80` |
| `--min-detector-confidence FLOAT` | Filter OpenCV box proposals on a 0–1 scale. `--min-confidence` is the older alias. | `0.60` |
| `--no-missing-box-detection` | Disable suggested rows for internal unit-number gaps. `--no-missing-box` is an alias. | Suggestions enabled |
| `-d`, `--detector qwen\|opencv` | Choose the primary detector; OpenCV is experimental. | `qwen` |
| `-t`, `--transcriber qwen` | Select the shared vision transcriber, including when a hosted model is selected. | `qwen` is the only choice |
| `-m`, `--model MODEL` | Select a local Ollama model. For hosted models, use `UMA_BCT_API_MODEL`. | Pinned `qwen3-vl:8b-instruct-q4_K_M` |
| `--hosted` | Use the provider and key from `UMA_BCT_API_MODEL` and `UMA_BCT_API_KEY`. Direct CLI runs require this opt-in. | Local; ZIP launchers add it when a key is set |
| `--request-timeout SECONDS` | Set a positive timeout for each model request. | `UMA_BCT_OLLAMA_TIMEOUT_SECONDS`, otherwise `600` |
| `--reprocess` | Create a separate managed batch for already completed photos; preserve earlier results. Requires managed mode. | Reuse completed work |
| `--json` | Write one JSON summary to stdout; send progress and warnings to stderr. Never prompt. | Readable report |
| `--human` | Keep the readable report; accepted for older commands. Cannot be combined with `--json`. | Already the default |
| `-h`, `--help` | Show help and exit. ZIP launchers do not download or start a runtime for this. | No processing |
| `--version` | Show the version and exit without processing or ZIP runtime setup. | No processing |

These path and setup options are for the **direct CLI only**. The ZIP launchers use
`Add Photos Here` and `Results` (or the existing legacy layout) and reject custom
paths, direct output options, and `setup`.

| Direct CLI option | What it does | Default / example |
| --- | --- | --- |
| `-i`, `--path-to-inputs PATH` | Read one image or a flat directory of images. | `inputs/` |
| `-o`, `--path-to-outputs PATH` | Set the CSV or Excel filename; `--path-to-output` is an alias. Its extension must agree with an explicit format option. | `outputs/catalogue.csv`; `-o catalogue.xlsx` selects Excel |
| `-b`, `--path-to-box-images PATH` | Set the review-crop folder; `--work-dir` is an alias. | `crops/` beside the catalogue |
| `--results-directory PATH` | Create or resume managed `Batch ...` folders. Cannot be combined with `-o`, `-b`, or `--force`. | Direct output without this option |
| `-f`, `--force` | Replace an existing direct catalogue while retaining timestamped prior files. | Refuse an existing output |
| `setup` | Prepare folders and a local model. Accepts `--path-to-inputs`, `--results-directory`, `-m` / `--model MODEL`, and `--hosted` for a hosted dependency check instead. | `uma-bct setup`; pinned model by default; returns JSON |

There is no `--date-detector` option. For custom managed folders:

```bash
uma-bct setup --path-to-inputs "/private/photos" --results-directory "/private/results"
uma-bct --path-to-inputs "/private/photos" --results-directory "/private/results"
```

### File extensions

Animated or multi-page inputs use the first frame or page unless noted below.

| Extension | How it is used |
| --- | --- |
| `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp` | Input photos. Extensions are matched without regard to case. |
| `.tif`, `.tiff`, `.gif`, `.avif` | Input images; animated or multi-page files use the first frame or page. |
| `.heic`, `.heif` | Input photos; use the container's primary image. |
| `.dng` | Develop the full RAW image, not its smaller embedded preview. |
| `.csv` | Default catalogue; no saved column widths or text wrapping. |
| `.xlsx` | Excel catalogue, selected with `--format excel` or a direct `.xlsx` output path. |
| `.png` in `Box Images/` | Review crops linked from the catalogue. |
| `.json` | Run details, unfinished progress, and benchmark evidence. Keep batch metadata intact. |
| `.zip` | The client delivery bundle; unzip it before running. |

## Run the tool

1. Download and unzip `archive-box-transcription.zip` into a private folder that
   is not synced to iCloud, OneDrive, Dropbox, or a network drive.
2. On Windows, double-click `Run UMA BCT.cmd`. On macOS or Linux, open Terminal
   in the unzipped folder and run `sh run-uma-bct.sh`.
3. Every interactive launch opens `Add Photos Here`, prints the full photo and
   results paths, and waits for Enter. Add photos directly to that folder, then
   return to the terminal and press Enter. If the folder does not open, use the
   printed path. Copying a photo here is local, not an internet upload.

Run the shell script with `sh`; do not source it with `.` or `source`. To select
Excel, including when you return to an existing unzipped folder:

```bash
sh run-uma-bct.sh --format excel
```

On Windows PowerShell: `& '.\Run UMA BCT.cmd' --format excel`. Existing completed CSV
batches stay intact and are reused. To make a **new Excel batch** from the same
photos, run `sh run-uma-bct.sh --format excel --reprocess` (or add the same two
options to the Windows launcher). Selecting Excel does not convert old CSV files.
Unfinished batches retain their saved format when the option is omitted; an explicit
different format is rejected.

The first run needs internet to install a verified private runtime and, in local mode,
the large model. Downloads finish before you are asked for photos. Later local runs
can work offline; hosted runs need internet. JSON and unattended runs never prompt.
`--help` and `--version` exit without preparing the runtime.

Supported computers: Windows 10 22H2+ on x86-64; macOS 14+ on Intel or Apple silicon;
Linux with glibc 2.28+ on x86-64 or arm64. Windows ARM, musl Linux, and older systems
are unsupported. Allow at least 16 GB of memory and enough free disk space on a
trusted, single-user workstation. CPU inference works but is slow.

Keep the terminal open until the final report. Live progress shows the current photo
and box, completed photo count, and elapsed time; captured stderr shows short
milestones. The final report lists photo outcomes, catalogue rows, elapsed time, and
results paths. Older offline runtimes without the live-display dependency still
process photos using milestone lines. Fresh bundles install live-display support
before photos are added.

During setup and processing, the CLI requests protection against automatic idle sleep
and releases it on exit. macOS confirms the request before work starts. Keep the
laptop lid open: manual sleep or low battery can still interrupt work, and lock or
screen-saver policies may turn off the display. Power-control errors do not change
transcription status. Warnings on stderr name the operation and available native
error; Windows states when no extended error is available. Linux reports helper
stderr/exit codes (for example, permission, system-bus, or `systemd-inhibit` failures)
and warns if the helper exits during work. Readiness waits and cleanup are bounded,
with no admin prompt or persistent diagnostic file. Adjust power settings before a
long batch if a warning appears.

## Photos and results

Each managed run creates or resumes a readable batch:

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

An unfinished batch has `Run Progress.json` and no catalogue until every photo and
requested verification has a terminal outcome. Restart the same launcher to resume:
committed complete, partial, or failed photos are reused; the interrupted photo is
processed again. Photos added during a run wait for the next launch. Later launches
process only new or changed photos. Each batch keeps its own catalogue; older
catalogues are neither combined nor overwritten.

Do not rename, remove, or edit source photos while a batch is unfinished. The CLI
preserves progress and names the inputs to restore if they change. Resume also
requires the original processing options, including `--ai-dates`. A batch started with
the former 80% review threshold needs `--min-transcription-confidence 80`; the current
default is 50%. Keep run details and crop files intact. You may edit a completed
catalogue and save corrected copies or notes beside it; later launches preserve them
and report a changed catalogue.

A partial batch publishes its selected format if it has usable rows; a zero-row batch
publishes only failed `Run Details.json`. Completed photos do not repeat model work
unless you use `--reprocess`. Reports covering several completed batches list all
relevant results and include earlier failures. Publication row counts describe the
original files, which may include photos since removed from the input folder. A
separate count covers boxes in current photos; spreadsheet edits are not recounted.

Legacy bundles with data only in `inputs/` and `outputs/` keep using those folders.
New runs create `outputs/Batch ...`; existing top-level catalogues, manifests, and
crops stay untouched. If both legacy and friendly directory pairs contain data, the
launcher stops and names both. It never merges, moves, or deletes them.

Direct runs default to `inputs/`, `outputs/catalogue.csv`, and `outputs/crops/`. For
Excel with verification: `uma-bct --format excel --verification always`. Direct
interruption exits cleanly (130 for Ctrl+C); use `--results-directory` for resumable
work. Direct runs have the same final report as launcher runs.

## Catalogue and human review

Both CSV and Excel keep this column order:

```text
Unit Type, Unit Number, Description, Dates, Earliest Date, Latest Date,
Transcription Confidence (%), Review Flags, Cropped Box Image Link
```

### Unit Type and Unit Number

`Unit Type` contains the model's `typed` transcription: all printed label text in
reading order except the unit-number phrase. Names, headings, logos, `Creator:`
labels, and identifiers remain. Line breaks and repeated whitespace become single
spaces. `Unit Number` is one unambiguous number padded to four digits. Recognized
labels are `Unit`, `Box`, `Box No.`, `Box Number`, and `No.`; the total in `Box 7 of
9` is excluded. Missing, invalid, or conflicting numbers stay blank and flagged,
including conflicts with handwritten identity evidence.

| Printed label text | Unit Type | Unit Number |
| --- | --- | --- |
| `HOME, Roderick Weir` / `2008.0036` / `Unit 7` | `HOME, Roderick Weir 2008.0036` | `0007` |
| `BURNET, Sir Frank Macfarlane` / `1986.0107 (1989.0034)` / `Unit 22` | `BURNET, Sir Frank Macfarlane 1986.0107 (1989.0034)` | `0022` |
| `Printing & Allied Trades` / `Employers' Association` / `of Victoria` / `1975.0064 Unit 8` | `Printing & Allied Trades Employers' Association of Victoria 1975.0064` | `0008` |
| `Victorian Chamber of Mines` / `107/56` / `Box 7 of 9` | `Victorian Chamber of Mines 107/56` | `0007` |

Only the unit phrase is removed from a shared line: `1975.0064 Unit 8` leaves
`1975.0064`. A label containing only `Unit 8` has a blank, flagged `Unit Type`.
Recognized identifiers include `YYYY.NNNN`, `YY/N`, and `UMA-(AC|SR)E?-\d{8,9}`; other
printed identifiers remain too. Malformed dotted text such as `1960.009` remains
visible with `Ambiguous identifier` and `Transcription mismatch`. Multiple valid
identifiers remain visible and receive `Multiple identifiers`; printed origin headings
are not extra identifiers.

`Description` joins handwriting and explicit content or condition annotations with
semicolons, excluding identity-only text. `Dates` comes from handwriting; `Earliest
Date` and `Latest Date` give its bounds. Local regex is the default; `--ai-dates` uses
the selected local or hosted model. When no dates are found, `Dates` is `No Dates`,
with `None` for the bounds. Failed extraction keeps the transcription, adds
`Date extraction failed`, and makes the run partial. Check dates against the crop.

Rows sort case-insensitively by complete `Unit Type`, then numeric `Unit Number`;
incomplete rows stay stable at the end. Separate physical boxes retain separate rows,
even with matching identities; those matches receive `Potential Duplicate`.

### Missing-box detection

Missing-box suggestions are enabled by default. Only internal gaps of at most 100
missing units between valid numbers in the same complete `Unit Type` group (ignoring
case) receive rows. Different headings, identifiers, or other label text create
different groups; nothing is inferred before the lowest or after the highest observed
number. Suggestions are not proof that physical boxes are missing.

For example, `HOME, Roderick Weir 2008.0036 / 0001` and the same type with `0003`
produce a `0002` row. It retains the type, has `Missing Box`, and leaves description,
confidence, and crop link blank. Use `--no-missing-box-detection` to disable this.
Readings `0001`, `0003`, `0005` suggest `0002`, `0004`; incorrect readings `0002`,
`0003` contain no gap and cannot recover the true sequence. Correct localization and
visible unit numbers first; disable suggestions when inspecting raw rows.

### Confidence, crops, and verification

`Low confidence` means strictly below 50% by default; exactly 50% is not low. `-c`
changes this review threshold without removing rows. Unavailable token confidence
exports as `0` and requires review, not a claim of measured failure. Available scores
use the least probable content token; they are not calibrated OCR accuracy or directly
comparable between model families. `Transcription mismatch` also covers
`needs_review`, including unavailable confidence, so it may appear alongside `Low
confidence` on correctly transcribed synthetic text. Review flags alone do not make
processing partial; unresolved work does.

CSV crop references are readable relative paths, such as `Box Images/Shelf - Box
01.png`, without hyperlink formulas. Excel crop filenames are clickable in desktop
Excel and LibreOffice and open with the system's associated image app. Keep the
catalogue and crop folder together when moving results. If links fail, open the named
file in `Box Images`; Apple Numbers and Google Sheets cannot open these private local
links. Existing completed catalogues stay unchanged; use `--reprocess` for a new batch.

Verification is off by default. `--verification always` adds a blind second reading of
the same images with the same selected model, local or hosted. `--verification ask`
asks after the first pass: **y** verifies this run; Enter finishes. The checking
prompt does not see the first answer. Only substantive differences receive a third
request; detection is not repeated.

A correction requires that third reading to corroborate an earlier candidate. Printed
text, identifiers, and unit number are reconciled together, including handwritten
identity claims. Unresolved differences, whole or partial content deletions, and
ambiguous box matches keep the first reading with review flags. Agreement never raises
confidence or proves accuracy. Run details preserve all readings, and managed runs
checkpoint them for resume. A failed model request keeps the first reading and marks
partial processing; a checkpoint failure stops the run. Only `--verification` enables
it for new runs; `UMA_BCT_VERIFICATION` and `.uma-bct-settings.json` do not.
Unfinished batches retain their recorded choice; older unfinished batches without that
choice continue without verification. Finish an unfinished batch before changing its
choice. Completed batches are reused unless `--reprocess` is set. JSON, redirected
stdin/stdout/stderr, and unattended runs never prompt; use `always` to request verification
there.

Before import, check every source photo for missed boxes or false detections, then
every row against its crop. Check flags and confidence, preserve separate physical
boxes, and review unflagged rows too. Accuracy gains require reviewed references, not
agreement between model answers.

Current limitations: transcription and reading order depend on the selected model.
`Carton`, `Container`, and `Package` are not unit labels; `Unit 8A`, ranges, unclear,
and crossed-out unit numbers are not ordinary accepted numbers. Only malformed dotted
standard-looking identifiers are explicitly detected; malformed UMA or slash values
may pass as arbitrary identifiers. The parser cannot distinguish incidental `Research
Unit 8` from a unit claim, so it removes `Unit 8`. Printed logos, headings, and
`Creator:` labels remain under the all-printed-text rule.

## Compatibility and privacy

Never commit, package, upload, put in Docker contexts, or use on hosted CI: private
photos (`Archive box photos/`, `add_repository_photos/`, `Add Photos Here/`,
`inputs/`), crops, catalogues, manifests, review pages, private expectations, model
weights, Ollama state, or `.uma-bct-runtime/`. Photos may carry GPS EXIF. Keep them
local and ignored; uploading private photos or derived content requires written client
approval. Keep credentials out of logs, manifests, command arguments, shell-history
lines, and committed files. Do not enable LiteLLM debug dumps for private photos or
keys.

Only numeric loopback Ollama endpoints are accepted by default; direct CLI use
defaults to `http://127.0.0.1:11434`. Remote Ollama needs written approval, HTTPS, and
`UMA_BCT_ALLOW_REMOTE_OLLAMA=1` because image pixels leave the workstation. Compose
opts into only its exact internal `ollama` hostname. Launchers and local Compose set
`OLLAMA_NO_CLOUD=1`; cloud-tagged models are unsupported. Disable cloud features on
separately managed local Ollama too, so loopback cannot proxy private images off the
workstation. Native Ollama is preferred; Docker is optional.

### Hosted model opt-in

Set `UMA_BCT_API_KEY` and provider-qualified `UMA_BCT_API_MODEL` in the process
environment. Direct CLI use also requires `--hosted`; without it, local Qwen runs even
when those variables are set. Only direct OpenAI (`openai/`), Claude (`anthropic/`),
and Gemini (`gemini/`) API origins are accepted. Sanitized detection and crop PNG
pixels leave over HTTPS; original files and EXIF do not. Obtain the required approval
and check provider handling and charges before using private photos.

A source checkout can install hosted support with `uv sync --locked --extra hosted`
(or install `uma-bct-cli[hosted]` directly), then run `uv run --locked uma-bct
--hosted --json --force`. ZIP launchers already include the hosted client and add
`--hosted` when a nonempty key is set; an empty key keeps local Qwen even with a
hosted model name. For a ZIP in Bash:

```bash
export UMA_BCT_API_MODEL=anthropic/your-vision-model
printf "API key: "
read -r -s UMA_BCT_API_KEY
printf "\n"
export UMA_BCT_API_KEY
sh run-uma-bct.sh
```

In PowerShell, set `$env:UMA_BCT_API_MODEL` and `$env:UMA_BCT_API_KEY` in the same
terminal before `./run-uma-bct.ps1`; the Gemini example below reads the key safely.
Direct `--hosted` requires both a nonempty key and qualified model; `--model` is local
only. Explicit `--hosted` without a key fails before local bootstrap. Hosted setup
checks the installed client without paid calls or Ollama. After
preflight, stderr reports `Selected model: <model> (<backend>)`; the first image
request determines actual key/model access. The opt-in Compose command is `docker
compose --profile hosted run --rm hosted`, with no Ollama dependency.

Errors report exception type, HTTP status, and fixed hints without provider reply text
or credentials. HTTP 500/502/503/504 gets at most two retries with exponential backoff
and jitter; authentication, model-selection, and quota errors are not retried. Gemini
uses `/v1beta`, while manifests retain its HTTPS origin. A Gemini HTTP 400 explicitly
rejecting optional log probabilities gets one retry without them. Missing
probabilities export confidence `0` with flags, without preventing CSV output.
LiteLLM's feedback banner is suppressed to keep JSON stdout clean. Managed resume
requires the same backend, provider, model, and HTTPS origin; changing models after
completion creates a separate batch.

Qwen, OpenAI, and Anthropic use `[xmin, ymin, xmax, ymax]`; Gemini uses `[ymin, xmin,
ymax, xmax]`, converted to X-first before cropping. Coordinates are normalized to
0–1000; see Google's [bounding-box
guide](https://ai.google.dev/gemini-api/docs/image-understanding#bounding-box-detection).
Hosted manifests still name the shared stages `qwen`. Detector confidence `1.0` means
accepted geometry, not measured detection probability; the detector threshold filters
OpenCV only. Hosted prompts request one rectangle per substantially visible physical
box front, excluding shelves, side panels, labels, and narrow textless fragments;
local Qwen retains its original prompts. A non-box response `{"boxes": []}` gives a
warning without a row; a real unreadable box retains its row. These rules do not
guarantee localization or impose a fixed count. Review thin fragments and the latest
run's crops and manifest, not a previous run's crop directory.

### Test Gemini without paying

**Historical check, 2026-10-07:** a Windows Google AI Studio project without billing
used `gemini/gemini-3.5-flash-lite` for a text reply and one invented image through
detection, transcription, and CSV export. It returned Unit Type `1960.0009`, Unit
Number `0004`, and unavailable confidence `0` with flags. This predates the Gemini
coordinate-order fix: rerun with a non-square, multi-box image to assess localization.
It is not handwriting accuracy evidence or a guarantee of current availability,
prices, or quotas. Later 3.8 Flash generation probes timed out in PowerShell and
Python despite successful model lookup and token counting; the same key and network
generated successfully with 3.5. A free-tier listing is not a passed test.

The saved notes cite Google's [pricing
page](https://ai.google.dev/gemini-api/docs/pricing#gemini-3.5-flash-lite) listing free
input/output tokens for 3.5 Flash Lite, an image-input/text-output model, and a free
tier for `gemini-3.8-flash`. Verify the project's **Billing Tier** and [active rate
limits](https://ai.google.dev/gemini-api/docs/rate-limits) in AI Studio before
testing; keep it on Free Tier to avoid paid use. See [model
capabilities](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite/) and
[billing](https://ai.google.dev/gemini-api/docs/billing). This reads images; it is not
the separate image-generation model. Free Tier content may be used to improve Google's
products: **use invented images only, never client or confidential photos**. Get a key
in [AI Studio](https://aistudio.google.com/apikey), confirm its project tier, and
follow the [key guide](https://ai.google.dev/gemini-api/docs/api-key).

The following requires a **source checkout**, not only the client ZIP. From its root,
Bash creates an invented box in a private temporary directory and runs the full
workflow. The hidden key prompt avoids shell history:

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

The JSON summary names the actual model and manifest. In
`$test_root/catalogue.run.json`, check `model.backend` is `hosted` and `model.model`
is `gemini/gemini-3.5-flash-lite`; review the CSV and crop if a row exists. Synthetic
drawings can be missed or misread, so inspect `status`, `failures`, and `images`
instead of using catalogue accuracy as the connection check. For `Hosted model request
failed`, check key, model, and quota in AI Studio. Each photo normally needs one
detection request plus one per crop, with extra fallback/retry requests. Finish with
`unset UMA_BCT_API_KEY UMA_BCT_API_MODEL` in Bash.

On Windows, make an invented PNG in Paint in a new temporary folder. From a source
checkout, use PowerShell and replace both paths below:

```powershell
uv sync --locked --extra hosted
$env:UMA_BCT_API_MODEL = 'gemini/gemini-3.5-flash-lite'
$secret = Read-Host 'Gemini API key' -AsSecureString
$env:UMA_BCT_API_KEY = [System.Net.NetworkCredential]::new('', $secret).Password
uv run --locked uma-bct --hosted --json `
  --path-to-inputs 'C:\path\to\invented-box.png' `
  --path-to-outputs 'C:\path\to\gemini-test.csv'
```

The CSV and manifest must not exist unless you use `--force`. CLI options are the same
on every platform. If access fails, keep the key in that terminal and isolate it from
image processing with these direct Google requests, which bypass LiteLLM and Ollama
and send only invented text:

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

The list confirms names, not free quota. Check active quota in AI Studio: 404 can mean
wrong endpoint/unavailable model; 401/403 means key/access; 429 means minute, token,
or daily limits; 503 means wait or try another available free-tier vision model. Rapid
retries cannot fix a project with no free quota. After a successful direct request,
rerun the full CLI on one invented image and inspect JSON and CSV. Successful calls
can still give bad crops or unit readings. Shared `qwen` stages use Gemini for both
detection and transcription when hosted, without Ollama. Clean up the PowerShell key
and related variables when finished:

```powershell
Remove-Item Env:UMA_BCT_API_KEY
Remove-Item Env:UMA_BCT_API_MODEL
Remove-Variable headers, secret -ErrorAction SilentlyContinue
```

## Development

This section and the build/benchmark commands require a **source checkout** with
`pyproject.toml`, tests, and scripts. The client ZIP includes this README for
reference, plus its wheel and launchers; clients need no manual Python packages.

Install `uv`, open a new terminal so it is on `PATH`, then check `uv --version`:

| Platform | Install command |
| --- | --- |
| macOS with Homebrew | `brew install uv` |
| macOS/Linux without Homebrew | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Windows PowerShell with WinGet | `winget install --id=astral-sh.uv -e` |

If it is still not found, close and reopen the terminal. `uv` installs/selects Python
3.12 below; a separate Python install is normally unnecessary. From the repository
root, run the locked, model-free checks:

```bash
uv sync --locked --dev --python 3.12
uv run ruff check .
uv run coverage run -m unittest discover -s tests -v
uv run coverage report
```

Run source commands with `uv run uma-bct --help`, `uv run uma-bct setup`, and `uv run
uma-bct`. For hosted development, use `uv sync --locked --dev --extra hosted` plus the
variables and explicit `--hosted` flag above. Tests use fake SDKs and synthetic
images: no paid requests or model installation. Use only invented photos for the Free
Tier walkthrough.

CI runs the full suite and all three CLI entry paths on Windows x86-64, macOS
Intel/arm64, and Linux x86-64/arm64, with explicit GitHub-hosted runner images. Docker
smoke tests run natively on both Linux architectures. Installed-package smokes use the
launchers' hash-pinned dependencies, including hosted support;
`scripts/smoke_hosted_install.py` imports the real SDK and tests all three adapters
using synthetic replies without provider requests.

For API docs with live reload from docstrings, run `uv run pdoc uma_bct`. For a static site, run
`uv run pdoc uma_bct -o dist/api-docs` and open `dist/api-docs/uma_bct.html`. This
output is generated and ignored.

### Client runtime

Launchers verify the wheel and hash-pinned `client-constraints.txt`, install private
Python 3.12 in `.uma-bct-runtime/`, and open the photo/results folders. Local mode
downloads verified pinned Ollama and load-probes pinned Qwen; hosted mode checks the
client and skips the Ollama download, service, and model pull. Constraints come from
`uv.lock`, including hosted dependencies; never edit them by hand.

Each local launch owns a new Ollama service on an OS-selected numeric loopback port.
It neither reuses nor terminates other folders' services or a foreign port-11434
service. Startup retries fresh ports; failures point to `.uma-bct-runtime/ollama.log`.
Stopping the launcher interrupts the foreground CLI first, then stops/reaps only owned
processes with bounded cleanup on all exit paths. An incomplete runtime is repaired
only when the selected photo/result folders are empty. Windows uses the
checksum-pinned portable Ollama archive, verifies both executables' Authenticode
signatures, and never runs the desktop installer or alters foreign services.

`--model MODEL` selects the local model for first-run setup as well as processing;
direct setup also accepts `uma-bct setup --model MODEL`. Choose an available model
before adding photos. Changing `--model` does not bypass the rule that runtime/model
repair requires empty photo/results folders.

### Processing, checkpoints, and recovery

```text
validate inputs/outputs -> model preflight -> shared vision detection
  -> experimental OpenCV fallback -> crops and bounded full-image recovery
  -> transcription -> optional verification -> conservative catalogue mapping
  -> per-photo atomic checkpoint -> recoverable catalogue and run details
```

Keep successful boxes when another crop or photo fails; use one full-image recovery
when localization or all crop transcriptions fail. Managed metadata keeps source
SHA-256 fingerprints, backend/model identity, endpoint history, detector routes, final
readings, warnings, and failures, but no keys, per-attempt timing, or proposal
telemetry. History uses source directory, filename, and fingerprint; different
filenames are not deduplicated by identical pixels. Snapshots exclude later additions.
Application-owned manifests, crops, and unfinished publications stay validated while
completed catalogue edits and reviewer files are preserved.

Managed processing holds an OS-backed `.uma-bct.lock` in the results root. Exclusive
acquisition recovers a well-formed dead-process lock; malformed, linked, non-regular,
or racing locks remain blocked. Direct mode keeps its catalogue-lock behavior. Paths
assume a trusted, single-user workspace: keep directories in place during
work/recovery. Existing symlinks and junctions are rejected before recovery mutations;
this does not isolate operations from an untrusted process replacing parent
directories concurrently.

After all photos and requested checks are terminal, sources and crop hashes are
revalidated, the hidden CSV/Excel is staged, and its digest is checkpointed before
publication without overwriting existing files. `Run Progress.json` becomes `Run
Details.json`. In forced direct runs, a manifest failure after CSV replacement
restores the prior CSV atomically or removes the new one, then attempts a uniquely
named failed-run manifest. Failed rollback gives partial status and a recovery path.

`verification.py` owns mode choice, bounded readings, and reconciliation;
`BoxResult.verification` stores optional evidence. First-pass commits are separate
from verification checkpoints. Application version/model checks protect resumed
inference. Publication-only recovery and cached second/third readings do not load a
model; identity is checked lazily before a missing reading. Old records without
verification remain readable. Legacy HTML metadata adds no publication/integrity
requirements; new runs generate no HTML and preserve completed manifests.

Completed summaries use the newest applicable outcome per current input.
`catalogue_outputs` and `run_manifests` list all contributing results; singular paths
remain populated for one batch. `catalogue_records` and `catalogue_record_counts`
retain publication row counts. `current_input_records` counts physical-box rows for
current photos only, excluding missing-box suggestions and later CSV edits.

### Optional Docker workflow

From a source checkout, the optional local container workflow is:

```bash
mkdir -p inputs outputs
docker compose config --quiet
UMA_BCT_UID="$(id -u)" UMA_BCT_GID="$(id -g)" \
  docker compose run --rm app
```

Docker defaults to CSV. Inputs mount read-only, outputs use the host user, and Ollama
has no exposed host port. The `hosted` profile has no Ollama dependency; use `docker
compose --profile hosted run --rm hosted` only with the environment key/model and
hosted approval described above. Free-tier checks use synthetic photos.

## Build the client deliverables

From the source checkout root, after installing `uv`, run this full sequence:

```bash
uv sync --locked --dev --python 3.12
uv export --quiet --locked --no-dev --extra hosted --group client-bootstrap \
  --no-emit-project --format requirements-txt \
  --output-file client-constraints.txt
uv build --no-build-isolation
uv run python scripts/build_client_bundle.py dist
uv run python scripts/check_release_artifacts.py dist
```

The final check must report that the release artifact privacy check passed. Send only
`dist/archive-box-transcription.zip` through approved private sharing; the client
downloads, unzips, and runs it. Do not send the standalone wheel, source archive,
repository, private photos, or catalogue outputs. `build/` and `dist/` are
generated/ignored. Delivery scripts build the ZIP, check artifact privacy, and safely
unpack Linux Ollama. Release and container-image publishing is deferred.

Before delivery, install wheel and sdist outside the checkout and smoke-test `uma-bct
--help`, `uma-bct --version`, and `python -m uma_bct --help`. Run the full suite and
platform matrix. Launcher changes need deterministic ownership fixtures in
`tests/test_client_delivery.py`: foreign port-11434 services, dynamic ports, missing
companions, bind retries, startup failure, exit codes, and owned cleanup, without a
model. Batch changes need interruption/resume, source/crop integrity, publication
recovery, formula safety, naming collisions, and zero-row tests.

Check synthetic spreadsheet links on macOS, Windows, and Linux with Unicode, spaces,
`#`, and `%`, also after moving the batch. Check CSV imports in Excel and LibreOffice
separately; Numbers users open named crops directly. Never upload private photos for
browser tests or run inference on hosted CI. The real-model gate stays local and is
handled separately by the test engineer.

## Verification and local accuracy references

From a source checkout, prepare a local reference draft from a run manifest:

```bash
uv run python scripts/verification_benchmark.py prepare \
  --manifest "/private/results/Run Details.json" \
  --output "add_repository_photos/verification-release-gate/reference-draft.json"
```

Review every source photo, correct fields, add missed boxes, remove false detections,
and set each photo's `review_status` to `reviewed`. Unreviewed drafts cannot be
scored. Compare a verification run with approved references:

```bash
uv run python scripts/verification_benchmark.py compare \
  --reference "add_repository_photos/verification-release-gate/reference-draft.json" \
  --manifest "/private/results/Run Details.json" \
  --output "add_repository_photos/verification-release-gate/comparison.json"
```

The comparison reports physical-box counts, missing/unexpected boxes, exact field
matches before/after verification, and newly incorrect fields. It uses stable box IDs:
confirm they still refer to the same physical boxes after detection changes. Record
elapsed time from the CLI report plus model/options beside the comparison. Model
agreement and draft labels are not accuracy evidence.

The local release set is all 21 images in `add_repository_photos/` plus these 11 JPEGs
in `Archive box photos/` (numeric suffixes of `PXL_*.jpg`): `230150462`, `230251945`,
`230445954`, `230518743`, `230624828`, `230737955`, `230824065`, `230850841`,
`231116923`, `231212408`, `231905575`. These directories are optional for
contributors; report the gate as not run if photos/model are unavailable. Keep photos,
drafts, human corrections, review notes, and benchmark output in ignored private
directories. Report outstanding checks explicitly.
