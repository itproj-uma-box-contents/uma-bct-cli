# CLI guide

[README](../README.md) · [Catalogue and review](catalogue.md) · [Hosted models](hosted-models.md)

See the [README quick start](../README.md#run-the-tool) for the client ZIP or
[source setup](../CONTRIBUTING.md#set-up-a-source-checkout) for development.
This guide describes the current CLI. The Confluence
[user manual](https://uma-box-contents.atlassian.net/wiki/spaces/UBC/pages/1016046)
covers the CLI and separate review GUI. The
[CLI manual](https://uma-box-contents.atlassian.net/wiki/spaces/UBC/pages/7077934)
also documents the launcher workflow and direct CLI options.

## Running a batch

The launcher prints the input/results paths and opens `Add Photos Here`; interactive
runs wait for Enter. Add photos directly to the folder, not subfolders. Run the shell
script with `sh`, not `source`. Use `--format excel --reprocess` to create a new
Excel batch from previously processed photos. Selecting Excel does not convert old CSVs.
Unfinished batches retain their saved format when the option is omitted; an explicit
different format is rejected.

JSON and unattended runs never prompt. `--help` and `--version` skip runtime setup.
Progress goes to stderr; the final report gives outcomes, row counts and paths.
Keep the terminal open, laptop lid open and power connected. The CLI requests idle
sleep prevention, but manual sleep or low battery can still interrupt processing.

## Photos and results

Each managed run creates or resumes a readable batch:

```text
Unzipped client folder/
  Add Photos Here/
  Results/
    Batch 2026-09-11 14-30-05/
      Catalogue.csv
      Run Details.json
      Box Images/
        Shelf - Box 01.png
        Shelf - Box 02.png
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
the former 80% review threshold needs `--confidence-threshold 80`; the current
default is 50%. Keep run details and crop files intact. You may edit a completed
catalogue and save corrected copies or notes beside it; later launches preserve them
and report a changed catalogue.

A partial batch publishes usable rows; a zero-row batch publishes failed run details
only. New runs generate no HTML review page. Reports spanning several batches include
earlier failures. Publication counts describe original output; current-input counts
exclude missing-box suggestions. Spreadsheet edits are not recounted.

Legacy bundles with data only in `inputs/` and `outputs/` keep using those folders.
New runs create `outputs/Batch ...`; existing top-level catalogues, manifests, and
crops stay untouched. If both legacy and friendly directory pairs contain data, the
launcher stops and names both. It never merges, moves, or deletes them.

Direct runs default to `inputs/`, `outputs/catalogue.csv`, and `outputs/crops/`. For
Excel with verification: `uma-bct --format excel --verification always`. Direct
interruption exits cleanly (130 for Ctrl+C); use `--batch-outputs` for resumable
work. Direct runs have the same final report as launcher runs.

## CLI arguments

Add these options after `uma-bct`, `sh run-uma-bct.sh`, or the Windows launcher. For
example, `sh run-uma-bct.sh --format excel` creates an Excel catalogue for a new
batch. Format names ignore case: `--format Excel` works too. In a source checkout,
put `uv run` before `uma-bct` in these examples.

Detection and transcription automatically use the selected local or hosted model.
You do not need to choose a detector or transcriber. Readable terminal output is
the default; add `--json` only when another program needs the run summary.

| Option | What it does | Default / example |
| --- | --- | --- |
| `--format csv` / `--format excel` | Choose a CSV file or an Excel `.xlsx` workbook for a new batch. | CSV; `--format excel` |
| `--verification never\|always\|ask` | Skip the second reading, always compare a blind second reading, or ask after transcription. Substantive differences may need a third reading. | `never`; `ask` skips when JSON or a stream is redirected |
| `--ai-dates` | Ask the selected local or hosted model for dates in transcribed handwriting; adds text requests. Pass `--ai-dates` without a value. | Built-in date matching; no extra date request |
| `-c`, `--confidence-threshold PERCENT` | Flag scores below this percentage; keep all rows. Accepts 0–100. | `50`; `-c 80` |
| `-mb`, `--disable-missing-box-rows` | Leave out suggested rows for gaps in the box-number sequence. Photographed boxes are still detected and exported. | Suggestions enabled |
| `-m`, `--model MODEL` | Select a local Ollama model. For hosted models, use `UMA_BCT_API_MODEL`. | Pinned `qwen3-vl:8b-instruct-q4_K_M` |
| `--hosted` | Use the provider and key from `UMA_BCT_API_MODEL` and `UMA_BCT_API_KEY`. Direct CLI runs require this opt-in. | Local; ZIP launchers add it when a key is set |
| `-t`, `--timeout SECONDS` | Maximum wait for each processing request, in seconds. This is not a limit for the whole batch or first-run downloads. Retries can add time. | `UMA_BCT_OLLAMA_TIMEOUT_SECONDS`, otherwise `600` |
| `-r`, `--reprocess` | Process the same photos again into a separate batch, keeping earlier results and edits. An unfinished batch still resumes first with its original options. | Reuse saved work |
| `--json` | Write a JSON run summary to stdout; progress and warnings go to stderr. Use `--format` to choose the catalogue file type. Never prompt. | Readable report |
| `-h`, `--help` | Show help and exit. ZIP launchers do not download or start a runtime for this. | No processing |
| `--version` | Show the version and exit without processing or ZIP runtime setup. | No processing |

These path and setup options are for the **direct CLI only**. The ZIP launchers use
`Add Photos Here` and `Results` (or the existing legacy layout) and reject custom
paths, direct output options, and `setup`.

| Direct CLI option | What it does | Default / example |
| --- | --- | --- |
| `-i`, `--inputs PATH` | Read one image or a flat directory of images. | `inputs/` |
| `-o`, `--outputs FILE` | Save one catalogue to this filename, including `.csv` or `.xlsx`. This is a file, not a folder. Its extension must agree with an explicit `--format`. | `outputs/catalogue.csv`; `-o catalogue.xlsx` selects Excel |
| `-co`, `--crop-outputs FOLDER` | Save the cropped box images used for review in this folder. | `crops/` beside the catalogue |
| `--batch-outputs FOLDER` | Save each batch in its own subfolder here, with its catalogue, crops and progress for restarting. Cannot combine with `--outputs`, `--crop-outputs` or `--force`. | Direct output without this option |
| `-f`, `--force` | Replace an existing direct catalogue while retaining timestamped prior files. | Refuse an existing output |
| `setup` | Prepare folders and a local model. Accepts `--inputs`, `--batch-outputs`, `-m` / `--model MODEL`, and `--hosted` for a hosted dependency check instead. | `uma-bct setup`; pinned model by default; returns JSON |

There is no `--date-detector` option. For custom managed folders:

```bash
uma-bct setup --inputs "/private/photos" --batch-outputs "/private/results"
uma-bct --inputs "/private/photos" --batch-outputs "/private/results"
```

Use `--batch-outputs` for long or repeatable jobs. For a single catalogue with a
chosen filename and crop folder, use direct output instead:

```bash
uma-bct --inputs "Photos" --outputs "Review/catalogue.xlsx" --crop-outputs "Review/Crops"
```

Put quotes around paths containing spaces. The short forms `-co` and `-mb` are
whole options: write `-co "Review/Crops"` and `-mb` separately, rather than joining
them to another short option. `-t` now means timeout in seconds.

### Choosing processing options

Photos always use a vision model for transcription. Without `--ai-dates`, built-in
text rules extract dates from the transcribed handwriting. Adding `--ai-dates` asks
the same selected model to interpret that handwriting in a separate text request;
it does not read the crop image again. It can take longer, and hosted requests may
add charges. Both methods fill the same `Dates`, `Earliest Date`, and `Latest Date` columns.
Both methods need human review. AI extraction failures are flagged; they do not
silently switch back to built-in matching.

`--verification always` separately compares a blind second transcription. It can be
used with or without `--ai-dates`; verification readings may add date requests too.
`-c 80` flags transcription scores below 80% and keeps the rows. By contrast,
the advanced `--min-detector-confidence 0.7` filters proposals from the experimental
OpenCV fallback on a 0–1 scale (default `0.60`). It does not change the transcription
review threshold or filter accepted model detections. Most users can leave it alone.

For a new ZIP batch on macOS/Linux:

```bash
sh run-uma-bct.sh --format excel
sh run-uma-bct.sh --ai-dates
sh run-uma-bct.sh --verification always
```

These are separate examples; options can be combined. In Windows PowerShell, use
`& '.\Run UMA BCT.cmd'` followed by the same options. Flags such as `--ai-dates`,
`--hosted`, `--reprocess`, and `--json` take no value; options such as `--format excel`
and `--verification always` need a value.

Saved batches keep their original results and processing settings. To apply new
options to the same photos, add `--reprocess`, for example
`sh run-uma-bct.sh --format excel --ai-dates --reprocess`. An unfinished batch still
needs its original options, including its timeout; `--reprocess` does not bypass it.

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

TIFF orientation is applied before detection and cropping. Transparent PNG/WebP
inputs use the same temporary white background as other transparent image formats;
the source photo is preserved and the temporary copy is removed after processing.

## Privacy and Ollama endpoints

Keep photos, derived results, credentials and model/runtime state local and ignored,
as described in the [README](../README.md#reference). Hosted use and safe credential
setup are covered in the [hosted-model guide](hosted-models.md).

Only numeric loopback Ollama endpoints are accepted by default; direct CLI use
defaults to `http://127.0.0.1:11434`. Remote Ollama needs written approval, HTTPS, and
`UMA_BCT_ALLOW_REMOTE_OLLAMA=1` because image pixels leave the workstation. Compose
opts into only its exact internal `ollama` hostname. Launchers and local Compose set
`OLLAMA_NO_CLOUD=1`; cloud-tagged models are unsupported. Disable cloud features on
separately managed local Ollama too, so loopback cannot proxy private images off the
workstation. Native Ollama is preferred; Docker is optional.
