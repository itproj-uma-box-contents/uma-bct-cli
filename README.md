# UMA Box Contents Transcription CLI

Turn local photographs of University of Melbourne Archives boxes into a CSV
catalogue and review crops. The tool uses Ollama on your computer; photos do not
leave the computer in the supported default configuration.

## Client quick start from the wheel

Use these steps from an access-controlled local folder where you want to keep
`inputs/` and `outputs/`. Do not use a folder synced by iCloud, OneDrive,
Dropbox, another cloud service, or a network share because those services can
upload private archive data independently of this tool.

1. Install [Ollama](https://ollama.com/download), open it, and leave it running.
   If you installed only the command-line service, run `ollama serve` in a
   separate terminal.
2. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
3. Put the supplied wheel in your working folder and install it:

   ```bash
   uv tool install --python 3.12 ./uma_bct_cli-0.1.0-py3-none-any.whl
   ```

4. Prepare the folders and model:

   ```bash
   uma-bct setup
   ```

5. Put the archive-box photos in `inputs/`, then run:

   ```bash
   uma-bct
   ```

`uv` can [download and manage Python
3.12](https://docs.astral.sh/uv/guides/install-python/) automatically when it is
not already installed; see the [uv tool installation
guide](https://docs.astral.sh/uv/guides/tools/) for tool-environment details.
`uma-bct setup` creates missing `inputs/` and `outputs/` folders, checks the
local Ollama version, and explicitly downloads
`qwen3-vl:8b-instruct-q4_K_M` if it is absent. The model is large, so this first
download can take a while and needs an internet connection. Existing folder
contents and an already installed model are left alone.

The command prints download progress and errors to stderr. It prints one JSON
result to stdout and returns exit code `0` when setup succeeds or `2` when setup
cannot finish.

## Results and repeat runs

After a successful transcription, the working folder looks like this:

```text
inputs/
  shelf-photo.jpg
outputs/
  catalogue.csv
  catalogue.run.json
  crops/
    run-20260903T090000Z-a1b2c3d4/
      ...review crop images...
```

Keep the catalogue, run manifest, and timestamped crop folder together. The
manifest records settings, model details, detector fallback, warnings, failed
images, and the crop-to-record provenance needed for review.

The default run refuses to replace `outputs/catalogue.csv` or
`outputs/catalogue.run.json`. After reviewing or copying those files, repeat the
batch with:

```bash
uma-bct --force
```

`--force` replaces the catalogue and normal manifest safely. It creates a new
timestamped crop folder rather than replacing previous crops. If a forced run
produces no usable records, the existing catalogue is preserved and a separate
failed-run manifest is written when possible.

CSV is the default. To create an Excel workbook instead, use an explicit output
path:

```bash
uma-bct --path-to-outputs outputs/catalogue.xlsx
```

## What to review

The catalogue starts with the client import columns in the required order:

| Column | Meaning |
| --- | --- |
| `Node ID` | Blank in version 0.1. |
| `Node Type` | Always `Unit`. |
| `Node Title` | First record identifier plus a four-digit box number, such as `1974.0029 Unit 0319`. |
| `Description` | Handwriting and explicit auxiliary content or condition notes, joined with semicolons. |
| `Date Range` | Blank in version 0.1. |
| `Confidence Score` | A guide for human review, not a guarantee. |
| `Flags` | Missing, ambiguous, low-confidence, duplicate, mismatch, or fallback conditions to check. |
| `Image` | The saved review crop for that physical box. |

Recognised record identifiers include `1960.0009`, `96/9`,
`UMA-AC-000000330`, and `UMA-SRE-20160038`. Separate physical boxes remain
separate rows, including duplicate titles. If text is ambiguous, the tool leaves
the field blank or flags the record instead of inventing a confident value.

## Supported inputs and options

Supported image filename extensions are:

```text
.jpg  .jpeg  .png  .tif  .tiff  .webp
```

You can give the tool one image or another input directory. Common controls are:

```text
-i, --path-to-inputs PATH          Input image or directory (default: inputs)
-o, --path-to-outputs PATH         CSV or XLSX catalogue (default: outputs/catalogue.csv)
-b, --path-to-box-images PATH      Review-crop root (default: outputs/crops)
-c, --min-confidence 0..100        Transcription review threshold (default: 80)
-f, --force                        Replace an existing catalogue and manifest
    --version                      Show the installed version
```

Run `uma-bct --help` for advanced detector, model, detector-confidence, and
request-timeout controls. Existing aliases such as `--path-to-output` and
`--work-dir` remain supported.

Progress and warnings go to stderr. Each completed command writes one JSON
summary to stdout. A transcription exit code of `0` means every image completed,
`1` means results were partial or failed but the run was recorded, and `2` means
the command or Ollama preflight was invalid.

## Platform notes

- **macOS (Intel or Apple silicon):** use native Ollama. It supports Apple GPU
  acceleration through Metal. Ollama GPU acceleration is not available through
  Docker Desktop on macOS. See [Ollama hardware
  support](https://docs.ollama.com/gpu) and the [Ollama Docker
  FAQ](https://docs.ollama.com/faq).
- **Windows x86-64:** run the commands in PowerShell or Windows Terminal. Keep
  the catalogue and its crop folder on the same drive so image links remain
  usable. The uv installer is available from the linked installation guide.
- **Linux x86-64 or arm64:** native Ollama is preferred. CPU inference works but
  is slow; supported GPU acceleration depends on the installed hardware and
  drivers. Docker is an optional alternative for container-based teams.

At least 16 GB of memory is recommended for the default 8B quantized model.

## Common setup and run failures

- **Cannot connect to the local Ollama server:** open the Ollama application or
  run `ollama serve`, then rerun `uma-bct setup`.
- **Ollama is too old:** update Ollama to version 0.12.7 or newer, restart it,
  and rerun setup.
- **Model download fails:** check the internet connection and free disk space,
  then rerun setup. It resumes through Ollama where supported. The error also
  shows the equivalent `ollama pull` command.
- **No supported images found:** confirm the photos are directly inside
  `inputs/` and use one of the supported extensions above. Symbolic-link inputs
  are intentionally rejected.
- **Output already exists:** review or copy the current result, then use
  `uma-bct --force` for the next batch.
- **A run is slow:** CPU inference is supported but the default model can take
  several minutes per request. Native GPU acceleration is preferred where
  available.

After uv has installed the wheel and `uma-bct setup` has downloaded the model,
normal transcription can run offline. Ollama must still be running locally.

## Privacy and remote Ollama

The default endpoint policy accepts numeric loopback addresses and normalises
`localhost` to `127.0.0.1`. Archive photos, generated catalogues, human-reviewed
expectations, model weights, and Ollama state must never be committed, uploaded,
put into a Docker build context, or sent to hosted CI.

An approved managed remote Ollama service is an explicit privacy exception
because image pixels leave the device. It must use HTTPS and requires both
settings below.

macOS or Linux:

```bash
export UMA_BCT_ALLOW_REMOTE_OLLAMA=1
export OLLAMA_HOST=https://approved-model-service.example.edu:11434
uma-bct
```

Windows PowerShell:

```powershell
$env:UMA_BCT_ALLOW_REMOTE_OLLAMA = "1"
$env:OLLAMA_HOST = "https://approved-model-service.example.edu:11434"
uma-bct
```

HTTP remote endpoints, embedded endpoint credentials, and unapproved hostnames
are rejected. Do not enable this override without organisational approval.

## Current architecture and recovery flow

```text
preflight local files and Ollama
  -> Qwen box localisation
  -> OpenCV localisation fallback when needed
  -> create detector-neutral review crops
  -> Qwen transcription for each crop
  -> one full-image recovery attempt when localisation or all crops fail
  -> consolidate overlapping observations without merging physical boxes
  -> map and flag client catalogue rows
  -> atomically write the catalogue and run manifest
```

Each image has its own exception boundary, so usable images can still be
exported when another image fails. Detection and transcription attempts are
recorded before consolidation for traceability.

This flow describes the current code. The protected Confluence image and
design-level class and system-sequence diagrams were not accessible for exact
comparison, so no exact diagram alignment is claimed and no GitHub copy was
substituted.

## Known version 0.1 limitations

- Printed and handwritten transcription requires human review; low confidence
  and fallback flags are conservative aids, not accuracy guarantees.
- Automatic text rotation, date extraction, sequence-gap analysis, Node ID
  reconciliation, a GUI, catalogue APIs, and signed executables are not included.
- Qwen detection is primary. OpenCV detection is an experimental fallback and
  may miss unusual shelves or box geometry.
- CPU-only inference is supported but slow.
- The real-photo release gate must run locally against approved, human-reviewed
  expectations. It is never run on hosted CI.

## Optional Docker workflow

Native Ollama is the primary client path. The bundled Compose configuration is
for teams already using containers. It keeps inputs read-only, writes only to
`outputs/`, stores Ollama state in a named volume, and exposes no Ollama host
port.

Create the bind-mount folders, validate the configuration, and run the app:

```bash
mkdir -p inputs outputs
docker compose config --quiet
docker compose run --rm app
```

On Linux, set the container user so results belong to the current user:

```bash
UMA_BCT_UID="$(id -u)" UMA_BCT_GID="$(id -g)" docker compose run --rm app
```

To smoke-test the model-free production image separately:

```bash
docker build --target production -t uma-bct-cli:local .
docker run --rm uma-bct-cli:local --help
docker run --rm uma-bct-cli:local --version
```

The Compose model-init service downloads the configured model into the local
Docker volume. Docker Desktop on macOS uses CPU inference for this stack.

## Contributor setup and release checks

From a source checkout:

```bash
uv sync --locked --dev
uv run ruff check .
uv run ruff format --check .
uv run coverage run -m unittest discover -s tests -v
uv run coverage report
uv build
```

Before release, install both the wheel and source distribution into clean
environments outside the checkout. Smoke-test:

```bash
uma-bct --help
uma-bct --version
uma-bct setup --help
python -m uma_bct --help
```

Run the release-artifact privacy inspector on both archives. Real-model testing
stays on approved local hardware and uses human-reviewed expectations stored
beside the ignored source photos.
