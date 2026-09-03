# University of Melbourne Archives Box Contents Transcription CLI

UMA BCT is an offline-first command-line application for turning photographs of
archive shelves into a reviewable CSV or XLSX catalogue. It detects each physical
box, saves full-resolution review crops, and asks a local Qwen3-VL model through
Ollama to transcribe printed labels and handwriting.

The hardened v0.1 workflow favours accuracy and traceability over speed:

```text
Qwen box detection
  -> OpenCV fallback when detection is unusable
  -> full-image transcription when no valid crops remain
  -> canonical catalogue rows + review flags + run manifest
```

By default, all image and model processing stays on the machine running the CLI.
The default test suite is model-free and uses fakes; private archive photographs
and model weights are never sent to GitHub Actions or included in release
artefacts.

## Requirements

- Python 3.12
- [`uv`](https://docs.astral.sh/uv/) (preferred) or `pipx`
- [Ollama](https://ollama.com/download), running locally
- At least 16 GB RAM for the default Qwen3-VL 8B model

macOS and Linux are supported on x86-64 and arm64; Windows is supported on
x86-64. Native Ollama is the recommended runtime because it can use the host's
available acceleration, including Metal on supported Macs. CPU inference works
but is substantially slower. Docker is provided as a reproducible Linux-container
alternative and should be treated as CPU-first unless the host's Docker/Ollama
GPU integration has been configured explicitly.

The first setup requires internet access to download Python dependencies and the
model. After that, normal transcription can run offline.

## Install

### From a private GitHub Release

Download the wheel from the repository's release page, then install it with one
of the following commands:

```bash
uv tool install ./uma_bct_cli-0.1.0-py3-none-any.whl
```

```bash
pipx install ./uma_bct_cli-0.1.0-py3-none-any.whl
```

Install Ollama separately, start it, and download the pinned model once:

```bash
ollama serve
ollama pull qwen3-vl:8b-instruct-q4_K_M
```

Ollama may already be running as a desktop service, in which case a second
`ollama serve` process is unnecessary. Model weights are approximately 6.1 GB
and are intentionally excluded from the Python package and container image.

### From a checkout

```bash
uv sync --locked --dev
uv run uma-bct --help
```

`uv.lock` is authoritative. Do not use an unlocked install for release or CI.

## Run natively

With photographs in `./inputs`, the default command writes `./outputs.csv`, a
timestamped crop run below `./outputs/crops`, and `./outputs.run.json`:

```bash
uma-bct
```

An explicit CSV run looks like this:

```bash
uma-bct \
  --path-to-inputs /path/to/archive-photos \
  --path-to-outputs ./outputs/catalogue.csv \
  --path-to-box-images ./outputs/crops
```

Use `.xlsx` for an Excel workbook. A single image can be supplied in place of a
directory. Input directories are scanned non-recursively for supported images.

The main options are:

```text
-i, --path-to-inputs PATH          image file or flat image directory (./inputs)
-o, --path-to-outputs PATH         .csv or .xlsx catalogue (./outputs.csv)
-b, --path-to-box-images PATH      review-crop root (./outputs/crops)
-d, --detector {qwen,opencv}       requested detector (qwen)
-t, --transcriber {qwen}           text transcriber (qwen)
-m, --model MODEL                  Ollama model tag
-c, --min-confidence 0..100        low-confidence review threshold (80)
    --min-detector-confidence 0..1 OpenCV-only detector threshold (0.60)
    --request-timeout SECONDS      per-model-request timeout (600)
-f, --force                        replace an existing catalogue
    --version                      print the installed version
```

The CLI checks the input, destination, Ollama service, and model before starting
the batch. It never overwrites an existing catalogue unless `--force` is passed.
Progress and warnings go to stderr; stdout contains one concise JSON summary with
status, counts, and output paths suitable for automation. Detailed source paths
and transcriptions stay in the local run manifest. The timeout default can also
be set with
`UMA_BCT_OLLAMA_TIMEOUT_SECONDS`.

Exit codes are stable:

- `0`: every image completed successfully.
- `1`: processing was partial or failed. Successful rows may still be exported,
  but an existing catalogue is never replaced by a zero-row run.
- `2`: command usage or preflight configuration is invalid.

Set `OLLAMA_HOST` when Ollama is not at its default local address. For example,
when networking from WSL to an Ollama service on Windows has been configured:

```bash
OLLAMA_HOST=http://host.docker.internal:11434 uma-bct -i /mnt/c/Archive/Photos
```

The CLI permits loopback and known local-container hostnames by default and
rejects other Ollama endpoints. If an organisation has explicitly approved a
managed remote Ollama server, opt in with `UMA_BCT_ALLOW_REMOTE_OLLAMA=1`. This
sends the archive image pixels off-device even though the CLI first rewrites
inputs as metadata-free PNGs; review the organisation's privacy controls before
using it. Ollama HTTP requests do not inherit proxy settings or follow redirects.

## Catalogue and review artefacts

CSV and XLSX exports use this stable column order:

| Column | v0.1 behaviour |
| --- | --- |
| `Node ID` | Blank; database reconciliation is deferred. |
| `Node Type` | Always `Unit`. |
| `Node Title` | `{first record identifier} Unit {four-digit box number}`. |
| `Description` | Handwriting and explicit auxiliary annotations, separated by `; `. |
| `Date Range` | Blank; structured date extraction is deferred. |
| `Confidence Score` | Model confidence used for human-review triage. |
| `Flags` | Controlled warnings such as ambiguity, fallback, or duplication. |
| `Image` | Catalogue-relative path to the saved review crop. |

Recognised record identifiers are `YYYY.NNNN`, `YY/N`, and
`UMA-(AC|SR)E?-#########` (eight or nine digits). If several identifiers appear,
the first is used in the canonical title and the row is flagged. Separate
physical boxes remain separate rows even if their titles match.

Every run also writes `<catalogue-name>.run.json`. The manifest captures the
configuration, requested and actual detector, fallback route, model tag, raw
validated text, warnings, failures, and summary counts. Keep the catalogue,
manifest, and crop run together during review so relative image paths continue
to work.

On Windows, place the catalogue and `--path-to-box-images` on the same drive so
the `Image` column can remain a portable relative path.

## Docker and Compose

The production image contains only the application and its locked Python
environment. It runs as an unprivileged user and has `uma-bct` as its entrypoint.
It does not contain Ollama, model weights, inputs, or outputs. Python, uv, and
Ollama image inputs are pinned to reviewed multi-architecture manifest digests;
update the human-readable tag and digest together during dependency maintenance.

Build and inspect the app image:

```bash
docker build --target production -t uma-bct-cli:local .
docker run --rm uma-bct-cli:local --help
```

For the complete CPU-oriented stack, create `inputs/` and `outputs/` beside the
Compose file, place copies of the intended images in `inputs/`, then run. On
native Linux, passing the host UID/GID keeps the writable bind mount usable by
the container's non-root process:

```bash
mkdir -p inputs outputs
UMA_BCT_UID="$(id -u)" UMA_BCT_GID="$(id -g)" docker compose run --rm app
```

Compose starts an internal Ollama service, downloads the selected model into the
named `ollama-models` volume if necessary, mounts inputs read-only, and mounts
outputs writable. Ollama has no published host port. Override the model without
editing YAML:

```bash
UMA_BCT_MODEL=qwen3-vl:4b docker compose run --rm app
```

The default safe-overwrite rule also applies in Compose. To repeat a run against
an existing destination, append the complete desired arguments and `--force`.
For native GPU setup or Docker GPU passthrough, follow Ollama's current
[hardware guidance](https://docs.ollama.com/gpu) and
[Docker guide](https://docs.ollama.com/docker).

Release automation publishes private GHCR images for `linux/amd64` and
`linux/arm64`. A container image can run only against a compatible host kernel
and architecture; it is not a universal native executable.

## Privacy and data handling

Archive photographs can contain sensitive collection information and GPS-bearing
EXIF metadata. The repository intentionally ignores `Archive box photos/`,
`add_repository_photos/`, and the default `inputs/` directory. The Docker build
context and source-distribution rules exclude those directories, all generated
outputs, Ollama state, and common model-weight formats.

Do not:

- add private images or human-reviewed ground truth to Git;
- upload them as workflow artefacts or use them on hosted CI runners;
- bake model weights into wheels, containers, releases, or CI caches;
- expose the Compose Ollama service publicly.

Real-model acceptance runs are performed only on approved managed local hardware.

## Develop and verify

Detection and transcription remain separate adapters behind domain protocols.
Qwen is the default detector; OpenCV is a conservative fallback. The shared
cropper handles bounds, overlap suppression, ordering, and crop persistence but
does not impose portrait-only geometry on Qwen proposals.

Run the same model-free checks as CI:

```bash
uv sync --locked --dev
uv run ruff check .
uv run ruff format --check .
uv run coverage run -m unittest discover -s tests -v
uv run coverage report
uv build --no-build-isolation
```

The initial coverage floor is 75%. CI runs unit tests on Python 3.12 for Ubuntu,
macOS, and Windows, then installs built distributions outside the checkout and
smoke-tests the console entry point. It also inspects wheel and sdist contents to
reject private input/output directories, archive images, generated catalogues,
and model files. Container smoke tests use only fake model clients and verify
that the host-UID process can write to a bind-mounted output directory. Pull
requests and pushes to `main` never start Ollama or download Qwen.

Before a v0.1 release, run the approved real-photo suite locally and record the
machine, elapsed time per image and per box, detector route, and review outcome.
Performance is documented, not used as a release gate. A supported visible box
must produce one row or an explicit nonzero failure; ambiguous text must be
flagged rather than invented.

### Provisional local benchmark

On 2026-09-03, a native-Ollama spot run used the pinned
`qwen3-vl:8b-instruct-q4_K_M` model on an Apple M5 MacBook Air (10 CPU cores,
24 GB memory, macOS 26.4). One photograph containing five boxes completed in
107.245 seconds: 107.245 seconds per image and 21.449 seconds per exported box.
Detection returned five boxes and all five canonical titles were exact. Several
description characters still required human review, so this is a throughput
observation, not a passed accuracy gate. The requested 16 GB reference-laptop
benchmark and the complete 32-image reviewed acceptance run must still be
recorded before tagging v0.1.0.

## Release

Push an annotated `v*` tag only after required CI checks pass. The release
workflow first verifies that the tagged commit is contained in `main`, rebuilds
and tests the exact wheel and sdist, reinspects both archives for private data,
writes SHA-256 checksums, attaches all three files to a private GitHub Release,
and publishes a matching multi-architecture image to the repository's private
GHCR namespace. The tagged production image is smoke-tested before any image is
pushed on both target architectures, and those exact tested images form the
published manifest. The tagged source's model-free test container must also
pass.

Repository administrators must mark the CI jobs as required branch-protection
checks, create a protected `release` environment for both publishing jobs, and
confirm that the repository and package visibility remain private. Restrict that
environment to approved `v*` tags and require the project's release reviewers.
Create an immutable protected-tag ruleset for `v*` that prevents tag updates and
deletions. Automation also refuses to replace an existing GitHub Release or any
existing release/architecture image tag.
