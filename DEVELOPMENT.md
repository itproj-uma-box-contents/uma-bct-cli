# UMA BCT developer guide

The supported client delivery is `uma-bct-client.zip`. It contains the wheel
and platform launchers, so the client does not need to install Python packages
manually.

## Developer prerequisites

Install `uv`, then open a new terminal so the executable is available on your
`PATH`.

On macOS with Homebrew:

```bash
brew install uv
```

On macOS or Linux without Homebrew:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On Windows with WinGet, run in PowerShell:

```powershell
winget install --id=astral-sh.uv -e
```

Confirm the installation before continuing:

```bash
uv --version
```

If this still reports `command not found`, close and reopen the terminal, then
try again. `uv` installs and selects Python 3.12 when the commands below are
run, so a separate Python installation is normally unnecessary.

## Development

From the repository root (the directory containing `pyproject.toml`), create
the locked Python 3.12 environment and run the model-free checks:

```bash
uv sync --locked --dev --python 3.12
uv run ruff check .
uv run ruff format --check .
uv run coverage run -m unittest discover -s tests -v
uv run coverage report
```

Model-free CI runs the full unit suite and all three CLI entry paths on the
supported architecture matrix: Windows x86-64, macOS arm64 and Intel, and Linux
x86-64 and arm64. Runner labels are pinned to explicit GitHub-hosted images.
Docker smoke tests run natively on both Linux architectures. Linux client hosts
must provide glibc 2.28 or newer; musl is not supported. macOS client hosts must
run macOS 14 or newer.

Run the source CLI with:

```bash
uv run uma-bct --help
uv run uma-bct setup
uv run uma-bct
```

The defaults are `inputs/`, `outputs/catalogue.csv`, and
`outputs/crops/`. CSV is the only v0.1 catalogue format. Existing input,
output, crop, detector, model, confidence, timeout, and force options remain
available through `--help`. The legacy `--min-confidence` option and its
explicit `--min-detector-confidence` alias set the OpenCV threshold from 0 to
1. The transcription review threshold uses `-c` or
`--min-transcription-confidence` and accepts an integer from 0 to 100.

## Client bundle

The launchers create `inputs/`, `outputs/`, and a private
`.uma-bct-runtime/` directory. On the first run they:

1. Verify the supplied wheel and hash-pinned dependency file.
2. Download and verify pinned uv and Ollama builds for the current platform.
3. Install a managed Python 3.12 runtime and the UMA BCT wheel.
4. force the numeric loopback Ollama endpoint and download the pinned model.
5. Open `inputs/`, process the photos, and open `outputs/`.

Downloads finish before the launcher asks the user to add private photographs.
Later runs reuse the local runtime and can run offline. The Windows Ollama
installer also requires a valid Ollama Inc. Authenticode signature.

`client-constraints.txt` is generated from `uv.lock`; do not edit it by
hand. It is required so every client launcher installs the same hash-verified
dependency versions.

## Processing and privacy

```text
local file/model preflight
  -> Qwen box detection
  -> OpenCV detection fallback
  -> review crops
  -> Qwen transcription
  -> one full-image recovery when localization or all crops fail
  -> conservative catalogue mapping
  -> atomic CSV and compact run manifest
```

The manifest keeps the model identity, detector route, final box readings,
warnings, and failures. It deliberately does not contain per-attempt timing or
proposal telemetry.

Only numeric loopback Ollama is allowed by default. The bundled Compose service
opts into its exact `ollama` hostname. An approved managed endpoint requires
`UMA_BCT_ALLOW_REMOTE_OLLAMA=1` and HTTPS because image pixels leave the
workstation.

Never commit, package, upload, or place in a Docker context:

- `Archive box photos/`, `add_repository_photos/`, or `inputs/`
- generated outputs and run manifests
- Ollama state, model weights, or private regression expectations

## Optional Docker workflow

Native Ollama is preferred. The optional local container workflow is:

```bash
mkdir -p inputs outputs
docker compose config --quiet
UMA_BCT_UID="$(id -u)" UMA_BCT_GID="$(id -g)" \
  docker compose run --rm app
```

Inputs are mounted read-only, outputs are written as the host user, and Ollama
is not exposed on a host port.

## Build the client deliverables

When you need to give the application to a client, run this complete sequence
from the repository root. The first command prepares the locked build
environment, so these commands also work from a fresh checkout after `uv` has
been installed.

```bash
uv sync --locked --dev --python 3.12
uv export --quiet --locked --no-dev --group client-bootstrap \
  --no-emit-project --format requirements-txt \
  --output-file client-constraints.txt
uv build --no-build-isolation
uv run python scripts/build_client_bundle.py dist
uv run python scripts/check_release_artifacts.py dist
```

The final check must report that the release artifact privacy check passed.
The finished file is `dist/uma-bct-client.zip`. Send only that ZIP to the
client through the approved private file-sharing method; the client can then
download it, unzip it, and run the launcher. Do not send the wheel, source
archive, repository, private photographs, or generated catalogue outputs.

`build/` and `dist/` are generated, ignored directories. The three scripts
build the client ZIP, reject private content in artifacts, and safely unpack
the Linux Ollama archive. Publishing releases and container images is deferred.

Before delivery, install the wheel and sdist outside the checkout and smoke-test
`uma-bct --help`, `uma-bct --version`, and `python -m uma_bct`. The
real-model photo gate remains local and is handled separately by the test
engineer.
