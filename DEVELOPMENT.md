# UMA BCT developer guide

The supported client delivery is `uma-bct-client.zip`. It contains the wheel
and platform launchers, so the client does not need to install Python packages
manually.

## Development

Use Python 3.12 and the locked environment:

```bash
uv sync --locked --dev --python 3.12
uv run ruff check .
uv run ruff format --check .
uv run coverage run -m unittest discover -s tests -v
uv run coverage report
```

Run the source CLI with:

```bash
uv run uma-bct --help
uv run uma-bct setup
uv run uma-bct
```

The defaults are `inputs/`, `outputs/catalogue.csv`, and
`outputs/crops/`. CSV is the only v0.1 catalogue format. Existing input,
output, crop, detector, model, confidence, timeout, and force options remain
available through `--help`.

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

```bash
uv export --quiet --locked --no-dev --group client-bootstrap \
  --no-emit-project --format requirements-txt \
  --output-file client-constraints.txt
uv build --no-build-isolation
uv run python scripts/build_client_bundle.py dist
uv run python scripts/check_release_artifacts.py dist
```

`build/` and `dist/` are generated, ignored directories. The three scripts
build the client ZIP, reject private content in artifacts, and safely unpack
the Linux Ollama archive. Publishing releases and container images is deferred.

Before delivery, install the wheel and sdist outside the checkout and smoke-test
`uma-bct --help`, `uma-bct --version`, and `python -m uma_bct`. The
real-model photo gate remains local and is handled separately by the test
engineer.
