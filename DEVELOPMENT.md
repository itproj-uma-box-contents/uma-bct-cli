# UMA BCT developer and operator guide

The client-facing release is `uma-bct-client.zip`. It contains the wheel and
platform launchers, so client users do not install Python packages or run setup
commands themselves.

## Development environment

Use Python 3.12 and the locked dependencies:

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

The default paths are `inputs/`, `outputs/catalogue.csv`, and `outputs/crops/`.
CSV is the default; pass an explicit `.xlsx` output for Excel. Existing aliases
such as `--path-to-output` and `--work-dir` remain supported.

## Client launcher behavior

The launchers create `inputs/`, `outputs/`, and an ignored
`.uma-bct-runtime/` directory beside themselves. They then:

1. Find the one supplied UMA BCT wheel.
2. Load the exact, hash-pinned dependency closure exported from `uv.lock`.
3. Install pinned uv 0.12.9 into `.uma-bct-runtime/` when it is absent or stale.
4. Download and verify the pinned Ollama 0.30.8 platform artifact when Ollama
   is absent or stale.
5. Force `http://127.0.0.1:11434`, disable remote opt-ins, and start Ollama.
6. Run `uma-bct setup` to validate folders, the server version, and the pinned
   model, downloading the model only when it is absent.
7. Ask the user to add supported photos after all online preparation finishes.
8. Preserve timestamped copies of an existing catalogue and manifest.
9. Run the batch offline through its atomic replacement path and open `outputs/`.

The uv archive and each Ollama platform artifact are version-pinned and checked
against their published release SHA-256 values before extraction or execution.
The Windows Ollama installer must also have a valid Ollama Inc. Authenticode
signature. Linux `.tar.zst` extraction permits internal relative links while
rejecting escaping links, special entries, and path traversal. See the official
Ollama [macOS](https://docs.ollama.com/macos),
[Linux](https://docs.ollama.com/linux), and
[Windows](https://docs.ollama.com/windows) documentation. The first run is
online and should be performed only from the verified client bundle.

The native Linux launcher targets glibc-based x86-64 and arm64 distributions.
Musl-based distributions require a different Ollama build and are not supported
by this bundle.

`UMA_BCT_NO_OPEN=1` suppresses opening folders for headless smoke tests.

## Catalogue behavior

The first five columns are `Node ID`, `Node Type`, `Node Title`, `Description`,
and `Date Range`. `Node Type` is `Unit`; Node ID and Date Range stay blank in
version 0.1. Canonical titles use the first recognised record identifier and a
four-digit unit number. Ambiguous values remain blank or review-flagged, and
separate physical boxes remain separate rows.

Progress and warnings go to stderr. Completed CLI commands emit one JSON result
to stdout. Exit codes are `0` for complete, `1` for partial or recorded failure,
and `2` for setup or usage failure.

## Privacy and remote Ollama

The supported client launcher always uses numeric loopback Ollama and removes
the remote privacy opt-ins from its child environment. Images and catalogue
data stay on the workstation. Loopback Ollama has no client authentication, so
the supported boundary is a trusted single-user workstation.

Direct CLI operators can opt into an approved managed endpoint only with HTTPS:

```bash
export UMA_BCT_ALLOW_REMOTE_OLLAMA=1
export OLLAMA_HOST=https://approved-model-service.example.edu:11434
uma-bct
```

Never commit, upload, package, place in a Docker context, or send to hosted CI:

- `Archive box photos/` or `add_repository_photos/`
- `inputs/`, `outputs/`, or `.uma-bct-runtime/`
- model weights, Ollama state, reviewed expectations, or benchmark output

## Architecture and recovery

```text
preflight local files and Ollama
  -> Qwen box localisation
  -> OpenCV localisation fallback when needed
  -> detector-neutral review crops
  -> Qwen crop transcription
  -> one full-image recovery attempt when required
  -> consolidate observations without merging physical boxes
  -> map and flag catalogue rows
  -> atomically write catalogue and run manifest
```

Each image has its own exception boundary. Detection and transcription attempts
are recorded before consolidation. The protected Confluence diagram was not
available for exact comparison, so this describes the implemented flow only.

## Optional Docker workflow

Native Ollama remains the preferred technical path. For container-based teams:

```bash
mkdir -p inputs outputs
docker compose config --quiet
UMA_BCT_UID="$(id -u)" UMA_BCT_GID="$(id -g)" \
  docker compose run --rm app
```

The Compose service keeps inputs read-only, writes only to `outputs/`, stores
Ollama state in a named volume, and exposes no Ollama host port. Docker Desktop
on macOS does not provide GPU acceleration to this Linux-container workflow.

## Build and release

Build and inspect all client artifacts with:

```bash
uv export --quiet --locked --no-dev --group client-bootstrap \
  --no-emit-project --no-header --format requirements-txt \
  --output-file client-constraints.txt
uv build --no-build-isolation
uv run python scripts/build_client_bundle.py dist
uv run python scripts/check_release_artifacts.py dist
```

The bundle builder requires exactly one wheel and creates
`dist/uma-bct-client.zip`. Release CI smoke-tests wheel and sdist entry points,
checks the client bundle, verifies checksums, and publishes only the inspected
artifacts.

The private real-model gate remains local. It uses human-reviewed expectations
stored beside ignored source photos and must never run on hosted CI.
