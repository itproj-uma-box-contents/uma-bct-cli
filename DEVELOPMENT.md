# UMA BCT developer guide

The supported client delivery is `archive-box-transcription.zip`. It contains the wheel
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
Interactive progress is refreshed in place on stderr; captured stderr gets
short milestones. Transcription runs end with a readable report by default;
`--json` selects one JSON object for scripts. Setup keeps its existing JSON
result. The CLI requests an idle-sleep inhibitor while setup or a run is active.

The additive managed workflow is selected with `--results-directory`. It owns
readable `Batch YYYY-MM-DD HH-MM-SS` folders, checkpoints each source photo,
and resumes the one unfinished batch under that results root. It conflicts with
explicit `-o`, `-b`, and `--force`; `--reprocess` is valid only with this mode.
Use `uma-bct setup --path-to-inputs PATH --results-directory PATH` to prepare
custom managed locations.

For hosted development, install the optional client with
`uv sync --locked --dev --extra hosted`. A nonempty `UMA_BCT_API_KEY` plus a
provider-qualified `UMA_BCT_API_MODEL` selects the same hosted client for box
detection and transcription. Local Qwen remains the default when the key is
empty. The [README Gemini walkthrough](README.md#test-gemini-without-paying)
uses an invented photo and a Free Tier key; do not use private client photos
for that smoke test. Unit tests use a fake SDK and never make paid API calls.

## API documentation

`pdoc` builds a browsable API doc site from the package's docstrings. To
browse it locally with live-reload while editing docstrings:

```bash
uv run pdoc uma_bct
```

To write a static site instead (output is generated and ignored, like
`dist/`):

```bash
uv run pdoc uma_bct -o dist/api-docs
```

Open `dist/api-docs/uma_bct.html` in a browser to view it.

## Client bundle

New launchers use `Add Photos Here/`, `Results/`, and a private
`.uma-bct-runtime/` directory. They verify the wheel and hash-pinned
`client-constraints.txt`, install the managed Python 3.12 runtime, and open
the photo and result folders. With an empty API key, they also download and
verify pinned Ollama, start an owned service on an OS-selected numeric
loopback port, and load-probe the pinned Qwen model. With a nonempty key, they
check the hosted client and skip the Ollama download, service, and model pull.

Runtime downloads finish before the launcher asks the user to add private
photographs. Later local runs reuse the runtime and can run offline; hosted
runs need internet access. The owned Ollama process is stopped and reaped on
every local exit path; services from other folders and anything already using
port 11434 are neither reused nor terminated. Startup is retried on fresh
loopback ports, and failures point to `.uma-bct-runtime/ollama.log`. An
incomplete runtime is repaired only when the selected photo and result folders
are empty. A legacy bundle with data only in `inputs/` and `outputs/` continues
to use them in place. Data in both directory pairs is an error and is never
merged. Windows uses the checksum-pinned portable Ollama archive for local
mode and verifies the Authenticode signatures on both required executables;
it does not run the desktop installer or alter foreign services.

`client-constraints.txt` is generated from `uv.lock`, including the optional
hosted client. Do not edit it by hand. Every launcher installs the same
hash-verified dependency versions.

## Processing and privacy

```text
input and output validation
  -> selected model preflight (local Qwen or hosted HTTPS client)
  -> one shared vision client for box detection and transcription
  -> OpenCV detection fallback
  -> review crops and full-image recovery when needed
  -> conservative catalogue mapping
  -> per-photo atomic managed checkpoint
  -> recoverable CSV and final-details publication
```

Managed progress/details keep input SHA-256 fingerprints, backend and model
identity, endpoint history, detector route, final box readings, warnings, and
failures. They never store API keys. A managed batch cannot resume with a
different backend, model, provider, or hosted HTTPS origin. They deliberately
do not contain per-attempt timing or proposal telemetry. Direct local CLI use
continues to default to `127.0.0.1:11434`; local client launchers select an
isolated port.

Managed processing holds an OS-backed `.uma-bct.lock` in the results root.
Exclusive acquisition safely recovers a well-formed lock left by a dead
process; malformed, linked, non-regular, or racing locks remain blocked. Direct
mode retains its existing catalogue-lock semantics.

Managed batches do not publish a catalogue while photos are unattempted. After
all photos become terminal, sources and crop hashes are revalidated, a hidden
CSV is staged, and its digest is checkpointed before no-clobber publication.
`Run Progress.json` is then promoted to `Run Details.json`. Forced direct runs
retain timestamped prior files. If the manifest commit fails after CSV
replacement, the prior CSV is restored atomically or a newly created CSV is
removed, then a uniquely named failed-run manifest is attempted. A rollback
failure produces partial status and reports the retained recovery path.

Only numeric loopback Ollama is allowed by default. The bundled local Compose
service opts into its exact `ollama` hostname. An approved managed Ollama
endpoint requires `UMA_BCT_ALLOW_REMOTE_OLLAMA=1` and HTTPS because image
pixels leave the workstation. Hosted OpenAI, Anthropic, and Gemini use fixed
HTTPS origins and the explicit key opt-in documented in the README. Native
launchers and the local Compose service set `OLLAMA_NO_CLOUD=1`; cloud-tagged
models are unsupported. A separately managed local Ollama used by the direct
CLI must have cloud features disabled as well.

Never commit, package, upload, or place in a Docker context:

- `Archive box photos/`, `add_repository_photos/`, `Add Photos Here/`, or `inputs/`
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
is not exposed on a host port. The `hosted` Compose profile has no Ollama
dependency. Use it only with a key and the README's privacy opt-in; the free
Gemini smoke test should use synthetic photos.

## Build the client deliverables

When you need to give the application to a client, run this complete sequence
from the repository root. The first command prepares the locked build
environment, so these commands also work from a fresh checkout after `uv` has
been installed.

```bash
uv sync --locked --dev --python 3.12
uv export --quiet --locked --no-dev --extra hosted --group client-bootstrap \
  --no-emit-project --format requirements-txt \
  --output-file client-constraints.txt
uv build --no-build-isolation
uv run python scripts/build_client_bundle.py dist
uv run python scripts/check_release_artifacts.py dist
```

The final check must report that the release artifact privacy check passed.
The finished file is `dist/archive-box-transcription.zip`. Send only that ZIP
to the client through the approved private file-sharing method; the client can
then download it, unzip it, and run the launcher. Do not send the wheel, source
archive, repository, private photographs, or generated catalogue outputs.

`build/` and `dist/` are generated, ignored directories. The three scripts
build the client ZIP, reject private content in artifacts, and safely unpack
the Linux Ollama archive. Publishing releases and container images is deferred.

Before delivery, install the wheel and sdist outside the checkout and smoke-test
`uma-bct --help`, `uma-bct --version`, and `python -m uma_bct`. The
real-model photo gate remains local and is handled separately by the test
engineer.

Launcher changes also require the deterministic ownership fixtures in
`tests/test_client_delivery.py`. They cover foreign port-11434 services,
dynamic port selection, missing companions, bind retries, startup failure,
exit-code propagation, and owned-process cleanup without using a real model.
Managed-batch changes require interruption/resume, input and crop integrity,
publication recovery, formula-safety, naming-collision, and zero-row tests.

Before release, manually import dummy managed catalogues in Excel on Windows
and macOS, Numbers, and LibreOffice and confirm each relative crop link opens
after moving the whole batch folder. Google Sheets is not supported for local
private-file links. The private real-model gate remains local and covers all 21
`add_repository_photos/` images plus the 11 named JPEGs listed in `AGENTS.md`;
its photos, expectations, notes, and output must remain in ignored private
directories.
