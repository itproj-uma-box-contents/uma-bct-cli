# Contributing

[README](README.md) · [Architecture](docs/architecture.md) · [CLI guide](docs/usage.md)

## Set up a source checkout

These commands require the repository with `pyproject.toml`, tests and scripts.
The client ZIP prepares its own runtime; it is not a development checkout.

Install `uv`, reopen your terminal and check `uv --version`:

| Platform | Install command |
| --- | --- |
| macOS with Homebrew | `brew install uv` |
| macOS/Linux without Homebrew | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Windows PowerShell with WinGet | `winget install --id=astral-sh.uv -e` |

```bash
uv sync --locked --dev --python 3.12
uv run uma-bct --help
```

`uv` selects Python 3.12. For local inference, start native Ollama with
`OLLAMA_NO_CLOUD=1` set in its environment, then run:

```sh
uv run uma-bct setup --batch-outputs Results
# Add photos directly to inputs/ before processing.
uv run uma-bct --batch-outputs Results
```

Tests need neither Ollama nor model weights. Optional hosted development uses
`uv sync --locked --dev --extra hosted --python 3.12` and the
[hosted-model configuration](docs/hosted-models.md).

## Propose a change

Open an issue for a bug or substantial behaviour change. Work on a focused branch
and submit a pull request explaining the problem, change and checks performed.
Use Conventional Commits, for example `fix(catalogue): preserve printed label text`.
Follow [AGENTS.md](AGENTS.md), Ruff's settings in `pyproject.toml`, and the existing
module structure. Preserve catalogue contracts, source evidence and reviewer edits;
add regression tests for changed behaviour using synthetic data and fake clients.
Keep client material, credentials and model state out of issues, commits and CI.

Team governance stays in Confluence:
[coding standards](https://uma-box-contents.atlassian.net/wiki/spaces/UBC/pages/11763856),
[PR rules](https://uma-box-contents.atlassian.net/wiki/spaces/UBC/pages/15368543) and
[test plan](https://uma-box-contents.atlassian.net/wiki/spaces/UBC/pages/688527)
(access may be required). PR rules include two team approvals before merging.
The commands below apply to the current checkout. Confluence validation results
apply to their recorded commits; record new checks against the code being delivered.

## Run checks

```bash
uv run ruff check .
uv run coverage run -m unittest discover -s tests -v
uv run coverage report
```

For documentation-only edits, check content, relative links, anchors and the diff.
Changes to documentation packaging also need bundle tests and artifact checks.
Behaviour changes need relevant regression tests and the full suite before handover.
See [AGENTS.md](AGENTS.md#tests-and-release-checks) for coverage requirements.

CI covers Windows x86-64, macOS Intel/arm64 and Linux x86-64/arm64. Docker checks run
on both Linux architectures. `scripts/smoke_hosted_install.py` checks the installed
hosted SDK with synthetic replies and no provider requests. Report checks not run.

For API documentation, run `uv run pdoc uma_bct`, or generate ignored HTML with
`uv run pdoc uma_bct -o dist/api-docs`.

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

The privacy check must pass. Hand over `dist/archive-box-transcription.zip` through
approved private sharing. `build/` and `dist/` are generated/ignored; private photos
and outputs must never enter artifacts. Public release awaits selection and addition
of the approved open-source licence; container-image publishing remains deferred.

The ZIP includes the linked Markdown guides with their directory layout intact.
Keep the explicit file allowlists in `scripts/build_client_bundle.py` and
`pyproject.toml` current when moving documentation; unlisted documents must stay
out of the bundle and source archive.

Before delivery, install wheel and sdist outside the checkout and smoke-test
`uma-bct --help`, `uma-bct --version`, and `python -m uma_bct --help`. Run the full
suite and platform matrix. Launcher ownership/recovery fixtures live in
`tests/test_client_delivery.py`; keep them model-free.

Check synthetic spreadsheet links on macOS, Windows, and Linux with Unicode, spaces,
`#`, and `%`, also after moving the batch. Check CSV imports in Excel and LibreOffice
separately; Numbers users open named crops directly. Never upload private photos for
browser tests or run inference on hosted CI. The real-model gate stays local and is
handled separately by the test engineer.

## Optional Docker workflow

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
[hosted approval](docs/hosted-models.md). Hosted smoke checks use synthetic photos.

With Docker running, `python3 scripts/check_docker_context.py` checks build-context
exclusions using an isolated set of invented files. It never copies checkout photos
or credentials. CI runs this check on both Linux architectures before building the app.

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
