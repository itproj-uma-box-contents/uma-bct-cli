# Architecture and recovery

[README](../README.md) · [Contributing](../CONTRIBUTING.md) · [Usage guide](usage.md)

## Code map

| Area | Main modules |
| --- | --- |
| Commands and setup | `cli.py`, `setup_command.py`, `bootstrap.py` |
| Direct and managed runs | `batch_runner.py`, `managed_batch.py` |
| Detection, image preparation and transcription | `pipeline.py`, `imaging.py`, `transcriptions.py`, `adapters/` |
| Label interpretation and catalogue output | `identity.py`, `catalogue.py`, `export.py`, `util/date_util.py` |
| Second readings and reconciliation | `verification.py` |
| Filesystem integrity and publication | `path_safety.py`, `atomic.py`, `run_artifacts.py` |
| Model identity and application version | `model_config.py`, `version.py` |
| Terminal feedback and sleep prevention | `terminal.py`, `keep_awake.py` |

Modules are under `src/uma_bct/`. `domain.py` contains shared values and `ports.py`
defines the adapter interfaces. The launchers prepare the runtime and invoke the
same CLI packaged by `pyproject.toml`.

## Design records

Confluence holds the [architectural requirements](https://uma-box-contents.atlassian.net/wiki/spaces/UBC/pages/1998895),
[domain model](https://uma-box-contents.atlassian.net/wiki/spaces/UBC/pages/3768406)
and [class/sequence diagrams](https://uma-box-contents.atlassian.net/wiki/spaces/UBC/pages/8912936)
(access may be required). Those exported designs describe an earlier release plan;
this module map and the notes below cover the current implementation.

## Runtime and recovery constraints

Launchers verify the wheel and locked dependencies, prepare private Python 3.12 in
`.uma-bct-runtime/`, and invoke the CLI. Local mode owns a separate Ollama service
on a temporary loopback port and stops only its own processes. Hosted mode skips
Ollama. Runtime/model repair requires empty photo/results folders. Generate
`client-constraints.txt` from `uv.lock`; never edit it by hand.

```text
validate -> model preflight -> detect -> crop -> transcribe -> optional verification
  -> catalogue mapping -> recoverable catalogue and run details
```

- Keep successful boxes when another crop/photo fails. OpenCV is an experimental
  fallback; use one full-image recovery when localisation or all crop readings fail.
- Managed batches snapshot inputs, hold an OS-backed results lock, and checkpoint
  photo outcomes and verification readings atomically. Preserve source/crop hashes,
  backend/model identity, routes, warnings, failures and alternative readings.
- Resume validates saved inputs, crops and options. Cached verification and
  publication-only recovery work without loading a model. Source directory,
  filename and fingerprint identify history; identical pixels alone do not.
- Publish after all photos and requested checks are terminal. Partial batches retain
  usable rows; zero-row batches publish failed run details only. Stage and hash the
  catalogue before publication; preserve completed catalogues and reviewer edits.
- Keep legacy checkpoints readable. Older staged mappings replay only if the whole
  regenerated file hash matches. Legacy HTML metadata does not block recovery;
  new runs generate no HTML.

See [AGENTS.md](../AGENTS.md) for detailed maintenance invariants,
[CLI guide](usage.md) for resume instructions, and
[Contributing](../CONTRIBUTING.md) for regression and delivery checks.
