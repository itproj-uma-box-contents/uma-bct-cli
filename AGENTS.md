# UMA BCT CLI repository guide

## Purpose and authority

This repository contains the University of Melbourne Archives Box Contents
Transcription CLI. It processes shelf photographs locally, detects each physical
archive box, transcribes printed labels and handwriting with Qwen3-VL through
Ollama, and exports a human-reviewable catalogue.

When requirements conflict, use this order of authority:

1. The client's import template and written email requirements.
2. The UMA Box Contents Confluence space.
3. This repository's tested behaviour and documentation.
4. `OpenCV-QWEN-Test` as a reference for proven detection fallback behaviour,
   not as a source to copy wholesale.

## Client catalogue rules

- The first five export columns are `Node ID`, `Node Type`, `Node Title`,
  `Description`, and `Date Range`.
- `Node Type` is `Unit`. `Node ID` and `Date Range` remain blank in v0.1.
- A canonical title is `{first record identifier} Unit {four-digit box number}`.
- Recognised identifiers include `YYYY.NNNN`, `YY/N`, and
  `UMA-(AC|SR)E?-\d{8,9}`. When multiple identifiers are present, use the first
  one and flag the record for review.
- `Description` contains handwritten and explicit auxiliary content/condition
  annotations joined with semicolons. Record-origin text is not part of the
  canonical title.
- Ambiguous text must be blank or review-flagged, never confidently invented.
  Preserve separate rows for separate physical boxes, including duplicates.

## Privacy and supported delivery

- `Archive box photos/` and `add_repository_photos/` contain private client
  photographs, potentially including GPS-bearing EXIF. Never commit, upload,
  publish, add to Docker contexts, or use them on hosted CI runners.
- Model weights and Ollama state are also local-only. Do not add them to Git,
  packages, containers, releases, or CI caches.
- The CLI permits only numeric loopback Ollama endpoints by default. The bundled
  Compose service explicitly opts into its exact internal `ollama` hostname.
  Any managed remote endpoint requires the privacy opt-in documented in the
  README and must use HTTPS because image pixels leave the device.
- Supported v0.1 platforms are macOS and Linux on x86-64/arm64 and Windows on
  x86-64, using Python 3.12. Native Ollama is preferred; Docker is an optional
  Linux-container path. CPU inference is supported but slow.
- The v0.1 default is the pinned Qwen3-VL 8B quantized model. Qwen detection is
  primary and OpenCV is an experimental fallback.

## Development commands

Use the locked environment and keep model-free checks deterministic:

```bash
uv sync --locked --dev
uv run ruff check .
uv run coverage run -m unittest discover -s tests -v
uv run coverage report
uv build
```

Before release, install both the built wheel and sdist outside the checkout and
smoke-test `uma-bct --help`, `uma-bct --version`, and `python -m uma_bct`.
Real-model regression runs stay local and use human-reviewed expectations stored
beside the ignored source photographs.

The v0.1 real-model release gate covers all 21 images in
`add_repository_photos/` plus these 11 JPEGs from `Archive box photos/`:
`230150462`, `230251945`, `230445954`, `230518743`, `230624828`, `230737955`,
`230824065`, `230850841`, `231116923`, `231212408`, and `231905575` (each is the
numeric suffix of its `PXL_*.jpg` filename). Keep expected results, review notes,
and benchmark output inside an ignored private-photo directory.

## Engineering constraints

- Keep the pipeline modular and local-first. Preserve per-image resilience,
  traceability, atomic exports, explicit partial/failure status, and clean JSON
  stdout; progress and warnings belong on stderr.
- Prefer small, tested fixes. Do not port experimental preprocessing, caches,
  telemetry, or debug visualisation without a demonstrated regression.
- Preserve existing formatting when editing or adding code; do not run
  automated formatters.
- Deferred beyond v0.1: Node-ID reconciliation, date extraction, automatic text
  rotation, sequence-gap analysis, GUI, catalogue APIs, and signed executables.
