# UMA BCT CLI repository guide

Shared guidance for this repository. Paths are relative to its root; keep personal
paths, credentials, and machine settings out. Change only the requested scope.

## Purpose and sources

The University of Melbourne Archives Box Contents Transcription CLI turns shelf
photos into a catalogue for human review. Local Qwen3-VL through Ollama is the
default. See [README.md](README.md) for user behaviour and
[DEVELOPMENT.md](DEVELOPMENT.md) for development, delivery, and benchmarks.

For product requirements, follow this order:

1. The client's import template and written email requirements.
2. The UMA Box Contents Confluence space.
3. This repository's tested behaviour and documentation.
4. `OpenCV-QWEN-Test`, when available, for proven detection fallback behaviour.

Ask about material requirement conflicts. The reference project is optional;
do not copy it wholesale.

## Data flow and saved results

```text
Input/output validation -> model preflight -> box detection
  -> review crops -> transcription -> optional verification
  -> catalogue mapping -> sorting, flags, and missing-box suggestions
  -> atomic catalogue and run details
```

- Qwen detection is primary; OpenCV is an experimental fallback. Use one
  full-image recovery when localisation or all crop transcriptions fail.
  Keep successful boxes when another crop or photo fails.
- Managed batches snapshot inputs and checkpoint photos and verification
  readings. Later additions wait for the next batch. Retain source/crop hashes,
  model identity, routes, warnings, failures, and alternative readings. Cached
  verification and publication-only recovery must work without loading a model.
- Publish only after every input and requested verification has a terminal
  outcome. Preserve `complete`, `partial`, and `failed` status. Publish usable
  rows from partial batches; zero-row batches publish failed run details only.
  Preserve completed batches and reviewer edits on later launches.
- CSV is the default; Excel requires an explicit CLI option.
  Keep relative crop links portable when a batch moves. Legacy HTML metadata
  must not block catalogue recovery or reuse; new runs do not generate HTML.
- With `--json`, stdout contains one JSON summary; progress and warnings go to
  stderr. Unattended and JSON runs must never prompt.

## Client catalogue rules

- Keep the first four columns: `Unit Type`, `Unit Number`, `Description`,
  `Date Range`. Current exports then add `Transcription Confidence (%)`,
  `Review Flags`, and `Cropped Box Image Link`.
- `Unit Type` preserves all printed label text in reading order, including
  headings and record-origin text, excluding only the unit-number phrase.
  Collapse line breaks and repeated whitespace to single spaces.
- `Unit Number` contains one unambiguous box/unit number, padded to four digits.
  Check the structured claim against visible printed or handwritten evidence;
  leave missing, invalid, or conflicting numbers blank and flagged.
- Recognised identifiers include `YYYY.NNNN`, `YY/N`, and
  `UMA-(AC|SR)E?-\d{8,9}`; preserve other printed identifiers too. Keep malformed
  text visible and flag multiple identifiers. Do not misread printed origin
  headings as extra identifiers.
- `Description` joins handwriting and explicit auxiliary content/condition
  annotations with semicolons. Exclude identity-only text. `Date Range` is
  currently blank. Never invent ambiguous text or dates: leave blank or flagged.
- Keep one row per physical box, including duplicates. Sort by complete
  `Unit Type` case-insensitively, then numeric `Unit Number`; incomplete rows stay
  stable at the end. Flag matching identities as `Potential Duplicate`.
- Missing-box detection is enabled by default and can be disabled. Only suggest
  internal gaps of at most 100 missing units within the same complete `Unit Type`
  group. Suggested rows carry `Missing Box`, with description, confidence, and
  crop link blank. They do not establish that a physical box is missing.

## Verification and human review

- Review every source photo for missed boxes and false detections, then every
  catalogue row against its crop. Unflagged rows and model agreement still need
  human review.
- Keep catalogue flags consistent, including identity problems,
  transcription mismatches, fallbacks, duplicates, and verification outcomes.
- `Low confidence` means strictly below 50% by default; exactly 50% is not low.
  `--min-transcription-confidence` accepts 0–100 and flags rather than removes
  rows. The separate OpenCV threshold defaults to 0.60 on a 0–1 scale.
  Unavailable token confidence exports as 0 and requires review; scores are not
  measured accuracy. Review flags alone do not imply partial processing.
- Optional verification uses a blind second reading with the same model, then
  a third request only for substantive differences. Do not repeat detection or
  raise confidence because readings agree.
- Corrections require corroboration. Reconcile printed text, identifiers, and
  unit number together, including handwritten identity claims. Unresolved
  differences, content deletions, or ambiguous box matches keep the original
  reading and review flags. Request failures keep the first reading and mark
  partial processing; checkpoint failures stop the run. Preserve the original
  and comparison readings.

## Privacy and portability

- Private photos may contain GPS-bearing EXIF. Keep `Archive box photos/`,
  `add_repository_photos/`, `Add Photos Here/`, `inputs/`, crops, catalogues,
  manifests, review pages, and private expectations local and ignored. Never
  commit, publish, package, add to Docker contexts, or use them on hosted CI.
  Do not upload private photos or derived content without written client approval.
- Keep model weights, Ollama state, and `.uma-bct-runtime/` out of Git, packages,
  containers, releases, and CI caches. Keep credentials out of logs, manifests,
  command arguments, and committed files.
- Ollama accepts numeric loopback endpoints by default. Compose opts into only
  its exact internal `ollama` hostname. Remote Ollama requires HTTPS, written
  approval, and `UMA_BCT_ALLOW_REMOTE_OLLAMA=1`. Disable Ollama cloud features
  (`OLLAMA_NO_CLOUD=1`); cloud-tagged models are unsupported.
- Use Python 3.12; local inference defaults to pinned Qwen3-VL 8B. Platforms:
  Windows 10 22H2+ on x86-64, macOS 14+ on x86-64/arm64, and Linux with glibc
  2.28+ on x86-64/arm64. Native Ollama is preferred; Docker is optional. CPU
  inference works but is slow.

## Tests and release checks

Run from the repository root using the locked environment:

```text
uv sync --locked --dev --python 3.12
uv run ruff check .
uv run coverage run -m unittest discover -s tests -v
uv run coverage report
```

Use mocks and synthetic images for model-free tests and hosted CI; no paid API
calls or model installation. Match checks to the change. For documentation-only
edits, check content, links, and the diff. Behaviour changes need regression tests:

- Catalogue/review: field mapping, flags, confidence boundaries, duplicate and
  missing-box rows, Excel/CSV parity, formula safety, and portable crop links.
- Pipeline/batches: per-photo failures, bounded recovery, verification disputes
  and deletions, interruption/resume, legacy checkpoints, source/crop integrity,
  model/options consistency, locks, publication recovery, and zero-row batches.
- Launchers/adapters: owned-process cleanup, foreign-service protection, platform
  and input-format compatibility; fake provider clients for hosted tests.

Before delivery, run the full suite and the
[build/privacy checks](DEVELOPMENT.md#build-the-client-deliverables), including
`uv build --no-build-isolation`. Install wheel and sdist outside the checkout;
smoke-test `uma-bct --help`, `uma-bct --version`, and `python -m uma_bct --help`.
Validate the CI platform matrix and synthetic spreadsheet crop links,
including moved batches and Unicode, spaces, `#`, and `%` filenames.
Generate `client-constraints.txt` from `uv.lock`; never edit it by hand. Deliver
`archive-box-transcription.zip`; publishing releases is deferred.

The local real-model release gate covers all 21 images in
`add_repository_photos/` plus these 11 JPEGs from `Archive box photos/`:
`230150462`, `230251945`, `230445954`, `230518743`, `230624828`, `230737955`,
`230824065`, `230850841`, `231116923`, `231212408`, and `231905575` (numeric
suffixes of `PXL_*.jpg` filenames). These directories are optional for contributors;
report the gate as not run if the photos/model are unavailable.

Follow [the benchmark guide](DEVELOPMENT.md#verification-and-local-accuracy-references)
with `scripts/verification_benchmark.py`. Check every source photo, correct fields
and box counts, and set `review_status` to `reviewed` before scoring. Confirm box
IDs still identify the same physical boxes. Record missing/unexpected boxes,
field matches before/after verification, newly incorrect fields, elapsed time,
and model/options. Keep references, notes, and output in ignored private-photo
directories. Report outstanding checks; unreviewed drafts cannot prove accuracy.

## Code Review Rules

- Flag changes that break the client schema, discard physical boxes/evidence,
  hide failures, or overwrite reviewer work. Preserve the existing contract or
  obtain a documented requirement change and compatible migration.
- Flag changes that bypass privacy, endpoint validation, credential protection,
  formula/HTML escaping, file integrity, or atomic publication safeguards.
- Prefer small, tested changes. Keep the pipeline modular; require a demonstrated
  regression before adding preprocessing, caches, telemetry, or debug visuals.
