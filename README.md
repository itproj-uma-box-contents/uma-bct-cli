# UMA Box Contents Transcription CLI

This tool turns photographs of archive boxes into a catalogue that a person can
review.

It finds boxes in each photo, saves a crop of each box, and uses a local Ollama
model to read labels and handwriting. It then creates a CSV or Excel file.


## What you need

- Python 3.12
- [uv](https://docs.astral.sh/uv/) for installing and running the project
- [Ollama](https://ollama.com/download) running on the same computer
- At least 16 GB of memory for the default model

The first setup needs an internet connection to download Python packages and the
model. After that, normal transcription can run offline.

Native Ollama is the recommended option, especially on a Mac where it can use
Metal acceleration. Docker is available as a slower, CPU-first alternative.

## Quick start

Install the model once:

```bash
ollama pull qwen3-vl:8b-instruct-q4_K_M
```

If Ollama is not already running through its desktop app, start it in a separate
terminal:

```bash
ollama serve
```

From a copy of this repository, install the locked dependencies and check that
the command works:

```bash
uv sync --locked --dev
uv run uma-bct --help
```

Put the photos you want to process in an `inputs/` folder, then run:

```bash
uv run uma-bct \
  --path-to-inputs ./inputs \
  --path-to-outputs ./outputs/catalogue.csv \
  --path-to-box-images ./outputs/crops
```

You can use an `.xlsx` file instead of `.csv` if you want an Excel workbook.
You can also give the tool one image file instead of a folder.

If you installed the released wheel instead of working from this repository,
install it with `uv tool install ./uma_bct_cli-*.whl`, then use `uma-bct` in
place of `uv run uma-bct`.

## What the tool creates

Each run creates three things:

- A CSV or Excel catalogue.
- A folder of review crops: one image for each detected box.
- A `.run.json` file with the settings, warnings, failures, and a full record of
  the run.

The catalogue has these columns:

| Column | Meaning |
| --- | --- |
| `Node ID` | Left blank for now. |
| `Node Type` | Always `Unit`. |
| `Node Title` | The record identifier and box number, for example `1974.0029 Unit 0319`. |
| `Description` | Handwritten notes and other useful label text. |
| `Date Range` | Left blank for now. |
| `Confidence Score` | A guide for human review, not a guarantee. |
| `Flags` | Things that need checking, such as unclear text or a fallback. |
| `Image` | The saved crop for that row. |

The tool understands common identifier formats such as `1960.0009`, `96/9`, and
`UMA-AC-000000330`. It pads valid box numbers to four digits. If it sees more
than one identifier, or cannot read an identifier or box number clearly, it
keeps the row and adds a flag for review.

Separate physical boxes stay as separate rows, even if they have the same title.

## Useful options

```text
-i, --path-to-inputs PATH          A photo or a folder of photos
-o, --path-to-outputs PATH         Where to write the CSV or XLSX file
-b, --path-to-box-images PATH      Where to save review crops
-c, --min-confidence 0..100        Flag lower-confidence transcription (default: 80)
-f, --force                        Replace an existing catalogue
    --version                      Show the installed version
```

Run `uma-bct --help` to see all options, including detector choice, model name,
and request timeout.

The command checks the input folder, output location, Ollama connection, and
model before it starts. It will not overwrite an existing catalogue unless you
add `--force`.

During a run, progress and warnings appear in the terminal. At the end, the
tool prints one short JSON summary. Exit code `0` means every image finished;
`1` means some images failed or need recovery; `2` means the command or setup
was invalid.

## How it handles problems

The normal process is:

```text
Find boxes with Qwen
  -> try OpenCV if that does not work
  -> read the whole photo if no useful crops remain
```

If one box cannot be read, the rest of the batch can still be exported. A failed
or uncertain box is recorded in the run file and marked for review. A run that
finds no usable rows will not replace an existing catalogue.

## Privacy

Photos stay on the computer by default. The app talks only to local Ollama
unless an organisation explicitly enables an approved remote Ollama server.

Do not add archive photos, human-reviewed results, generated catalogues, or
model files to Git. The repository, Docker build, packages, and GitHub Actions
are set up to exclude them. In particular, `Archive box photos/`,
`add_repository_photos/`, `inputs/`, and `outputs/` are ignored.

Keep the catalogue, its `.run.json` file, and its crop folder together while
reviewing results. On Windows, keep the catalogue and crop folder on the same
drive.

## Docker

Docker is optional. It is useful when the team wants the same setup on different
computers, but native Ollama is usually faster on a Mac.

To build the app image and check it starts:

```bash
docker build --target production -t uma-bct-cli:local .
docker run --rm uma-bct-cli:local --help
```

To run the full Docker setup, put copies of the photos in `inputs/` and create
an `outputs/` folder beside the Compose file:

```bash
mkdir -p inputs outputs
docker compose run --rm app
```

On Linux, use this command instead so output files belong to your user:

```bash
UMA_BCT_UID="$(id -u)" UMA_BCT_GID="$(id -g)" docker compose run --rm app
```

Compose keeps input files read-only, writes results to `outputs/`, stores the
model in a named Docker volume, and does not expose Ollama to the network.

## Checks and GitHub Actions

GitHub Actions runs on every pull request and every push to `main`. It checks:

- code style and formatting;
- unit tests on Ubuntu, macOS, and Windows;
- that the built package installs and starts correctly; and
- that the Docker image starts correctly.

These checks do not upload private photos, start Ollama, or download the model.

When a protected release tag is pushed, the release workflow builds the package,
checks it again, creates checksums, and publishes the private GitHub Release and
private multi-platform Docker image. Repository administrators should mark the
CI jobs as required checks before merging pull requests.

## For contributors

Run the same basic checks as GitHub Actions before opening a pull request:

```bash
uv sync --locked --dev
uv run ruff check .
uv run ruff format --check .
uv run coverage run -m unittest discover -s tests -v
uv run coverage report
uv build --no-build-isolation
```

The automated tests use fake model responses. Real-photo testing must happen on
approved local hardware only, followed by human review of the catalogue.
