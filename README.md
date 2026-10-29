# UMA Box Contents Transcription

Built by Dimitrios Petrakis, Morayya, Zane, William Zhang and Finley Owen with
University of Melbourne Archives (UMA). Thank you to our client, Senior Archivist
Lachlan Glanville, for guiding the project.

A CLI that detects archive boxes in shelf photos, transcribes their labels and
handwriting, and exports a CSV or Excel catalogue for human review. Local Qwen3-VL
through Ollama is the default; local processing works offline after initial setup.
For hosted processing, **Gemini is the team's tested and recommended option**;
see the [hosted-model guide](docs/hosted-models.md).

## Run the tool

1. Get `archive-box-transcription.zip` from the project maintainers and unzip it
   into a private local folder outside cloud-sync and network drives.
2. On **Windows**, double-click `Run UMA BCT.cmd`. On **macOS/Linux**, open a terminal
   in the unzipped folder and run:

   ```sh
   sh run-uma-bct.sh
   ```

3. Let first-run setup finish, add photos to `Add Photos Here` when prompted,
   and press Enter. Keep the terminal open until processing finishes.
4. Open the catalogue in `Results/Batch ...`. Review every source photo for missed
   boxes, then check every row against its image in `Box Images` before import.

CSV is the default. For a new Excel batch, use `sh run-uma-bct.sh --format excel`
or, in Windows PowerShell, `& '.\Run UMA BCT.cmd' --format excel`.
Restart the same launcher to resume interrupted work with its original photos and
options. Completed results and reviewer edits are preserved; `--reprocess` creates
separate results for photos already processed.

First-run setup needs internet and disk space for the runtime and model. Allow at
least 16 GB RAM; CPU inference works but is slow. Supported platforms are Windows
10 22H2+ (x86-64), macOS 14+ (Intel/Apple silicon), and Linux with glibc 2.28+
(x86-64/arm64). See the [CLI guide](docs/usage.md) for options, formats and recovery.

## Contribute or develop

From a source checkout, install `uv` and run:

```sh
uv sync --locked --dev --python 3.12
uv run uma-bct --help
```

Read [CONTRIBUTING.md](CONTRIBUTING.md) for setup, pull requests, testing, builds and
local accuracy checks. Report bugs or propose improvements through repository issues.

## Reference

| Guide | Contents |
| --- | --- |
| [CLI guide](docs/usage.md) | Arguments, supported files, managed batches and recovery |
| [Catalogue and review](docs/catalogue.md) | Column rules, flags, confidence and verification |
| [Hosted models](docs/hosted-models.md) | Optional API configuration and troubleshooting |
| [Architecture](docs/architecture.md) | Module map, runtime, checkpoints and publication |

Project background, team roles, requirements and sprint records remain in
[Confluence](https://uma-box-contents.atlassian.net/wiki/spaces/UBC/pages/1015827)
(access may be required). The guides above describe the current CLI; the exported
Confluence manuals include earlier behaviour.

Keep private photos, crops, catalogues, model weights and credentials local and out
of Git, Docker contexts and hosted CI. Hosted inference sends image pixels to a
provider and requires written client approval for private material. All output needs
human review, including unflagged rows and readings where models agree.

## Publication approval

Public release has been approved by the University of Melbourne and
University of Melbourne Archives (UMA).
