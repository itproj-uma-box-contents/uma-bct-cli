# UMA Box Contents Transcription

This tool turns archive-box photos into a catalogue on your computer. The
supplied launcher workflow keeps photos local and does not upload them.

## Run the tool

1. Download and unzip the supplied `uma-bct-client.zip` file into a private
   folder that is not synced to iCloud, OneDrive, Dropbox, or a network drive.
2. Start UMA BCT:

   - **Windows:** double-click `Run UMA BCT.cmd`.
   - **macOS or Linux:** open Terminal in the unzipped folder and run:

     ```bash
     sh run-uma-bct.sh
     ```

3. The launcher opens an `inputs` folder. Drag or copy the photos into that
   folder, then return to the launcher and press Enter.

That is all the client needs to do. The launcher checks and prepares everything
else, processes the photos, and opens the `outputs` folder when it finishes.

The first run needs an internet connection and can take a while because it
installs the private local runtime and downloads the large transcription model.
The `inputs` folder opens after that preparation is complete. The operating
system may ask for permission to install Ollama. Later runs reuse what was
downloaded and can work offline. For another batch, replace the photos in
`inputs` and start the same launcher again. Keep the unzipped folder in the
same place after the first run.

Supported computers are Windows 10 22H2 or newer on x86-64, macOS 14 or newer
on Intel or Apple silicon, and Linux with glibc 2.28 or newer on x86-64 or
arm64. Windows ARM, musl-based Linux, and older operating-system releases are
not supported. Allow at least 16 GB of memory and enough free disk space for the
large model. Use a trusted, single-user workstation rather than a shared
computer account.

The supplied launchers always connect to Ollama on numeric loopback. A developer
may use an approved managed endpoint only by setting `OLLAMA_HOST` to an HTTPS
URL and explicitly opting in with `UMA_BCT_ALLOW_REMOTE_OLLAMA=1`. In that mode,
image pixels leave the workstation; do not enable it without written approval.

## Photos

Put one or more of these image types directly in `inputs`:

```text
.jpg  .jpeg  .png  .tif  .tiff  .webp
```

Do not put private archive photos in a cloud-synced folder. Adding a photo to
`inputs` is a local file copy, not an internet upload.

## Results

The launcher opens `outputs` when processing finishes. Review these together:

```text
outputs/
  catalogue.csv
  catalogue.run.json
  crops/
    run-.../
      ...review images...
```

`catalogue.csv` is the result to review. Rows are ordered by `Unit Type` and
then `Unit Number`. The run manifest and crop images show where each row came
from. Missing-box review rows have an empty `Crop Image` value because they do
not correspond to a detected physical box.


Before importing the catalogue, review the rows in `catalogue.csv`. Rows with
valid `Unit Type` and `Unit Number` values are sorted by `Unit Type` and then
by numeric `Unit Number` to make it easier to navigate the physical boxes.

Rows marked `Missing Box` are inserted between detected unit numbers when a
number is absent from an otherwise continuous sequence. These rows represent
boxes that may be missing and require confirmation. Use
`--no-missing-box` if missing-box detection is not required.

For every row:

- Open or inspect the file shown in the `Crop Image` column.
- Compare the crop with the transcribed unit type, unit number, description,
  and date range.
- Check every non-empty `Review Notes` value and the
  `Transcription Confidence (0-100)` value.
- Keep separate rows for separate physical boxes, even when their unit type
  and number are identical.

If the launcher stops, read the short message in its window, correct the stated
problem, and run the same launcher again. Common first-run causes are no internet
connection, insufficient free disk space, or the operating system declining an
Ollama installation prompt.
