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
same place after the first run; the existing runtime, `inputs`, and `outputs`
folders are reused in place.

Supported computers are Windows 10 22H2 or newer on x86-64, macOS 14 or newer
on Intel or Apple silicon, and Linux with glibc 2.28 or newer on x86-64 or
arm64. Windows ARM, musl-based Linux, and older operating-system releases are
not supported. Allow at least 16 GB of memory and enough free disk space for the
large model. Use a trusted, single-user workstation rather than a shared
computer account.

Each supplied launcher starts its own Ollama service on a temporary numeric
loopback port. It checks that the service belongs to the current run and stops
it when the run finishes or is interrupted. An unrelated Ollama service on the
usual port is left alone. Direct CLI use still defaults to `127.0.0.1:11434`.
A developer using the CLI directly may use an approved managed endpoint only
by setting `OLLAMA_HOST` to an HTTPS URL and explicitly opting in with
`UMA_BCT_ALLOW_REMOTE_OLLAMA=1`. In that mode, image pixels leave the
workstation; do not enable it without written approval.

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

`catalogue.csv` is the result to review. The run manifest and crop images show
where each row came from. Before a later run replaces the current catalogue and
manifest, the launcher preserves timestamped copies beside them. Timestamped
crop folders are also kept. A hidden `.uma-bct.lock` exists in `outputs` only
while setup or a catalogue run is active. If the launcher reports a stale lock,
first make sure no UMA BCT run is still active, then remove the exact lock path
shown in the message and run the launcher again.

The catalogue and manifest are published together. If the manifest cannot be
saved after a new CSV is written, the previous CSV is restored (or a brand-new
CSV is removed) and a uniquely named failed-run manifest is attempted.
Timestamped previous results are retained, and rollback failures report the
recovery path to review.

Before importing the catalogue, work through the rows in their original photo
and left-to-right/top-to-bottom crop order:

- Compare each row with its `Crop Image` review evidence.
- Check every non-empty `Review Notes` value and the
  `Transcription Confidence (0-100)`. Empty review notes mean only that the tool
  raised no automated warning; every transcription still requires human review.
- Keep separate rows for separate physical boxes, even when their titles match.

If the launcher stops, read the short message in its window, correct the stated
problem, and run the same launcher again. Common first-run causes are no internet
connection, insufficient free disk space, or the operating system declining an
Ollama installation prompt. Setup also performs a one-token inference check, so
an incomplete Ollama runtime or insufficient memory is reported before any
photo crops are processed. The repair message includes the local Ollama log
path. Runtime repair is allowed only while `inputs` and `outputs` are empty, so
private files never coexist with online preparation.
