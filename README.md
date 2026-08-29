# University of Melbourne Archives Box Contents Transcription CLI

This project detects individual archive boxes in shelf photographs, creates
full-resolution crops, and transcribes printed labels and handwriting with a
local vision model. Detection and transcription are independent components:
choose either OpenCV or Qwen for detection while using Qwen for transcription.

```text
OpenCV detector ─┐
                 ├─> DetectedRegion[] ─> shared OpenCV cropper ─> Box[] ─> Qwen transcriber
Qwen detector ───┘
```

The Qwen adapters use a shared local Ollama client, but do not know about each
other. A new detector or transcriber only needs to implement the corresponding
protocol in `src/ports.py` and be registered in `src/bootstrap.py`.

## Requirements

- Python 3.12 or newer
- [`uv`](https://docs.astral.sh/uv/) for Python dependency management
- Ollama installed locally and running
- A local vision model; the default is `qwen3-vl`

`uv sync` installs the Python dependencies declared in `pyproject.toml`:

- `numpy` for image arrays and detector geometry
- `opencv-python` for OpenCV detection, image loading, and crop creation
- `ollama` for calls to the locally running vision model
- `pandas` for catalogue mapping, CSV/XLSX tables, and duplicate flagging
- `openpyxl` for writing XLSX workbooks

Install the Python environment:

```bash
uv sync
```

Before the first offline run, install Ollama using its platform instructions and
download the model while connected:

```bash
ollama pull qwen3-vl
```

Model weights are intentionally not downloaded by this application. Once the
model is present, normal image processing is local and offline.

### Reference-model compatibility

The default model request matches `~/projects/opencv-qwen-test`:

- Ollama model tag: `qwen3-vl`
- Expected reference model: Qwen3-VL 8.8B, `Q4_K_M`
- Thinking: disabled
- Temperature: `0`
- Context length: `16384`
- Response mode: streamed structured JSON

The OpenCV thresholds and crop geometry, Qwen detection prompt/schema, Qwen
transcription prompt/schema, and these request settings are protected by
regression tests in `tests/test_reference_compatibility.py`.

## Running

The input may be one image or a flat directory of images. Choose the detector
independently from the Qwen text transcriber.

### OpenCV detection + Qwen transcription

Use this route to test the local OpenCV geometry detector while retaining Qwen
for reading labels and handwriting:

```bash
uv run main.py \
  -i /path/to/images \
  -o outputs/opencv-catalogue.csv \
  --work-dir outputs/opencv-crops \
  --detector opencv \
  --transcriber qwen
```

### Qwen detection + Qwen transcription

Use Qwen for both shelf-level box detection and crop-level transcription:

```bash
uv run main.py \
  -i /path/to/images \
  -o outputs/qwen-catalogue.xlsx \
  --work-dir outputs/qwen-crops \
  --detector qwen \
  --transcriber qwen
```

### One image instead of a directory

```bash
uv run main.py \
  -i /path/to/shelf-001.jpg \
  -o outputs/shelf-001.csv \
  --detector opencv \
  --transcriber qwen
```

### Windows Ollama server from WSL

When the CLI runs in WSL but Ollama runs locally on Windows, use the Windows
server's local API address (after configuring WSL networking so it is reachable):

```bash
OLLAMA_HOST=http://127.0.0.1:11434 uv run main.py \
  -i "/mnt/c/Users/username/Pictures/shelf-001.jpg" \
  -o outputs/windows-ollama-test.csv \
  --detector opencv \
  --transcriber qwen
```

Clean box crops are written to the selected `--work-dir` (or
`outputs/crops/<source-name>/` by default). The catalogue is written as CSV or
XLSX according to the `-o` filename, and a processing summary is printed as
JSON. Duplicate `Title` values are flagged on every implicated row without
deleting records. Date parsing and preprocessing are not included, so `Start
Date` and `End Date` remain empty.

### OpenCV detection status

The OpenCV route runs, and its current thresholds and crop geometry are checked
against the reference implementation. Its real-world detection accuracy and
stability across archival shelf photographs have not yet been formally
benchmarked. Do not assume it is 100% accurate or that reference accuracy has
been maintained: inspect the generated crops and compare both detector routes
against manually verified images before selecting the production default.

The current catalogue mapping is:

- `Node`: `Unit`
- `Title`: printed label text through the detected unit line
- `Description`: remaining printed lines followed by handwritten lines
- `Start Date` and `End Date`: empty until date extraction is implemented
- `Confidence Score`: raw Qwen transcription confidence from 0 to 100
- `Flags`: `Missing unit` where applicable, plus `Potential Duplicate` after
  whole-batch duplicate checking and `Low confidence` below 50
- `Image`: path to the full-resolution box crop

## Useful future improvements

The following features would make the tool more practical after the current
MVP scaffold:

- **Date extraction:** populate `Start Date` and `End Date` while preserving the
  original annotation and flagging partial or uncertain dates.
- **Batch resilience:** retain successfully processed crops when another crop or
  image fails, and support safely resuming an interrupted batch.
- **Progress reporting:** show detected-box counts, the current transcription
  number, timings, and a concise end-of-batch summary in the terminal.
- **Image review links:** make the `Image` column easy to open from the review
  CSV/XLSX, including when the export is opened in Windows Excel.

Useful options:

```text
--model MODEL             local Ollama model tag; default qwen3-vl
--work-dir DIRECTORY      crop output directory; default outputs/crops
--min-confidence FLOAT    detector threshold from 0 to 1; default 0.60
```

The installed command is equivalent to `uv run main.py`:

```bash
uv run uma-bct -i /path/to/images -o outputs/catalogue.csv --detector opencv
```

## Replacing an adapter

The adapters intentionally resemble the functions in
`~/projects/opencv-qwen-test`:

- OpenCV detection logic is isolated in
  `src/adapters/opencv_detector.py`.
- The Qwen grounding prompt and validator are in
  `src/adapters/qwen_detector.py`.
- The Qwen transcription prompt and validator are in
  `src/adapters/qwen_transcriber.py`.
- Local model transport is in `src/adapters/ollama_client.py`.

Keep the public `detect(InputImage) -> list[DetectedRegion]` or
`transcribe(Box) -> BoxTranscription` method intact, replace the adapter's model
logic, and add the implementation to the appropriate registry in `bootstrap.py`.
The detector must return original-image pixel coordinates. The transcriber only
receives clean `Box` crops and must not call a detector.

## Testing

The default tests use fake detectors and model clients; they do not require an
Ollama service, model weights, or network connection:

```bash
uv run python -m unittest discover -s tests -v
```
