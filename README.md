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
- `pandas` for the retained duplicate-record flagging used by catalogue exports

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

## Running

Use Qwen to detect and transcribe boxes:

```bash
uv run main.py -i /path/to/images --detector qwen --transcriber qwen
```

Use OpenCV for detection and Qwen only for transcription:

```bash
uv run main.py -i /path/to/images --detector opencv --transcriber qwen
```

The input may be one image or a flat directory of images. Clean box crops are
written to `outputs/crops/<source-name>/` by default and the basic transcription
result is printed as JSON. No catalogue export, date parsing, duplicate
report generation, or preprocessing pipeline is included at this stage. The
existing duplicate-row validator is retained for the later CSV export stage.

Useful options:

```text
--model MODEL             local Ollama model tag; default qwen3-vl
--work-dir DIRECTORY      crop output directory; default outputs/crops
--min-confidence FLOAT    detector threshold from 0 to 1; default 0.60
```

The installed command is equivalent to `uv run main.py`:

```bash
uv run uma-bct -i /path/to/images --detector opencv
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
