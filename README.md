# The University of Melbourne Archives' Box Contents Transcription CLI

Official repository for the University of Melbourne Archives' Box Contents Transcription CLI.

## Getting started

Setup is easiest with [uv](https://docs.astral.sh/uv/). Simply clone the repository, open it in a terminal, and run:
```
uv sync
```

Activate the virtual environment you just created:
```
source .venv/bin/activate # macOS, Linux
```
```
.venv\Scripts\activate.bat # Windows
```

Then run the code:
```
uv run main.py --path-to-inputs="/path/to/input/folder" --path-to-outputs="/path/to/outputs"
```

Or use the short-hand flags:
```
uv run main.py -i="/path/to/input/folder" -o="/path/to/outputs"
```

## CLI Notes

- If `--path-to-inputs`/`-i` isn't set, it defaults to the current working directory.
- If `--path-to-output`/`-o` isn't set, it defaults to `/outputs` inside the current working directory.

## Testing

Run unit tests with:
```
uv run python -m unittest discover -s tests -v
```