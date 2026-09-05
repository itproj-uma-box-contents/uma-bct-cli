"""Prepare the local client folders and Ollama model."""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

from .adapters.ollama_client import DEFAULT_MODEL, OllamaVisionModel
from .path_safety import redirected_path_kind


def ensure_directory(path: Path) -> int:
    try:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        metadata = path.lstat()
    except OSError as error:
        raise RuntimeError(f"Could not prepare directory {path}: {error}") from error
    if kind := redirected_path_kind(path, metadata):
        raise RuntimeError(f"Setup directory must not be a {kind}: {path}")
    if not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError(f"Setup path is not a directory: {path}")
    if not os.access(path, os.R_OK | os.W_OK | os.X_OK):
        raise RuntimeError(f"Setup directory is not fully accessible: {path}")
    return metadata.st_mode


def warn_about_permissions(path: Path, mode: int) -> None:
    if sys.platform != "win32" and mode & (stat.S_IRWXG | stat.S_IRWXO):
        print(
            f"Warning: {ascii(str(path))} is accessible by other users; "
            "use owner-only permissions for private archive data.",
            file=sys.stderr,
        )


def _safe_error(error: Exception) -> str:
    printable = "".join(
        character if character.isprintable() else " " for character in str(error)
    )
    return (" ".join(printable.split()) or error.__class__.__name__)[:1000]


def run_setup(*, root: Path | None = None, model_name: str = DEFAULT_MODEL) -> int:
    working_directory = (root or Path.cwd()).absolute()
    inputs = working_directory / "inputs"
    outputs = working_directory / "outputs"
    result: dict[str, object] = {
        "schema_version": 1,
        "command": "setup",
        "status": "complete",
        "input_directory": str(inputs),
        "output_directory": str(outputs),
        "model": model_name,
    }
    try:
        warn_about_permissions(inputs, ensure_directory(inputs))
        warn_about_permissions(outputs, ensure_directory(outputs))
        OllamaVisionModel(model=model_name).ensure_model(progress_stream=sys.stderr)
    except Exception as error:
        message = _safe_error(error)
        result.update(status="failed", error=message)
        print(f"Setup failed: {message}", file=sys.stderr)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0
