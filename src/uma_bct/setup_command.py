"""One-command setup for the local client workflow."""

from __future__ import annotations

import json
import stat
import sys
from pathlib import Path

from .adapters.ollama_client import DEFAULT_MODEL, OllamaVisionModel
from .path_safety import redirected_path_kind


def ensure_directory(path: Path) -> int:
    """Create one private working directory without replacing existing content."""

    try:
        path.mkdir(mode=0o700, exist_ok=False)
    except FileExistsError:
        pass
    except OSError as error:
        raise RuntimeError(f"Could not create directory {path}: {error}") from error

    try:
        metadata = path.lstat()
        redirected_kind = redirected_path_kind(path, metadata)
    except OSError as error:
        raise RuntimeError(f"Could not inspect directory {path}: {error}") from error
    if redirected_kind is not None:
        raise RuntimeError(
            f"Setup directory is a {redirected_kind}; use a real directory "
            f"instead: {path}"
        )
    mode = metadata.st_mode
    if not stat.S_ISDIR(mode):
        raise RuntimeError(f"Setup path exists but is not a directory: {path}")
    return mode


def warn_about_directory_permissions(path: Path, mode: int) -> None:
    """Warn about existing POSIX permissions without changing user content."""

    shared_permissions = stat.S_IRWXG | stat.S_IRWXO
    if sys.platform != "win32" and mode & shared_permissions:
        print(
            f"Warning: {ascii(str(path))} is accessible by other users. Restrict "
            "it to owner-only permissions (mode 700) before adding private "
            "archive data.",
            file=sys.stderr,
        )


def safe_error_message(error: Exception) -> str:
    """Return bounded single-line error text safe for JSON and a terminal."""

    raw = str(error) or error.__class__.__name__
    printable = "".join(
        character if character.isprintable() else " " for character in raw
    )
    return (" ".join(printable.split()) or error.__class__.__name__)[:2000]


def run_setup(*, root: Path | None = None, model_name: str = DEFAULT_MODEL) -> int:
    """Prepare local folders and the pinned model, then emit one JSON result."""

    working_directory = (root if root is not None else Path.cwd()).absolute()
    input_directory = working_directory / "inputs"
    output_directory = working_directory / "outputs"
    result: dict[str, object] = {
        "schema_version": 1,
        "command": "setup",
        "status": "complete",
        "input_directory": str(input_directory),
        "output_directory": str(output_directory),
        "model": model_name,
    }
    try:
        input_mode = ensure_directory(input_directory)
        output_mode = ensure_directory(output_directory)
        warn_about_directory_permissions(input_directory, input_mode)
        warn_about_directory_permissions(output_directory, output_mode)
        model = OllamaVisionModel(model=model_name)
        model.ensure_model(progress_stream=sys.stderr)
    except Exception as error:
        message = safe_error_message(error)
        result["status"] = "failed"
        result["error"] = message
        print(f"Setup failed: {message}", file=sys.stderr)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 2

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0
