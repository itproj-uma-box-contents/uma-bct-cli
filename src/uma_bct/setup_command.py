"""Prepare the local folders and Ollama model.

Setup can be run more than once. It checks folder access, downloads the model
only when needed, and reports failures as simple JSON.
"""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

from .adapters.hosted_client import HostedVisionModel
from .adapters.ollama_client import DEFAULT_MODEL, OllamaVisionModel
from .managed_batch import ResultsRootLock
from .model_config import HostedModelConfig
from .path_safety import check_path, redirected_path_kind
from .run_artifacts import CatalogueLock
from .terminal import TerminalProgress


def ensure_directory(path: Path) -> int:
    try:
        check_path(path, "Setup directory")
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        check_path(path, "Setup directory")
        metadata = path.lstat()
    except (OSError, ValueError) as error:
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


def safe_error(error: Exception) -> str:
    printable = "".join(character if character.isprintable() else " " for character in str(error))
    return (" ".join(printable.split()) or error.__class__.__name__)[:1000]


def validate_setup_layout(inputs: Path, outputs: Path) -> None:
    input_path = os.path.realpath(inputs)
    output_path = os.path.realpath(outputs)
    try:
        common = os.path.commonpath((input_path, output_path))
    except ValueError:
        return
    if common in {input_path, output_path}:
        raise RuntimeError("Setup input and results directories must be separate and non-nested.")


def run_setup(*, root: Path | None = None, model_name: str = DEFAULT_MODEL, input_directory: Path | None = None,
    results_directory: Path | None = None, hosted_model: HostedModelConfig | None = None) -> int:
    """Prepare folders and model while showing that a long download is active."""

    with TerminalProgress(None) as reporter:
        return _run_setup(
            root=root, model_name=model_name, input_directory=input_directory,
            results_directory=results_directory, reporter=reporter, hosted_model=hosted_model,
        )


def _run_setup(*, root: Path | None, model_name: str, input_directory: Path | None,
    results_directory: Path | None, reporter: TerminalProgress, hosted_model: HostedModelConfig | None = None) -> int:
    """Keep the existing setup result and cleanup in one place."""

    working_directory = (root or Path.cwd()).absolute()
    inputs = input_directory or working_directory / "inputs"
    outputs = results_directory or working_directory / "outputs"
    if not inputs.is_absolute():
        inputs = working_directory / inputs
    if not outputs.is_absolute():
        outputs = working_directory / outputs
    result: dict[str, object] = {
        "schema_version": 1,
        "command": "setup",
        "status": "complete",
        "input_directory": str(inputs),
        "output_directory": str(outputs),
        "model": hosted_model.model if hosted_model is not None else model_name,
    }
    lock: CatalogueLock | ResultsRootLock | None = None
    status = 0
    # Verify private folders and reserve the selected output layout before a
    # model pull can spend time or write state for the wrong client folder.
    try:
        reporter.phase("Checking local folders")
        validate_setup_layout(inputs, outputs)
        warn_about_permissions(inputs, ensure_directory(inputs))
        warn_about_permissions(outputs, ensure_directory(outputs))
        lock = ResultsRootLock(outputs) if results_directory is not None else CatalogueLock(outputs / "catalogue.csv")
        lock.acquire()
        if hosted_model is not None:
            reporter.phase("Checking the hosted client")
            HostedVisionModel(hosted_model).preflight()
        else:
            reporter.phase("Checking or downloading the local model")
            OllamaVisionModel(model=model_name).ensure_model(progress_stream=sys.stderr)
    except Exception as error:
        message = safe_error(error)
        result.update(status="failed", error=message)
        print(f"Setup failed: {message}", file=sys.stderr)
        status = 2
    finally:
        if lock is not None:
            try:
                lock.release()
            except Exception as error:
                message = safe_error(error)
                existing_error = result.get("error")
                combined = f"{existing_error}; lock cleanup failed: {message}" if existing_error else message
                result.update(status="failed", error=combined[:1000])
                print(f"Lock cleanup failed: {message}", file=sys.stderr)
                status = 2
    # Setup returns exactly one JSON object on stdout. Progress, warnings, and
    # download updates stay on stderr through the reporter and model client.
    if status == 0:
        reporter.phase("Setup complete.")
    reporter.close()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return status
