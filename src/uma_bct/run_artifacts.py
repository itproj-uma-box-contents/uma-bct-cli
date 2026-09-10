"""Run directories, manifests, and recoverable previous outputs.

Every run gets its own timestamped, owner-only (``0o700``) directory
(``run_directory``) so review crops from different runs never collide or get
silently overwritten. ``backup_existing_results`` is what makes ``--force``
safe: before a catalogue or manifest is replaced, both are copied aside under
a ``previous-<run-id>`` name first, so a forced re-run can never destroy the
only copy of a previous result. ``write_json_atomic`` uses the same
write-to-temp-file-then-replace pattern as ``export.export_catalogue``, so a
manifest is likewise never left half-written.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .path_safety import validate_path_hierarchy


def _run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


def run_directory(root: Path) -> Path:
    validate_path_hierarchy(root, "Review-crop root")
    destination = root / f"run-{_run_id()}"
    destination.mkdir(mode=0o700, parents=True)
    validate_path_hierarchy(destination, "Review-crop run directory")
    return destination


def manifest_path(output_path: Path) -> Path:
    return output_path.with_suffix(".run.json")


def failed_manifest_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}.failed-{_run_id()}.run.json")


def previous_result_path(path: Path, run_id: str) -> Path:
    if path.name.endswith(".run.json"):
        stem = path.name[: -len(".run.json")]
        return path.with_name(f"{stem}.previous-{run_id}.run.json")
    return path.with_name(f"{path.stem}.previous-{run_id}{path.suffix}")


def backup_existing_results(paths: tuple[Path, ...]) -> tuple[Path, ...]:
    """Keep visible timestamped copies before a forced replacement."""

    run_id = _run_id()
    backups: list[Path] = []
    for path in paths:
        validate_path_hierarchy(path, "Existing result")
        if not path.exists():
            continue
        backup = previous_result_path(path, run_id)
        validate_path_hierarchy(backup, "Previous result")
        shutil.copy2(path, backup)
        backups.append(backup)
    return tuple(backups)


def write_json_atomic(
    destination: Path,
    value: dict[str, Any],
    *,
    replace_existing: bool = True,
) -> None:
    validate_path_hierarchy(destination, "Run manifest")
    if destination.exists() and not replace_existing:
        raise FileExistsError(f"Run manifest already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    validate_path_hierarchy(destination, "Run manifest")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=".uma-bct-json-",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(value, temporary_file, indent=2, ensure_ascii=False)
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        validate_path_hierarchy(destination, "Run manifest")
        if destination.exists() and not replace_existing:
            raise FileExistsError(f"Run manifest already exists: {destination}")
        temporary_path.replace(destination)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
