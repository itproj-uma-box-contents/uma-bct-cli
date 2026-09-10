"""Create run folders and safely write run files.

Each run gets a private folder. Forced runs keep the old files, and manifests
are replaced only after the new copy is fully written.
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

from .path_safety import check_path


def new_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


def run_directory(root: Path) -> Path:
    check_path(root, "Review-crop root")
    destination = root / f"run-{new_run_id()}"
    destination.mkdir(mode=0o700, parents=True)
    check_path(destination, "Review-crop run directory")
    return destination


def manifest_path(output_path: Path) -> Path:
    return output_path.with_suffix(".run.json")


def failed_manifest_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}.failed-{new_run_id()}.run.json")


def previous_result_path(path: Path, run_id: str) -> Path:
    if path.name.endswith(".run.json"):
        stem = path.name[: -len(".run.json")]
        return path.with_name(f"{stem}.previous-{run_id}.run.json")
    return path.with_name(f"{path.stem}.previous-{run_id}{path.suffix}")


def backup_existing_results(paths: tuple[Path, ...]) -> tuple[Path, ...]:
    """Keep dated copies before a forced replacement."""

    run_id = new_run_id()
    backups: list[Path] = []
    for path in paths:
        check_path(path, "Existing result")
        if not path.exists():
            continue
        backup = previous_result_path(path, run_id)
        check_path(backup, "Previous result")
        shutil.copy2(path, backup)
        backups.append(backup)
    return tuple(backups)


def write_json_atomic(destination: Path, value: dict[str, Any], *, replace_existing: bool = True) -> None:
    check_path(destination, "Run manifest")
    if destination.exists() and not replace_existing:
        raise FileExistsError(f"Run manifest already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    check_path(destination, "Run manifest")
    temporary_path: Path | None = None
    # Write beside the manifest, then swap it in only when complete.
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent, prefix=".uma-bct-json-", suffix=".tmp", delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(value, temporary_file, indent=2, ensure_ascii=False)
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        check_path(destination, "Run manifest")
        if destination.exists() and not replace_existing:
            raise FileExistsError(f"Run manifest already exists: {destination}")
        temporary_path.replace(destination)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
