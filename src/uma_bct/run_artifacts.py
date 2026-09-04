"""Private run files, catalogue rollback, and output reservations."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

GENERATED_STEM_MAX_BYTES = 120


def bounded_generated_stem(value: str) -> str:
    """Bound generated sibling names while keeping long stems collision-safe."""

    original = value
    normalized = original.rstrip(" .") or "output"
    encoded = normalized.encode("utf-8")
    if normalized == original and len(encoded) <= GENERATED_STEM_MAX_BYTES:
        return normalized
    digest = hashlib.sha256(original.encode("utf-8")).hexdigest()[:10]
    prefix_budget = GENERATED_STEM_MAX_BYTES - len(digest) - 1
    prefix = encoded[:prefix_budget].decode("utf-8", errors="ignore").rstrip(" .")
    return f"{prefix or 'output'}-{digest}"


def run_directory(root: Path) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    destination = root / f"run-{timestamp}-{uuid.uuid4().hex[:8]}"
    destination.mkdir(mode=0o700, parents=True, exist_ok=False)
    destination.chmod(0o700)
    return destination


def manifest_path(output_path: Path) -> Path:
    return output_path.with_name(f"{bounded_generated_stem(output_path.stem)}.run.json")


def failed_manifest_path(output_path: Path) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return output_path.with_name(
        f"{bounded_generated_stem(output_path.stem)}.failed-"
        f"{timestamp}-{uuid.uuid4().hex[:8]}.run.json"
    )


def output_lock_path(output_path: Path) -> Path:
    # CSV and XLSX destinations with the same stem also share one manifest.
    stem = bounded_generated_stem(output_path.stem)
    return output_path.with_name(f".{stem}.uma-bct.lock")


def acquire_output_lock(output_path: Path) -> tuple[int, Path, tuple[int, int]]:
    """Reserve an output name for the duration of a batch."""

    lock_path = output_lock_path(output_path)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(lock_path, flags, 0o600)
    except FileExistsError as error:
        raise ValueError(
            f"another run is using this output (remove a stale lock only after "
            f"confirming no run is active): {lock_path}"
        ) from error
    except OSError as error:
        raise ValueError(f"could not reserve output {output_path}: {error}") from error
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode())
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        return descriptor, lock_path, (metadata.st_dev, metadata.st_ino)
    except OSError as error:
        os.close(descriptor)
        try:
            lock_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise ValueError(f"could not reserve output {output_path}: {error}") from error


def release_output_lock(
    descriptor: int, lock_path: Path, identity: tuple[int, int]
) -> None:
    """Release only the reservation created by this process."""

    os.close(descriptor)
    try:
        metadata = lock_path.stat(follow_symlinks=False)
        if (metadata.st_dev, metadata.st_ino) == identity:
            lock_path.unlink(missing_ok=True)
    except FileNotFoundError:
        pass
    except OSError as error:
        print(
            f"Warning: could not remove output lock {lock_path}: {error}",
            file=sys.stderr,
        )


def write_json_atomic(destination: Path, value: dict[str, Any]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
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
        temporary_path.replace(destination)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def backup_existing_catalogue(destination: Path) -> Path | None:
    """Create a private same-filesystem backup for catalogue/manifest rollback."""

    if not destination.exists():
        return None
    backup_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=destination.parent,
            prefix=".uma-bct-catalogue-backup-",
            suffix=".tmp",
            delete=False,
        ) as backup_file:
            backup_path = Path(backup_file.name)
            with destination.open("rb") as current_catalogue:
                shutil.copyfileobj(current_catalogue, backup_file)
            backup_file.flush()
            os.fsync(backup_file.fileno())
        backup_path.chmod(0o600)
        return backup_path
    except Exception:
        if backup_path is not None:
            backup_path.unlink(missing_ok=True)
        raise


def restore_catalogue(destination: Path, backup_path: Path | None) -> None:
    """Remove a new catalogue or atomically restore the previous one."""

    if backup_path is None:
        destination.unlink(missing_ok=True)
    else:
        backup_path.replace(destination)
