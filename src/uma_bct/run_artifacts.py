"""Create run folders and safely write run files.

Each run gets a private folder and an exclusive catalogue lock. Forced runs
keep the old files, and failed publication can restore the prior catalogue.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .path_safety import check_path

LOCK_FILE_NAME = ".uma-bct.lock"


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


def catalogue_lock_path(output_path: Path) -> Path:
    return output_path.with_name(LOCK_FILE_NAME)


class CatalogueLock:
    """Reserve one catalogue and remove only this instance's lock file."""

    def __init__(self, output_path: Path) -> None:
        self.path = catalogue_lock_path(output_path)
        self._descriptor: int | None = None
        self._identity: tuple[int, int] | None = None

    def _owner_text(self) -> str:
        descriptor: int | None = None
        try:
            check_path(self.path, "Catalogue lock")
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(self.path, flags)
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                return "PID unavailable"
            value = json.loads(os.read(descriptor, 256).decode())
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
            return "PID unavailable"
        finally:
            if descriptor is not None:
                os.close(descriptor)
        pid = value.get("pid") if isinstance(value, dict) else None
        return f"PID {pid}" if isinstance(pid, int) and pid > 0 else "PID unavailable"

    def acquire(self) -> None:
        check_path(self.path, "Catalogue lock")
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        check_path(self.path, "Catalogue lock")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0)
        try:
            descriptor = os.open(self.path, flags, 0o600)
        except FileExistsError:
            owner = self._owner_text()
            raise RuntimeError(
                f"Another UMA BCT run reserved this catalogue ({owner}). If no run is active, "
                f"remove {self.path} and try again."
            ) from None
        metadata = os.fstat(descriptor)
        self._descriptor = descriptor
        self._identity = (metadata.st_dev, metadata.st_ino)
        try:
            content = (json.dumps({"pid": os.getpid()}) + "\n").encode()
            written = 0
            while written < len(content):
                count = os.write(descriptor, content[written:])
                if count == 0:
                    raise OSError("Could not write the catalogue lock.")
                written += count
            os.fsync(descriptor)
        except Exception:
            self.release()
            raise
        if os.name == "nt":
            os.close(descriptor)
            self._descriptor = None

    def release(self) -> None:
        descriptor = self._descriptor
        identity = self._identity
        self._descriptor = None
        self._identity = None
        if identity is None:
            return

        try:
            metadata = self.path.lstat()
        except FileNotFoundError:
            if descriptor is not None:
                os.close(descriptor)
            return
        except OSError:
            if descriptor is not None:
                os.close(descriptor)
            raise
        if (metadata.st_dev, metadata.st_ino) != identity:
            if descriptor is not None:
                os.close(descriptor)
            return

        quarantine = self.path.with_name(f".uma-bct-lock-release-{uuid.uuid4().hex}.tmp")
        try:
            os.replace(self.path, quarantine)
        except FileNotFoundError:
            pass
        else:
            metadata = quarantine.lstat()
            if (metadata.st_dev, metadata.st_ino) == identity:
                quarantine.unlink()
            else:
                try:
                    os.link(quarantine, self.path, follow_symlinks=False)
                except OSError as error:
                    raise RuntimeError(
                        f"The catalogue lock changed during cleanup; the replacement was retained at {quarantine}."
                    ) from error
                quarantine.unlink()
        finally:
            if descriptor is not None:
                os.close(descriptor)


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


def rollback_catalogue(destination: Path, previous_catalogue: Path | None) -> None:
    """Restore the old CSV atomically, or remove a newly published CSV."""

    check_path(destination, "Catalogue rollback")
    if previous_catalogue is None:
        destination.unlink(missing_ok=True)
        return

    check_path(previous_catalogue, "Previous catalogue")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=".uma-bct-rollback-", suffix=".csv", delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        shutil.copy2(previous_catalogue, temporary_path)
        with temporary_path.open("rb") as restored:
            os.fsync(restored.fileno())
        check_path(destination, "Catalogue rollback")
        temporary_path.replace(destination)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


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
