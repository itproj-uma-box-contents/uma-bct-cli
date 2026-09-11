"""Create run folders and safely write run files.

Each run gets a private folder and an exclusive catalogue lock. Forced runs
keep the old files, and failed publication can restore the prior catalogue.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .atomic import open_stable_file, rename_new_file
from .path_safety import check_path

LOCK_FILE_NAME = ".uma-bct.lock"


def regular_file_sha256(path: Path, purpose: str) -> str:
    """Hash an unlinked regular file while checking its identity."""

    digest = hashlib.sha256()
    with open_stable_file(path, purpose) as (source, _metadata):
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def same_regular_file_content(first: Path, second: Path, purpose: str) -> bool:
    try:
        return regular_file_sha256(first, purpose) == regular_file_sha256(second, purpose)
    except (OSError, ValueError):
        return False


def path_entry_exists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    except OSError:
        return True
    return True


def replace_backed_file(staging: Path, destination: Path, backup: Path, purpose: str) -> None:
    """Replace a backed-up file without overwriting a racing unbacked edit."""

    check_path(staging, f"Staged {purpose}")
    check_path(destination, purpose)
    check_path(backup, f"Backed-up {purpose}")
    new_digest = regular_file_sha256(staging, f"Staged {purpose}")
    quarantine = destination.with_name(f".uma-bct-rollback-{uuid.uuid4().hex}.tmp")
    check_path(quarantine, f"{purpose} publication quarantine")
    try:
        os.replace(destination, quarantine)
        if not same_regular_file_content(quarantine, backup, f"{purpose} publication source"):
            rename_new_file(
                quarantine,
                destination,
                existing_message=f"{purpose} changed again while its unbacked edit was restored: {destination}",
            )
            quarantine = None
            raise RuntimeError(f"{purpose} changed after its backup and was not overwritten: {destination}")
        rename_new_file(staging, destination, existing_message=f"{purpose} appeared during publication: {destination}")
        quarantine.unlink()
        quarantine = None
    except BaseException as original_error:
        if quarantine is not None and path_entry_exists(quarantine):
            try:
                if not path_entry_exists(destination):
                    rename_new_file(
                        quarantine,
                        destination,
                        existing_message=f"{purpose} appeared during rollback: {destination}",
                    )
                    quarantine = None
                elif regular_file_sha256(destination, purpose) == new_digest:
                    os.replace(quarantine, destination)
                    quarantine = None
                elif same_regular_file_content(quarantine, backup, f"Backed-up {purpose}"):
                    quarantine.unlink()
                    quarantine = None
            except (OSError, ValueError):
                pass
        if quarantine is not None and path_entry_exists(quarantine):
            raise RuntimeError(
                f"{purpose} changed during publication; recovery data was retained at {quarantine}."
            ) from original_error
        raise


def restore_backup_no_clobber(backup: Path, destination: Path, purpose: str) -> None:
    temporary_path: Path | None = None
    try:
        with open_stable_file(backup, f"Backed-up {purpose}") as (source, _metadata):
            with tempfile.NamedTemporaryFile(
                mode="w+b", dir=destination.parent, prefix=".uma-bct-rollback-", suffix=".tmp", delete=False
            ) as temporary:
                temporary_path = Path(temporary.name)
                shutil.copyfileobj(source, temporary)
                temporary.flush()
                os.fsync(temporary.fileno())
        rename_new_file(
            temporary_path,
            destination,
            existing_message=f"{purpose} appeared during rollback: {destination}",
        )
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def rollback_owned_file(destination: Path, expected_digest: str, backup: Path | None, purpose: str) -> None:
    """Atomically displace and verify an owned publication before rollback."""

    quarantine: Path | None = destination.with_name(f".uma-bct-rollback-{uuid.uuid4().hex}.tmp")
    try:
        os.replace(destination, quarantine)
        if regular_file_sha256(quarantine, f"Published {purpose}") != expected_digest:
            rename_new_file(
                quarantine,
                destination,
                existing_message=f"{purpose} changed again while it was restored: {destination}",
            )
            quarantine = None
            raise RuntimeError(f"{purpose} changed after publication and was not rolled back: {destination}")
        if backup is not None:
            restore_backup_no_clobber(backup, destination, purpose)
        quarantine.unlink()
        quarantine = None
    except BaseException as original_error:
        if quarantine is not None and path_entry_exists(quarantine):
            try:
                quarantine_digest = regular_file_sha256(quarantine, f"Rollback {purpose}")
                if quarantine_digest != expected_digest and not path_entry_exists(destination):
                    rename_new_file(
                        quarantine,
                        destination,
                        existing_message=f"{purpose} appeared while an unknown edit was restored: {destination}",
                    )
                    quarantine = None
                elif quarantine_digest == expected_digest:
                    if backup is not None and not path_entry_exists(destination):
                        restore_backup_no_clobber(backup, destination, purpose)
                    quarantine.unlink()
                    quarantine = None
            except (OSError, ValueError):
                pass
        if quarantine is not None and path_entry_exists(quarantine):
            raise RuntimeError(f"{purpose} rollback data was retained at {quarantine}.") from original_error
        raise


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
        except BaseException:
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
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(f"Existing result must be one unlinked regular file: {path}")
        backup = previous_result_path(path, run_id)
        check_path(backup, "Previous result")
        temporary_path: Path | None = None
        try:
            with open_stable_file(path, "Existing result") as (source, before):
                with tempfile.NamedTemporaryFile(
                    mode="w+b", dir=path.parent, prefix=".uma-bct-backup-", suffix=".tmp", delete=False
                ) as temporary_file:
                    temporary_path = Path(temporary_file.name)
                    shutil.copyfileobj(source, temporary_file)
                    temporary_file.flush()
                    os.fsync(temporary_file.fileno())
            temporary_path.chmod(stat.S_IMODE(before.st_mode))
            os.utime(temporary_path, ns=(before.st_atime_ns, before.st_mtime_ns))
            rename_new_file(temporary_path, backup, existing_message=f"Previous result already exists: {backup}")
            temporary_path = None
            backups.append(backup)
        except BaseException:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise
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
        with temporary_path.open("r+b") as restored:
            os.fsync(restored.fileno())
        check_path(destination, "Catalogue rollback")
        temporary_path.replace(destination)
    except BaseException:
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
            mode="w", encoding="utf-8", newline="\n", dir=destination.parent,
            prefix=".uma-bct-json-", suffix=".tmp", delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(value, temporary_file, indent=2, ensure_ascii=False)
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        check_path(destination, "Run manifest")
        if destination.exists() and not replace_existing:
            raise FileExistsError(f"Run manifest already exists: {destination}")
        if replace_existing:
            temporary_path.replace(destination)
        else:
            rename_new_file(temporary_path, destination, existing_message=f"Run manifest already exists: {destination}")
            temporary_path = None
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
