"""Coordinate resumable, human-readable transcription batches.

The ordinary :func:`uma_bct.batch_runner.run_batch` path deliberately remains
small and backwards compatible.  This module owns the stronger invariants
needed by the client launchers: input snapshots, resumable checkpoints,
readable crop names, and recoverable final publication.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import stat
import sys
import unicodedata
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

from .adapters.hosted_client import PROVIDER_API_BASES
from .adapters.ollama_client import timeout_from_environment
from .atomic import open_stable_file, rename_new_file
from .bootstrap import DATE_DETECTORS, build_pipeline, selected_date_detector
from .constants import (
    DATE_OUTPUT_COLUMNS,
    IMAGE_EXTENSIONS,
    LEGACY_OUTPUT_COLUMNS,
    OUTPUT_COLUMNS,
    REVIEW_DATE_OUTPUT_COLUMNS,
)
from .domain import BoundingBox, Box, BoxResult, BoxTranscription, InputImage
from .export import export_catalogue
from .model_config import selected_model_metadata, selected_model_name
from .path_safety import check_path, redirected_path_kind
from .pipeline import BoxTranscriptionPipeline, ImageResult, warning_text
from .run_artifacts import LOCK_FILE_NAME, write_json_atomic
from .terminal import TerminalProgress, print_run_summary
from .util.date_util import deserialize_dates
from .verification import choose_verification, configured_mode, verify_image

PROGRESS_FILE_NAME = "Run Progress.json"
DETAILS_FILE_NAME = "Run Details.json"
CATALOGUE_FILE_NAME = "Catalogue.csv"
BOX_IMAGES_DIRECTORY_NAME = "Box Images"
WORKING_DIRECTORY_NAME = ".working"
PENDING_CATALOGUE_NAME = ".uma-bct-catalogue-pending.csv"
PENDING_CATALOGUE_NAMES = frozenset({PENDING_CATALOGUE_NAME, ".uma-bct-catalogue-pending.xlsx"})
MANAGED_SCHEMA_VERSION = 1
TERMINAL_IMAGE_STATES = frozenset({"complete", "partial", "failed"})
ACTIVE_WINDOWS_GUARDS: set[str] = set()
DESKTOP_METADATA = frozenset({".DS_Store", "Thumbs.db", "desktop.ini"})
WINDOWS_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL", *(f"COM{number}" for number in range(1, 10)), *(f"LPT{number}" for number in range(1, 10))}
)


class ManagedBatchError(RuntimeError):
    """A safe, operator-actionable managed-batch error."""


class ManagedBatchSelectionError(ManagedBatchError):
    """An error tied to one unfinished batch selected from the results root."""

    def __init__(self, batch_directory: Path, message: str) -> None:
        super().__init__(message)
        self.batch_directory = batch_directory


def catalogue_file_name(value: dict[str, Any]) -> str:
    # Checkpoints without a format predate Excel support and must stay CSV.
    file_type = value.get("file_type", "csv")
    if file_type not in {"excel", "csv"}:
        raise ManagedBatchError("Saved catalogue format must be excel or csv.")
    return "Catalogue.xlsx" if file_type == "excel" else CATALOGUE_FILE_NAME


def pending_catalogue_name(value: dict[str, Any]) -> str:
    return ".uma-bct-catalogue-pending" + Path(catalogue_file_name(value)).suffix


class ManagedInterruption(KeyboardInterrupt):
    """An operating-system interruption with the intended process exit code."""

    def __init__(self, exit_code: int) -> None:
        super().__init__(exit_code)
        self.exit_code = exit_code


def lock_descriptor(descriptor: int) -> None:
    if os.name == "nt":
        import msvcrt

        if os.fstat(descriptor).st_size == 0:
            os.write(descriptor, b"\0")
            os.fsync(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        try:
            msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
        except OSError as error:
            raise BlockingIOError from error
    else:
        import fcntl

        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)


def unlock_descriptor(descriptor: int) -> None:
    if os.name == "nt":
        import msvcrt

        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(descriptor, fcntl.LOCK_UN)


def application_version() -> str:
    try:
        return version("uma-bct-cli")
    except PackageNotFoundError:
        return "0.1.0"


def canonical_lock_name(path: Path) -> str:
    physical = os.path.realpath(path)
    if sys.platform == "darwin":
        return unicodedata.normalize("NFC", physical).casefold()
    return os.path.normcase(physical)


def fingerprint_file(path: Path, purpose: str) -> dict[str, int | str]:
    """Hash one unlinked regular file while proving its metadata stayed stable."""

    try:
        digest = hashlib.sha256()
        byte_count = 0
        with open_stable_file(path, purpose, error_type=ManagedBatchError) as (source, metadata):
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
                byte_count += len(chunk)
    except ManagedBatchError:
        raise
    except OSError as error:
        raise ManagedBatchError(f"Could not read {purpose} {path}: {error}") from error
    return {"size": byte_count, "sha256": digest.hexdigest()}


def input_snapshot(images: list[Path], input_path: Path) -> list[dict[str, Any]]:
    """Remember photo order and bytes so a resumed run uses the same evidence."""

    root_is_directory = input_path.is_dir()
    snapshot: list[dict[str, Any]] = []
    for image in images:
        relative_name = image.relative_to(input_path).as_posix() if root_is_directory else image.name
        fingerprint = fingerprint_file(image, "Input image")
        snapshot.append(
            {
                "relative_name": relative_name,
                "original_filename": image.name,
                "source_path": str(image),
                **fingerprint,
            }
        )
    return snapshot


def rediscover_input_snapshot(input_path: Path, *, names: set[str] | None = None) -> list[dict[str, Any]]:
    """Re-list supported inputs so additions and late changes cannot publish."""

    check_path(input_path, "Input path")
    try:
        metadata = input_path.lstat()
    except OSError as error:
        raise ManagedBatchError(f"Could not recheck input path {input_path}: {error}") from error
    if stat.S_ISREG(metadata.st_mode):
        images = [input_path]
    elif stat.S_ISDIR(metadata.st_mode):
        try:
            candidates = tuple(input_path.iterdir())
        except OSError as error:
            raise ManagedBatchError(f"Could not re-list input directory {input_path}: {error}") from error
        images = []
        for candidate in candidates:
            if candidate.suffix.casefold() not in IMAGE_EXTENSIONS or (names is not None and candidate.name not in names):
                continue
            check_path(candidate, "Input image")
            candidate_metadata = candidate.lstat()
            if stat.S_ISREG(candidate_metadata.st_mode):
                images.append(candidate)
        images.sort(key=lambda item: (item.name.casefold(), item.name))
    else:
        raise ManagedBatchError(f"Input path is no longer a file or directory: {input_path}")
    return input_snapshot(images, input_path)


def snapshot_signature(snapshot: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"relative_name": item["relative_name"], "size": item["size"], "sha256": item["sha256"]}
        for item in snapshot
    ]


def describe_snapshot_change(expected: list[dict[str, Any]], actual: list[dict[str, Any]]) -> str:
    expected_by_name = {item["relative_name"]: item for item in expected}
    actual_by_name = {item["relative_name"]: item for item in actual}
    removed = [name for name in expected_by_name if name not in actual_by_name]
    added = [name for name in actual_by_name if name not in expected_by_name]
    changed = [
        name
        for name in expected_by_name.keys() & actual_by_name.keys()
        if expected_by_name[name]["size"] != actual_by_name[name]["size"]
        or expected_by_name[name]["sha256"] != actual_by_name[name]["sha256"]
    ]
    expected_order = [item["relative_name"] for item in expected]
    actual_order = [item["relative_name"] for item in actual]
    details: list[str] = []
    if removed:
        details.append(f"removed or renamed: {', '.join(map(repr, removed[:3]))}")
    if added:
        details.append(f"added or renamed: {', '.join(map(repr, added[:3]))}")
    if changed:
        details.append(f"byte-changed: {', '.join(map(repr, changed[:3]))}")
    if not (removed or added or changed) and expected_order != actual_order:
        details.append("the input order changed")
    return "; ".join(details) or "the input snapshot changed"


def semantic_configuration(arguments: argparse.Namespace, request_timeout: float | None) -> dict[str, Any]:
    configuration = {
        "detector": arguments.detector,
        "transcriber": arguments.transcriber,
        "model": selected_model_name(arguments),
        "date_detector": selected_date_detector(arguments),
        "minimum_transcription_confidence": arguments.min_transcription_confidence,
        "minimum_detector_confidence": arguments.min_detector_confidence,
        "detect_missing_boxes": getattr(arguments, "detect_missing_boxes", True),
        "request_timeout_seconds": request_timeout,
    }
    if getattr(arguments, "hosted_model", None) is not None:
        configuration["backend"] = "hosted"
    return configuration


def truncate_utf8(value: str, maximum_bytes: int) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= maximum_bytes:
        return value
    encoded = encoded[:maximum_bytes]
    while encoded:
        try:
            return encoded.decode("utf-8").rstrip(" .")
        except UnicodeDecodeError:
            encoded = encoded[:-1]
    return ""


def sanitize_source_name(filename: str) -> str:
    """Return a readable component valid on supported desktop platforms."""

    suffix_at = filename.rfind(".")
    raw_stem = filename[:suffix_at] if suffix_at > 0 else filename
    stem = unicodedata.normalize("NFC", raw_stem)
    cleaned = "".join(
        "-" if ord(character) < 32 or character in '<>:"/\\|?*#%' else character for character in stem
    )
    cleaned = " ".join(cleaned.split()).strip(" .")
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    cleaned = cleaned or "Photo"
    if cleaned.split(".", 1)[0].upper() in WINDOWS_RESERVED_NAMES:
        cleaned = f"Photo {cleaned}"
    return truncate_utf8(cleaned, 150) or "Photo"


def allocate_source_names(snapshot: list[dict[str, Any]]) -> None:
    used: set[str] = set()
    # Names can collide after Windows-safe cleanup or case folding. Number
    # them before processing so published crop links remain stable on resume.
    for item in snapshot:
        base = sanitize_source_name(item["original_filename"])
        candidate = base
        suffix = 2
        while unicodedata.normalize("NFC", candidate).casefold() in used:
            marker = f" ({suffix})"
            candidate = f"{truncate_utf8(base, 150 - len(marker))}{marker}"
            suffix += 1
        used.add(unicodedata.normalize("NFC", candidate).casefold())
        item["readable_source_name"] = candidate


def batch_name(moment: datetime | None = None) -> str:
    return (moment or datetime.now().astimezone()).strftime("Batch %Y-%m-%d %H-%M-%S")


def create_batch_directory(results_root: Path, moment: datetime | None = None) -> Path:
    base = batch_name(moment)
    suffix = 1
    while True:
        name = base if suffix == 1 else f"{base} ({suffix})"
        destination = results_root / name
        check_path(destination, "Batch directory")
        try:
            destination.mkdir(mode=0o700)
        except FileExistsError:
            suffix += 1
            continue
        try:
            (destination / BOX_IMAGES_DIRECTORY_NAME).mkdir(mode=0o700)
        except BaseException:
            destination.rmdir()
            raise
        return destination


def read_json_file(path: Path, purpose: str) -> dict[str, Any]:
    try:
        with open_stable_file(path, purpose, error_type=ManagedBatchError) as (source, metadata):
            if metadata.st_size > 32 * 1024 * 1024:
                raise ManagedBatchError(f"{purpose} is unexpectedly large: {path}")
            content = source.read(32 * 1024 * 1024 + 1)
            if len(content) > 32 * 1024 * 1024:
                raise ManagedBatchError(f"{purpose} is unexpectedly large: {path}")
        value = json.loads(content.decode("utf-8"))
    except ManagedBatchError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ManagedBatchError(f"Could not read {purpose} {path}: {warning_text(error)}") from error
    if not isinstance(value, dict):
        raise ManagedBatchError(f"{purpose} must contain a JSON object: {path}")
    return value


class ResultsRootLock:
    """Hold an OS-backed lock for one results root.

    The small lock file intentionally persists.  A crashed owner therefore
    leaves audit information, while the operating-system lock proves whether
    a later process may recover it.
    """

    def __init__(self, results_root: Path) -> None:
        self.results_root = results_root
        self.path = results_root / LOCK_FILE_NAME
        self._descriptor: int | None = None
        self._identity: tuple[int, int] | None = None
        self._guard_descriptor: int | None = None
        self._guard_identity: tuple[Path, int, int] | None = None
        self._guard_name: str | None = None

    def verify_lock(self) -> None:
        identity = self._identity
        if identity is None:
            return
        try:
            current = self.path.lstat()
        except OSError as error:
            raise ManagedBatchError(f"Results lock changed while it was held: {self.path}: {error}") from error
        if (
            redirected_path_kind(self.path, current)
            or not stat.S_ISREG(current.st_mode)
            or current.st_nlink != 1
            or identity != (current.st_dev, current.st_ino)
        ):
            raise ManagedBatchError(f"Results lock changed while it was held: {self.path}")

    def acquire_guard(self) -> None:
        # The sibling guard serializes recovery even if someone replaces the
        # results directory or its visible lock while a run is active.
        self.results_root.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        check_path(self.results_root.parent, "Results parent directory")
        normalized = canonical_lock_name(self.results_root)
        guard_name = f".uma-bct-lock-release-results-{hashlib.sha256(normalized.encode()).hexdigest()}.tmp"
        guard_path = self.results_root.parent / guard_name
        check_path(guard_path, "Results lock guard")
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(guard_path, flags, 0o600)
        try:
            metadata = os.fstat(descriptor)
            current = guard_path.lstat()
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or (metadata.st_dev, metadata.st_ino) != (current.st_dev, current.st_ino)
            ):
                raise ManagedBatchError(f"Results lock guard must be one unlinked regular file: {guard_path}")
            if os.name == "nt":
                if normalized in ACTIVE_WINDOWS_GUARDS:
                    raise BlockingIOError
            lock_descriptor(descriptor)
            if os.name == "nt":
                ACTIVE_WINDOWS_GUARDS.add(normalized)
        except BaseException:
            os.close(descriptor)
            raise
        self._guard_descriptor = descriptor
        self._guard_identity = (guard_path, metadata.st_dev, metadata.st_ino)
        self._guard_name = normalized
        self.results_root.mkdir(mode=0o700, exist_ok=True)

    def release_guard(self) -> None:
        descriptor = self._guard_descriptor
        identity = self._guard_identity
        name = self._guard_name
        self._guard_descriptor = None
        self._guard_identity = None
        self._guard_name = None
        if descriptor is None:
            return
        error: OSError | None = None
        if identity is not None:
            path, device, inode = identity
            try:
                metadata = path.lstat()
                if (metadata.st_dev, metadata.st_ino) != (device, inode):
                    raise OSError(f"Results lock guard changed while held: {path}")
            except OSError as caught:
                error = caught
        try:
            unlock_descriptor(descriptor)
        except OSError as caught:
            error = error or caught
        finally:
            os.close(descriptor)
            if name is not None:
                ACTIVE_WINDOWS_GUARDS.discard(name)
        if error is not None:
            raise ManagedBatchError(str(error)) from error

    def acquire(self) -> None:
        try:
            self.acquire_guard()
            self.write_lock()
        except BlockingIOError:
            self.release_guard()
            raise ManagedBatchError(
                f"Another UMA BCT run is already using this Results directory: {self.results_root}"
            ) from None
        except BaseException:
            self.release_guard()
            raise

    def write_lock(self) -> None:
        check_path(self.results_root, "Results directory")
        self.results_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        check_path(self.results_root, "Results directory")
        root_metadata = self.results_root.lstat()
        if not stat.S_ISDIR(root_metadata.st_mode):
            raise ManagedBatchError(f"Results directory must be a directory: {self.results_root}")

        # An old lock file is kept as an audit record. Reuse it only after the
        # OS lock is held and its inode and contents prove it is still ours.
        check_path(self.path, "Results lock")
        flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor: int | None = None
        created = False
        try:
            try:
                descriptor = os.open(self.path, flags | os.O_CREAT | os.O_EXCL, 0o600)
                created = True
            except FileExistsError:
                descriptor = os.open(self.path, flags)
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise ManagedBatchError(f"Results lock must be one unlinked regular file: {self.path}")
            current = self.path.lstat()
            if (metadata.st_dev, metadata.st_ino) != (current.st_dev, current.st_ino):
                raise ManagedBatchError(f"Results lock changed while it was opened: {self.path}")
            lock_descriptor(descriptor)
            current = self.path.lstat()
            if (metadata.st_dev, metadata.st_ino) != (current.st_dev, current.st_ino):
                raise ManagedBatchError(f"Results lock changed while it was locked: {self.path}")
            if not created:
                os.lseek(descriptor, 0, os.SEEK_SET)
                raw = os.read(descriptor, 4096)
                if re.fullmatch(rb'\{"schema_version": 1, "pid": [1-9][0-9]*\}\n', raw) is None:
                    raise ManagedBatchError(
                        f"Results lock is malformed and was not recovered automatically: {self.path}"
                    )
            content = (json.dumps({"schema_version": 1, "pid": os.getpid()}) + "\n").encode()
            with os.fdopen(descriptor, "r+b", closefd=False) as target:
                target.seek(0)
                target.write(content)
                target.truncate()
                target.flush()
                os.fsync(descriptor)
            current = self.path.lstat()
            if (metadata.st_dev, metadata.st_ino) != (current.st_dev, current.st_ino):
                raise ManagedBatchError(f"Results lock changed while it was written: {self.path}")
            self._descriptor = descriptor
            self._identity = (metadata.st_dev, metadata.st_ino)
            descriptor = None
        except BlockingIOError:
            raise ManagedBatchError(
                f"Another UMA BCT run is already using this Results directory: {self.results_root}"
            ) from None
        except OSError as error:
            raise ManagedBatchError(f"Could not reserve Results directory {self.results_root}: {error}") from error
        finally:
            if descriptor is not None:
                try:
                    unlock_descriptor(descriptor)
                except OSError:
                    pass
                os.close(descriptor)

    def release(self) -> None:
        descriptor = self._descriptor
        self._descriptor = None
        try:
            self.verify_lock()
            if descriptor is not None:
                os.lseek(descriptor, 0, os.SEEK_SET)
                expected = (json.dumps({"schema_version": 1, "pid": os.getpid()}) + "\n").encode()
                if os.read(descriptor, len(expected) + 1) != expected:
                    raise ManagedBatchError(f"Results lock contents changed while it was held: {self.path}")
        finally:
            self._identity = None
            try:
                if descriptor is not None:
                    try:
                        unlock_descriptor(descriptor)
                    finally:
                        os.close(descriptor)
            finally:
                self.release_guard()


def managed_batches(results_root: Path) -> tuple[list[tuple[Path, dict[str, Any]]], list[tuple[Path, dict[str, Any]]]]:
    unfinished: list[tuple[Path, dict[str, Any]]] = []
    completed: list[tuple[Path, dict[str, Any]]] = []
    try:
        children = tuple(results_root.iterdir())
    except OSError as error:
        raise ManagedBatchError(f"Could not inspect Results directory {results_root}: {error}") from error
    for child in children:
        if not child.name.startswith("Batch "):
            continue
        child = child.absolute()
        check_path(child, "Batch directory")
        metadata = child.lstat()
        if not stat.S_ISDIR(metadata.st_mode):
            raise ManagedBatchError(f"Managed batch entry must be a directory: {child}")
        progress_path = child / PROGRESS_FILE_NAME
        details_path = child / DETAILS_FILE_NAME
        if path_entry_exists(progress_path) and path_entry_exists(details_path):
            raise ManagedBatchError(f"Batch has contradictory progress and details files: {child}")
        if path_entry_exists(progress_path):
            unfinished.append((child, read_json_file(progress_path, "Run progress")))
        elif path_entry_exists(details_path):
            completed.append((child, read_json_file(details_path, "Run details")))
    if len(unfinished) > 1:
        names = ", ".join(str(path) for path, value in unfinished)
        raise ManagedBatchError(f"More than one unfinished batch exists; move all but the intended batch elsewhere: {names}")
    return unfinished, completed


def initial_progress(arguments: argparse.Namespace, batch_directory: Path,
    snapshot: list[dict[str, Any]], request_timeout: float | None) -> dict[str, Any]:
    inputs = []
    for item in snapshot:
        inputs.append({**item, "state": "pending", "attempts": 0})
    model = selected_model_metadata(arguments)
    if hosted := getattr(arguments, "hosted_model", None):
        model.update(backend="hosted", provider=hosted.provider, api_base=PROVIDER_API_BASES[hosted.provider])
    return {
        "schema_version": MANAGED_SCHEMA_VERSION,
        "application": {"name": "uma-bct-cli", "version": application_version()},
        "phase": "processing",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "batch_directory": str(batch_directory),
        "configuration": semantic_configuration(arguments, request_timeout),
        "file_type": getattr(arguments, "file_type", None) or "csv",
        "input_directory": str(arguments.path_to_inputs),
        "input_snapshot": snapshot_signature(snapshot),
        "inputs": inputs,
        "model": model,
        "endpoint_history": [],
        "resume_count": 0,
        "run_failures": [],
        "publication": {"phase": "not_started"},
        "verification": {"mode": configured_mode(arguments), "decision": None},
    }


def validate_progress_shape(progress: dict[str, Any], batch_directory: Path) -> None:
    if progress.get("schema_version") != MANAGED_SCHEMA_VERSION:
        raise ManagedBatchError(f"Unsupported Run Progress schema in {batch_directory}")
    catalogue_file_name(progress)
    stored_batch = progress.get("batch_directory")
    if not isinstance(stored_batch, str) or canonical_lock_name(Path(stored_batch)) != canonical_lock_name(batch_directory):
        raise ManagedBatchError(f"Run Progress names a different batch directory: {batch_directory}")
    inputs = progress.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise ManagedBatchError(f"Run Progress has no valid input list: {batch_directory}")
    for item in inputs:
        if not isinstance(item, dict) or item.get("state") not in {*TERMINAL_IMAGE_STATES, "pending", "active"}:
            raise ManagedBatchError(f"Run Progress contains an invalid image state: {batch_directory}")
    phase = progress.get("phase")
    if phase not in {"processing", "interrupted", "staging", "publishing", "finalizing", "final",
                     "awaiting_verification", "verifying"}:
        raise ManagedBatchError(f"Run Progress contains an invalid publication phase: {batch_directory}")


def same_snapshot(expected: list[dict[str, Any]], actual: list[dict[str, Any]]) -> bool:
    return expected == snapshot_signature(actual)


def batch_snapshot(expected: list[dict[str, Any]], actual: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select a batch's original photos, preserving order and deferring additions."""

    by_name = {item["relative_name"]: item for item in actual}
    return [by_name[item["relative_name"]] for item in expected if item["relative_name"] in by_name]


def completed_batch_order_key(item: tuple[Path, dict[str, Any]]) -> tuple[str, str, int]:
    match = re.fullmatch(r"(Batch \d{4}-\d{2}-\d{2} \d{2}-\d{2}-\d{2})(?: \((\d+)\))?", item[0].name)
    base = match.group(1) if match else item[0].name
    collision = int(match.group(2) or 1) if match else 1
    return str(item[1].get("created_at", "")), base, collision


def latest_completed(completed: list[tuple[Path, dict[str, Any]]]) -> tuple[Path, dict[str, Any]] | None:
    """Keep the existing helper available to callers inspecting batch history."""

    return max(completed, key=completed_batch_order_key, default=None)


def input_identity(item: dict[str, Any]) -> tuple[str, int, str]:
    return item["relative_name"], item["size"], item["sha256"]


def select_batch(arguments: argparse.Namespace, results_root: Path,
    snapshot: list[dict[str, Any]], request_timeout: float | None,
    ) -> tuple[Path | None, dict[str, Any] | None, bool, list[tuple[Path, dict[str, Any]]] | None]:

    discovered_names = [item["relative_name"] for item in snapshot]
    unfinished, completed = managed_batches(results_root)
    # An unfinished batch has priority. Its inputs, options, and application
    # version must still match before any saved photo can be reused.
    if unfinished:
        batch_directory, progress = unfinished[0]
        validate_progress_shape(progress, batch_directory)
        for name in PENDING_CATALOGUE_NAMES:
            if name != pending_catalogue_name(progress) and path_entry_exists(batch_directory / name):
                raise ManagedBatchSelectionError(
                    batch_directory, f"Unexpected pending catalogue format: {batch_directory / name}",
                )
        requested_format = getattr(arguments, "file_type", None)
        if requested_format is not None and requested_format != progress.get("file_type", "csv"):
            raise ManagedBatchSelectionError(batch_directory, "Cannot change catalogue format in an unfinished batch.")
        expected_snapshot = progress.get("input_snapshot")
        original_snapshot = batch_snapshot(expected_snapshot, snapshot) if isinstance(expected_snapshot, list) else []
        if not isinstance(expected_snapshot, list) or not same_snapshot(expected_snapshot, original_snapshot):
            details = describe_snapshot_change(expected_snapshot if isinstance(expected_snapshot, list) else [], snapshot)
            raise ManagedBatchSelectionError(
                batch_directory,
                f"Cannot resume {batch_directory}: {details}. Restore its original photo set or move the whole "
                "unfinished batch elsewhere before starting another.",
            )
        requested_verification = getattr(arguments, "verification", None)
        if requested_verification and requested_verification != progress.get("verification", {}).get("mode", "never"):
            raise ManagedBatchSelectionError(batch_directory, "Cannot change verification mode in an unfinished batch.")
        stored_configuration = progress.get("configuration")
        if isinstance(stored_configuration, dict):
            legacy_date_configuration = "date_detector" not in stored_configuration
            # Older managed runs always generated missing-box suggestions.
            saved_detectors = {item["result"].get("date_detector", "none")
                               for item in progress["inputs"] if item.get("result")}
            if "date_detector" not in stored_configuration and len(saved_detectors) > 1:
                raise ManagedBatchSelectionError(batch_directory, "Saved date extraction options are inconsistent.")
            stored_configuration = {"detect_missing_boxes": True,
                                    "date_detector": next(iter(saved_detectors), "none"), **stored_configuration}
            # Preserve extraction-free legacy checkpoints and offline publication.
            # Current batches always require the same explicit AI switch.
            if (stored_configuration["date_detector"] == "none"
                    and not getattr(arguments, "ai_dates", False)
                    and (legacy_date_configuration
                         or all(item["state"] in TERMINAL_IMAGE_STATES for item in progress["inputs"]))):
                arguments._resume_date_detector = "none"
        current_configuration = semantic_configuration(arguments, request_timeout)
        if stored_configuration != current_configuration:
            raise ManagedBatchSelectionError(
                batch_directory,
                f"Cannot resume {batch_directory}: the processing configuration changed. Restore the original options.",
            )
        if progress.get("application") != {"name": "uma-bct-cli", "version": application_version()}:
            raise ManagedBatchSelectionError(
                batch_directory,
                f"Cannot resume {batch_directory}: the UMA BCT application version changed.",
            )
        for saved, current in zip(progress["inputs"], original_snapshot, strict=True):
            if saved.get("source_path") != current["source_path"]:
                saved.setdefault("original_source_path", saved.get("source_path"))
                saved["source_path"] = current["source_path"]
        progress["configuration"] = stored_configuration
        progress["input_directory"] = str(arguments.path_to_inputs)
        return batch_directory, progress, True, None

    # An identical completed set is reported as already done unless the
    # operator explicitly asked for a separate new batch.
    if not arguments.reprocess:
        # Filenames remain part of identity: identical photographs supplied
        # under separate names must never silently collapse into one input.
        processed: set[tuple[str, int, str]] = set()
        current_keys = {input_identity(item) for item in snapshot}
        matching_batches = []
        for path, details in sorted(completed, key=completed_batch_order_key, reverse=True):
            saved_configuration = details.get("configuration", {})
            if (saved_configuration.get("model") != selected_model_name(arguments)
                    or saved_configuration.get("backend", "ollama")
                    != ("hosted" if getattr(arguments, "hosted_model", None) is not None else "ollama")):
                continue
            source_root = details.get("input_directory")
            if not isinstance(source_root, str) or canonical_lock_name(Path(source_root)) != canonical_lock_name(
                arguments.path_to_inputs
            ):
                continue
            recorded = {input_identity(item) for item in details.get("input_snapshot", [])}
            if recorded & (current_keys - processed):
                validate_sources_and_crops(details, path, details["input_snapshot"])
                validate_completed_catalogue(details, path)
                matching_batches.append((path, details))
                processed.update(recorded)
        remaining = [item for item in snapshot if input_identity(item) not in processed]
        if not remaining and matching_batches:
            return None, None, False, matching_batches
        snapshot = remaining

    batch_directory = create_batch_directory(results_root).absolute()
    allocate_source_names(snapshot)
    progress = initial_progress(arguments, batch_directory, snapshot, request_timeout)
    progress["observed_input_names"] = discovered_names
    try:
        write_json_atomic(batch_directory / PROGRESS_FILE_NAME, progress, replace_existing=False)
    except BaseException:
        try:
            (batch_directory / BOX_IMAGES_DIRECTORY_NAME).rmdir()
            batch_directory.rmdir()
        except OSError:
            pass
        raise
    return batch_directory, progress, False, None


def safe_relative_batch_path(batch_directory: Path, value: str, purpose: str) -> Path:
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts or "\\" in value or ":" in value:
        raise ManagedBatchError(f"{purpose} has an unsafe path in {batch_directory}: {value!r}")
    return batch_directory.joinpath(*relative.parts)


def crop_evidence_path(batch_directory: Path, evidence: dict[str, Any]) -> Path:
    relative = evidence.get("relative_path")
    if not isinstance(relative, str):
        raise ManagedBatchError(f"Crop evidence has no valid relative path in {batch_directory}")
    pure = PurePosixPath(relative)
    if len(pure.parts) != 2 or pure.parts[0] != BOX_IMAGES_DIRECTORY_NAME:
        raise ManagedBatchError(f"Crop evidence is outside Box Images in {batch_directory}: {relative!r}")
    return safe_relative_batch_path(batch_directory, relative, "Crop evidence")


def validate_crop_evidence(batch_directory: Path, evidence: dict[str, Any]) -> None:
    path = crop_evidence_path(batch_directory, evidence)
    actual = fingerprint_file(path, "Crop evidence")
    if actual.get("size") == 0:
        raise ManagedBatchError(f"Crop evidence is empty: {path}")
    if actual != {"size": evidence.get("size"), "sha256": evidence.get("sha256")}:
        raise ManagedBatchError(f"Crop evidence changed after it was committed: {path}")


def active_crop_digests(evidence: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    crop = {"size": evidence.get("size"), "sha256": evidence.get("sha256")}
    reservation = {
        "size": evidence.get("reservation_size"),
        "sha256": evidence.get("reservation_sha256"),
    }
    return crop, reservation


def open_cleanup_descriptor(path: Path) -> int:
    if os.name != "nt":
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        return os.open(path, flags)

    # Windows needs a handle that permits a concurrent rename or deletion so
    # cleanup can still compare the opened file after its path is removed.
    import ctypes
    import msvcrt

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = (
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
    )
    kernel.CreateFileW.restype = ctypes.c_void_p
    kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
    kernel.CloseHandle.restype = ctypes.c_bool
    generic_read = 0x80000000
    share_read_write_delete = 0x00000007
    open_existing = 3
    normal_attributes = 0x00000080
    handle = kernel.CreateFileW(
        str(path), generic_read, share_read_write_delete, None, open_existing, normal_attributes, None
    )
    if handle in {None, ctypes.c_void_p(-1).value}:
        raise OSError(ctypes.get_last_error(), f"Could not open cleanup file: {path}")
    try:
        return msvcrt.open_osfhandle(int(handle), os.O_RDONLY | getattr(os, "O_BINARY", 0))
    except BaseException:
        kernel.CloseHandle(handle)
        raise


def dispose_or_restore_quarantine(
    quarantine: Path, target: Path, allowed: tuple[dict[str, Any], ...], purpose: str
) -> None:
    """Remove only recorded bytes, restoring a file changed during cleanup."""

    check_path(target, f"{purpose} restoration target")
    fingerprint_file(quarantine, purpose)
    descriptor: int | None = None
    try:
        descriptor = open_cleanup_descriptor(quarantine)
        opened = os.fstat(descriptor)
        current = quarantine.lstat()
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)
        ):
            raise ManagedBatchError(f"{purpose} changed before cleanup: {quarantine}")
        content = b"".join(iter(lambda: os.read(descriptor, 1024 * 1024), b""))
        actual = {"size": len(content), "sha256": hashlib.sha256(content).hexdigest()}
        if actual not in allowed:
            raise ManagedBatchError(f"Unknown {purpose.lower()} content was retained: {quarantine}")
        current = quarantine.lstat()
        if (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino):
            raise ManagedBatchError(f"{purpose} changed before cleanup: {quarantine}")
        # The open descriptor survives unlink. Read it again to catch a write
        # racing the deletion and restore those new bytes for human recovery.
        quarantine.unlink()
        os.lseek(descriptor, 0, os.SEEK_SET)
        after_content = b"".join(iter(lambda: os.read(descriptor, 1024 * 1024), b""))
        if after_content != content:
            restore_unlinked_crop_content(quarantine, target, after_content, purpose)
            raise ManagedBatchError(f"{purpose} changed during cleanup and was restored: {target}")
    except ManagedBatchError:
        if path_entry_exists(quarantine) and not path_entry_exists(target):
            rename_new_file(
                quarantine,
                target,
                existing_message=f"{purpose} target appeared while an unknown file was restored: {target}",
            )
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)


def restore_unlinked_crop_content(quarantine: Path, target: Path, content: bytes, purpose: str) -> None:
    check_path(quarantine, f"{purpose} recovery file")
    check_path(target, f"{purpose} restoration target")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(quarantine, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as restored:
            restored.write(content)
            restored.flush()
            os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if not path_entry_exists(target):
        rename_new_file(
            quarantine,
            target,
            existing_message=f"{purpose} target appeared while an unknown file was restored: {target}",
        )
        return
    raise ManagedBatchError(f"Changed {purpose.lower()} content was retained for recovery: {quarantine}")


def quarantine_active_file(
    path: Path,
    quarantine: Path,
    target: Path,
    allowed: tuple[dict[str, Any], ...],
    purpose: str,
) -> None:
    # Validate every destination before moving or disposing of evidence.
    # An existing quarantine can take the restoration branch immediately.
    for candidate in (path, quarantine, target):
        check_path(candidate, purpose)
    if path_entry_exists(quarantine):
        dispose_or_restore_quarantine(quarantine, target, allowed, purpose)
    if not path_entry_exists(path):
        return
    rename_new_file(path, quarantine, existing_message=f"{purpose} quarantine already exists: {quarantine}")
    dispose_or_restore_quarantine(quarantine, target, allowed, purpose)


def remove_interrupted_crop(batch_directory: Path, path: Path, evidence: dict[str, Any]) -> None:
    """Remove only a target proven to come from this interrupted publication."""

    crop, reservation = active_crop_digests(evidence)
    state = evidence.get("publication_state")
    if state == "unreserved":
        allowed = (reservation,)
    elif state == "reserved":
        allowed = (reservation, crop)
    elif state == "published":
        allowed = (crop,)
    else:
        allowed = ()
    cleanup_value = evidence.get("cleanup_path")
    if not isinstance(cleanup_value, str):
        raise ManagedBatchError(f"Interrupted crop has no cleanup path: {path}")
    if not PurePosixPath(cleanup_value).parts or PurePosixPath(cleanup_value).parts[0] != WORKING_DIRECTORY_NAME:
        raise ManagedBatchError(f"Interrupted crop cleanup is outside the working directory: {cleanup_value!r}")
    cleanup = safe_relative_batch_path(batch_directory, cleanup_value, "Interrupted crop cleanup")
    quarantine_active_file(path, cleanup, path, allowed, "Interrupted crop publication")


def recover_displaced_crop_target(batch_directory: Path, target: Path, evidence: dict[str, Any]) -> None:
    relative = evidence.get("displaced_path")
    if not isinstance(relative, str):
        return
    pure = PurePosixPath(relative)
    if not pure.parts or pure.parts[0] != WORKING_DIRECTORY_NAME:
        raise ManagedBatchError(f"Displaced crop reservation is outside the working directory: {relative!r}")
    displaced = safe_relative_batch_path(batch_directory, relative, "Displaced crop reservation")
    cleanup_value = evidence.get("displaced_cleanup_path")
    if not isinstance(cleanup_value, str):
        raise ManagedBatchError(f"Displaced crop reservation has no cleanup path: {relative!r}")
    if not PurePosixPath(cleanup_value).parts or PurePosixPath(cleanup_value).parts[0] != WORKING_DIRECTORY_NAME:
        raise ManagedBatchError(f"Displaced crop cleanup is outside the working directory: {cleanup_value!r}")
    cleanup = safe_relative_batch_path(batch_directory, cleanup_value, "Displaced crop cleanup")
    crop, reservation = active_crop_digests(evidence)
    quarantine_active_file(displaced, cleanup, target, (reservation,), "Displaced crop reservation")


def remove_working_directory(batch_directory: Path, relative_value: str | None) -> None:
    if not relative_value:
        return
    working = safe_relative_batch_path(batch_directory, relative_value, "Working directory")
    relative = PurePosixPath(relative_value)
    if len(relative.parts) != 2 or relative.parts[0] != WORKING_DIRECTORY_NAME:
        raise ManagedBatchError(f"Working directory is outside the managed work root: {relative_value!r}")
    # The leaf can be a real directory while .working or an ancestor redirects
    # outside this batch. Recheck the whole path before recursive deletion.
    check_path(working, "Managed working directory")
    try:
        metadata = working.lstat()
    except FileNotFoundError:
        return
    if redirected_path_kind(working, metadata) or not stat.S_ISDIR(metadata.st_mode):
        raise ManagedBatchError(f"Managed working path is not a plain directory: {working}")
    shutil.rmtree(working)


def validate_batch_entries(batch_directory: Path) -> None:
    allowed = {
        PROGRESS_FILE_NAME,
        DETAILS_FILE_NAME,
        BOX_IMAGES_DIRECTORY_NAME,
        WORKING_DIRECTORY_NAME,
        *PENDING_CATALOGUE_NAMES,
        CATALOGUE_FILE_NAME,
        "Catalogue.xlsx",
    }
    # Reviewers may save corrected catalogues or notes beside our outputs.
    # Only application-owned paths participate in recovery and publication.
    unexpected = [
        path for path in batch_directory.iterdir()
        if path.name not in allowed and not harmless_extra_file(path)
    ]
    if unexpected:
        raise ManagedBatchError(f"Unexpected file in managed batch directory: {unexpected[0]}")


def harmless_extra_file(path: Path, *, desktop_only: bool = False) -> bool:
    if desktop_only and path.name not in DESKTOP_METADATA:
        return False
    metadata = path.lstat()
    return stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1 and not redirected_path_kind(path, metadata)


def prepare_resume(progress: dict[str, Any], batch_directory: Path) -> int:
    """Keep proven photo commits and reset only the photo active at interruption."""

    # Active crop recovery runs before working-directory cleanup. Reject an
    # already redirected layout before either path can mutate saved evidence.
    for directory in (batch_directory, batch_directory / WORKING_DIRECTORY_NAME,
                      batch_directory / BOX_IMAGES_DIRECTORY_NAME):
        check_path(directory, "Managed recovery directory")
    effective_phase = (
        progress.get("interrupted_phase") if progress.get("phase") == "interrupted" else progress.get("phase")
    )
    preserve_pending = effective_phase in {"staging", "publishing", "finalizing", "final"}
    for path in batch_directory.iterdir():
        generated_temporary = path.name.startswith(".uma-bct-json-") and path.name.endswith(".tmp")
        generated_review = path.name.startswith(".uma-bct-review-") and path.name.endswith(".tmp")
        generated_catalogue = path.name.startswith(".uma-bct-catalogue-") and path.name not in PENDING_CATALOGUE_NAMES
        if path.name in PENDING_CATALOGUE_NAMES and not preserve_pending:
            raise ManagedBatchError(f"Unexpected pending catalogue in managed batch: {path}")
        if not (generated_temporary or generated_catalogue or generated_review):
            continue
        metadata = path.lstat()
        if redirected_path_kind(path, metadata) or not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ManagedBatchError(f"Generated temporary path is not one unlinked regular file: {path}")
        path.unlink()

    validate_batch_entries(batch_directory)

    # A terminal photo is trusted only with matching crop hashes and result
    # paths. Its work directory can then be removed without touching evidence.
    progress.pop("html_review", None)
    progress.get("publication", {}).pop("review", None)
    progress.pop("review_output", None)
    reused = 0
    committed_paths: set[Path] = set()
    for item in progress["inputs"]:
        state = item["state"]
        if state in TERMINAL_IMAGE_STATES:
            item_paths: set[Path] = set()
            for evidence in item.get("crop_evidence", []):
                validate_crop_evidence(batch_directory, evidence)
                evidence_path = crop_evidence_path(batch_directory, evidence)
                item_paths.add(evidence_path)
                committed_paths.add(evidence_path)
            result_value = item.get("result")
            if isinstance(result_value, dict):
                result = deserialize_result(result_value, batch_directory)
                if result.status != state:
                    raise ManagedBatchError(f"Committed photo status disagrees with its result in {batch_directory}")
                if {record.box.image_file_path for record in result.records} != item_paths:
                    raise ManagedBatchError(f"Committed result and crop evidence disagree in {batch_directory}")
            elif state != "failed" or not isinstance(item.get("error"), str):
                raise ManagedBatchError(f"Committed photo result is missing in {batch_directory}")
            remove_working_directory(batch_directory, item.get("working_directory"))
            item.pop("working_directory", None)
            reused += 1
            continue
        # A crop move may have stopped halfway through. Restore or remove only
        # files named by this attempt's recorded evidence, then retry the photo.
        if state == "active":
            for evidence in item.get("crop_evidence", []):
                path = crop_evidence_path(batch_directory, evidence)
                recover_displaced_crop_target(batch_directory, path, evidence)
                remove_interrupted_crop(batch_directory, path, evidence)
            remove_working_directory(batch_directory, item.get("working_directory"))
            item["state"] = "pending"
            item.pop("result", None)
            item.pop("error", None)
            item.pop("crop_evidence", None)
            item.pop("working_directory", None)
    box_images = batch_directory / BOX_IMAGES_DIRECTORY_NAME
    check_path(box_images, "Box Images directory")
    if not stat.S_ISDIR(box_images.lstat().st_mode):
        raise ManagedBatchError(f"Box Images path is not a directory: {box_images}")
    for path in box_images.iterdir():
        if path not in committed_paths and not harmless_extra_file(path, desktop_only=True):
            raise ManagedBatchError(f"Unexpected file in managed Box Images directory: {path}")
    return reused


def endpoint_identity(host: str) -> dict[str, str]:
    endpoint = urlsplit(host)
    hostname = endpoint.hostname or ""
    loopback = hostname == "::1" or hostname.startswith("127.")
    origin_host = f"[{hostname}]" if ":" in hostname else hostname
    port = endpoint.port
    default_port = 443 if endpoint.scheme == "https" else 80
    origin = f"{endpoint.scheme}://{origin_host}:{port or default_port}"
    return {"class": "numeric_loopback" if loopback else "remote", "origin": origin}


def validate_model_resume(progress: dict[str, Any], current: dict[str, str], batch_directory: Path) -> None:
    previous = progress.get("model", {})
    if current.get("backend", "ollama") != previous.get("backend", "ollama"):
        raise ManagedBatchError(f"Cannot resume {batch_directory}: the model backend changed.")
    if current.get("model") != previous.get("model"):
        raise ManagedBatchError(f"Cannot resume {batch_directory}: the preflight model identity changed.")
    if current.get("backend") == "hosted":
        if (
            previous.get("provider") != current.get("provider")
            or previous.get("api_base") != current.get("api_base")
        ):
            raise ManagedBatchError(f"Cannot resume {batch_directory}: the hosted model endpoint changed.")
        return
    previous_digest = previous.get("model_digest")
    current_digest = current.get("model_digest")
    if previous_digest and previous_digest != current_digest:
        raise ManagedBatchError(f"Cannot resume {batch_directory}: the model digest changed.")
    previous_host = previous.get("ollama_host")
    current_host = current.get("ollama_host")
    if previous_host and not current_host:
        raise ManagedBatchError(f"Cannot resume {batch_directory}: the Ollama endpoint identity is missing.")
    # A launcher gets a fresh loopback port each start, but switching between
    # local and remote inference or changing remote origin changes privacy.
    if previous_host and current_host:
        previous_endpoint = endpoint_identity(previous_host)
        current_endpoint = endpoint_identity(current_host)
        if previous_endpoint["class"] != current_endpoint["class"]:
            raise ManagedBatchError(f"Cannot resume {batch_directory}: the Ollama endpoint class changed.")
        if previous_endpoint["class"] != "numeric_loopback" and previous_endpoint["origin"] != current_endpoint["origin"]:
            raise ManagedBatchError(f"Cannot resume {batch_directory}: the remote Ollama origin changed.")


def build_managed_pipeline(
    arguments: argparse.Namespace, batch_directory: Path, request_timeout: float | None
) -> tuple[Any, dict[str, str]]:
    working_root = batch_directory / WORKING_DIRECTORY_NAME
    check_path(working_root, "Managed work root")
    working_root.mkdir(mode=0o700, exist_ok=True)
    pipeline = build_pipeline(
        detector_name=arguments.detector,
        transcriber_name=arguments.transcriber,
        date_detector_name=selected_date_detector(arguments),
        model_name=selected_model_name(arguments),
        work_directory=working_root,
        minimum_detector_confidence=arguments.min_detector_confidence,
        request_timeout_seconds=request_timeout,
        hosted_model=getattr(arguments, "hosted_model", None),
    )
    metadata = dict(pipeline.model_metadata)
    required = ("model", "backend", "provider", "api_base") if getattr(arguments, "hosted_model", None) else (
        "model", "model_digest", "ollama_host"
    )
    for field in required:
        if not isinstance(metadata.get(field), str) or not metadata[field]:
            raise ManagedBatchError(f"Model preflight did not return {field}.")
    return pipeline, metadata


def serialize_result(result: ImageResult, batch_directory: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for record in result.records:
        try:
            crop_relative = record.box.image_file_path.relative_to(batch_directory).as_posix()
        except ValueError:
            raise ManagedBatchError(f"A committed crop is outside its batch: {record.box.image_file_path}") from None
        records.append(
            {
                "box": {
                    "box_id": record.box.box_id,
                    "bbox": list(record.box.bounding_box.as_xywh()),
                    "image": crop_relative,
                    "position": record.box.position,
                    "detector_confidence": record.box.detector_confidence,
                },
                "transcription": asdict(record.transcription),
                **({"verification": record.verification} if record.verification is not None else {}),
            }
        )
    return {
        "input_image": {
            "file_path": str(result.input_image.file_path),
            "width": result.input_image.width,
            "height": result.input_image.height,
        },
        "requested_detector": result.requested_detector,
        "actual_detector": result.actual_detector,
        "fallback_path": list(result.fallback_path),
        "transcriber": result.transcriber,
        "date_detector": result.date_detector,
        "status": result.status,
        "records": records,
        "warnings": list(result.warnings),
        "model_metadata": dict(result.model_metadata),
    }


def deserialize_result(value: dict[str, Any], batch_directory: Path) -> ImageResult:
    image_value = value["input_image"]
    image = InputImage(Path(image_value["file_path"]), int(image_value["width"]), int(image_value["height"]))
    records: list[BoxResult] = []
    for item in value["records"]:
        box_value = item["box"]
        x, y, width, height = (int(number) for number in box_value["bbox"])
        crop_path = safe_relative_batch_path(batch_directory, box_value["image"], "Committed crop")
        box = Box(
            box_id=box_value["box_id"],
            source_image=image,
            bounding_box=BoundingBox((x, y), (x + width, y + height)),
            image_file_path=crop_path,
            position=box_value["position"],
            detector_confidence=float(box_value["detector_confidence"]),
        )
        transcription_value = item["transcription"]

        transcription = BoxTranscription(
            **{
                **transcription_value,
                "confidence": int(transcription_value["confidence"]),
                "record_identifiers": tuple(transcription_value["record_identifiers"]),
                "typed_lines": tuple(transcription_value["typed_lines"]),
                "handwritten_lines": tuple(transcription_value["handwritten_lines"]),
                "dates": deserialize_dates(transcription_value.get("dates", [])),
                "auxiliary_annotations": tuple(transcription_value["auxiliary_annotations"]),
                "needs_review": bool(transcription_value["needs_review"]),
            }
        )
        records.append(BoxResult(box, transcription, item.get("verification")))
    return ImageResult(
        input_image=image,
        requested_detector=value["requested_detector"],
        actual_detector=value["actual_detector"],
        fallback_path=tuple(value["fallback_path"]),
        transcriber=value["transcriber"],
        date_detector=value.get("date_detector", "none"),
        status=value["status"],
        records=tuple(records),
        warnings=tuple(value.get("warnings", [])),
        model_metadata=dict(value.get("model_metadata", {})),
    )


def is_full_photo(record: BoxResult) -> bool:
    return record.box.image_file_path.name == "full_image.png" or record.box.box_id.endswith(":full-image")


def within_directory(path: Path, directory: Path) -> bool:
    try:
        return os.path.commonpath((str(path.absolute()), str(directory.absolute()))) == str(directory.absolute())
    except ValueError:
        return False


def prepare_crop_publication(
    result: ImageResult, item: dict[str, Any], batch_directory: Path, working_directory: Path
) -> tuple[ImageResult, list[dict[str, Any]], list[tuple[Path, Path, dict[str, Any]]]]:
    published_by_source: dict[Path, Path] = {}
    evidence_by_target: dict[Path, dict[str, Any]] = {}
    moves: list[tuple[Path, Path, dict[str, Any]]] = []
    new_records: list[BoxResult] = []
    box_number = 0
    full_number = 0
    source_label = item["readable_source_name"]
    box_images = batch_directory / BOX_IMAGES_DIRECTORY_NAME

    # Several records can share one full-photo crop. Publish that source once
    # and point each record at the same verified review image.
    for record in result.records:
        source = record.box.image_file_path
        if source not in published_by_source:
            check_path(source, "Generated crop")
            if not within_directory(source, working_directory):
                raise ManagedBatchError(f"Generated crop is outside the active photo work directory: {source}")
            crop_fingerprint = fingerprint_file(source, "Generated crop")
            if crop_fingerprint["size"] == 0:
                raise ManagedBatchError(f"Generated crop is empty: {source}")
            if is_full_photo(record):
                full_number += 1
                suffix = "Full Photo" if full_number == 1 else f"Full Photo ({full_number})"
            else:
                box_number += 1
                suffix = f"Box {box_number:02d}"
            target = box_images / f"{source_label} - {suffix}.png"
            check_path(target, "Readable crop")
            if path_entry_exists(target):
                raise ManagedBatchError(f"Readable crop already exists and was not overwritten: {target}")
            published_by_source[source] = target
            reservation = f"UMA BCT crop reservation {uuid.uuid4().hex}\n"
            displaced = source.with_name(f".uma-bct-rollback-crop-reservation-{uuid.uuid4().hex}.tmp")
            cleanup = source.with_name(f".uma-bct-rollback-crop-cleanup-{uuid.uuid4().hex}.tmp")
            displaced_cleanup = source.with_name(f".uma-bct-rollback-crop-cleanup-{uuid.uuid4().hex}.tmp")
            evidence = {
                "relative_path": target.relative_to(batch_directory).as_posix(),
                **crop_fingerprint,
                "publication_state": "unreserved",
                "reservation": reservation,
                "reservation_size": len(reservation.encode()),
                "reservation_sha256": hashlib.sha256(reservation.encode()).hexdigest(),
                "displaced_path": displaced.relative_to(batch_directory).as_posix(),
                "cleanup_path": cleanup.relative_to(batch_directory).as_posix(),
                "displaced_cleanup_path": displaced_cleanup.relative_to(batch_directory).as_posix(),
            }
            evidence_by_target[target] = evidence
            moves.append((source, target, evidence))
        target = published_by_source[source]
        new_records.append(BoxResult(replace(record.box, image_file_path=target), record.transcription))

    updated = replace(result, records=tuple(new_records))
    return updated, list(evidence_by_target.values()), moves


def reserve_crop_target(source: Path, target: Path, evidence: dict[str, Any]) -> None:
    """Create a durable no-clobber reservation for one readable crop name."""

    content = evidence["reservation"].encode()
    reservation_path: Path | None = source.with_name(f".uma-bct-rollback-crop-reservation-{uuid.uuid4().hex}.tmp")
    check_path(reservation_path, "Crop-name reservation temporary file")
    try:
        with reservation_path.open("xb") as reservation_file:
            reservation_file.write(content)
            reservation_file.flush()
            os.fsync(reservation_file.fileno())
        rename_new_file(
            reservation_path,
            target,
            existing_message=f"Readable crop already exists and was not overwritten: {target}",
        )
        reservation_path = None
    finally:
        if reservation_path is not None:
            reservation_path.unlink(missing_ok=True)
    actual = fingerprint_file(target, "Crop-name reservation")
    if actual != {"size": evidence["reservation_size"], "sha256": evidence["reservation_sha256"]}:
        raise ManagedBatchError(f"Crop-name reservation changed before publication: {target}")


def publish_reserved_crop(source: Path, target: Path, evidence: dict[str, Any]) -> None:
    displaced = safe_relative_batch_path(target.parent.parent, evidence["displaced_path"], "Displaced reservation")
    check_path(displaced, "Displaced crop-name reservation")
    expected_reservation = {
        "size": evidence["reservation_size"],
        "sha256": evidence["reservation_sha256"],
    }
    rename_new_file(
        target,
        displaced,
        existing_message=f"Displaced crop-name reservation already exists: {displaced}",
    )
    reservation = fingerprint_file(displaced, "Crop-name reservation")
    if reservation != expected_reservation:
        rename_new_file(
            displaced,
            target,
            existing_message=f"Readable crop changed again while its unknown edit was restored: {target}",
        )
        raise ManagedBatchError(f"Crop-name reservation changed before publication: {target}")
    rename_new_file(source, target, existing_message=f"Readable crop appeared during publication: {target}")
    validate_crop_evidence(target.parent.parent, evidence)
    displaced.unlink()


def write_progress(batch_directory: Path, progress: dict[str, Any]) -> None:
    write_json_atomic(batch_directory / PROGRESS_FILE_NAME, progress)


def path_entry_exists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    except OSError:
        return True
    return True


def can_write_progress(batch_directory: Path) -> bool:
    """Never recreate progress after the authoritative details rename."""

    return not path_entry_exists(batch_directory / DETAILS_FILE_NAME)


def current_snapshot_entry(item: dict[str, Any]) -> dict[str, Any]:
    fingerprint = fingerprint_file(Path(item["source_path"]), "Input image")
    return {"relative_name": item["relative_name"], **fingerprint}


def expected_snapshot_entry(item: dict[str, Any]) -> dict[str, Any]:
    return {"relative_name": item["relative_name"], "size": item["size"], "sha256": item["sha256"]}


def verify_source_unchanged(item: dict[str, Any]) -> None:
    if current_snapshot_entry(item) != expected_snapshot_entry(item):
        raise ManagedBatchError(f"Input image changed during processing: {item['source_path']}")


def attempt_working_directory(batch_directory: Path, image_index: int, attempt: int) -> tuple[str, Path]:
    relative = f"{WORKING_DIRECTORY_NAME}/image-{image_index + 1:04d}-attempt-{attempt:03d}-{uuid.uuid4().hex[:8]}"
    return relative, batch_directory.joinpath(*PurePosixPath(relative).parts)


def cleanup_committed_work(progress: dict[str, Any], batch_directory: Path, item: dict[str, Any]) -> None:
    remove_working_directory(batch_directory, item.get("working_directory"))
    if "working_directory" in item:
        item.pop("working_directory", None)
        write_progress(batch_directory, progress)


def process_pending_images(
    arguments: argparse.Namespace,
    images: list[Path],
    progress: dict[str, Any],
    batch_directory: Path,
    pipeline: Any,
    reporter: TerminalProgress,
) -> int:
    """Checkpoint each photo before and after publishing its review crops."""

    processed = 0
    for index, (image_path, item) in enumerate(zip(images, progress["inputs"], strict=True)):
        if item["state"] in TERMINAL_IMAGE_STATES:
            continue
        attempt = int(item.get("attempts", 0)) + 1
        working_relative, working_directory = attempt_working_directory(batch_directory, index, attempt)
        item.update(state="active", attempts=attempt, working_directory=working_relative)
        item.pop("error", None)
        item.pop("result", None)
        item.pop("crop_evidence", None)
        progress["phase"] = "processing"
        write_progress(batch_directory, progress)
        check_path(working_directory, "Photo work directory")
        working_directory.mkdir(mode=0o700, parents=True)
        pipeline.work_directory = working_directory
        verify_source_unchanged(item)
        reporter.photo_started(index + 1, image_path)
        try:
            result = pipeline.process(image_path)
        except Exception as error:
            verify_source_unchanged(item)
            message = warning_text(error)
            item.update(state="failed", error=message)
            write_progress(batch_directory, progress)
            cleanup_committed_work(progress, batch_directory, item)
            print(f"Failed {ascii(str(image_path))}: {message}", file=sys.stderr)
            reporter.photo_finished(index + 1, "failed", 0)
            processed += 1
            continue

        verify_source_unchanged(item)
        # Record intended crop names and hashes before moving any file. On
        # restart, that evidence decides which interrupted moves are ours.
        updated, evidence, moves = prepare_crop_publication(result, item, batch_directory, working_directory)
        item["result"] = serialize_result(updated, batch_directory)
        item["crop_evidence"] = evidence
        write_progress(batch_directory, progress)
        for source, target, crop_evidence in moves:
            reserve_crop_target(source, target, crop_evidence)
        for crop_evidence in evidence:
            crop_evidence["publication_state"] = "reserved"
        write_progress(batch_directory, progress)
        for source, target, crop_evidence in moves:
            publish_reserved_crop(source, target, crop_evidence)
            crop_evidence["publication_state"] = "published"
            write_progress(batch_directory, progress)
        for crop in evidence:
            validate_crop_evidence(batch_directory, crop)
        # The photo becomes terminal only after every published crop matches
        # its recorded digest; earlier checkpoints remain recoverable work.
        item["state"] = updated.status
        write_progress(batch_directory, progress)
        item["crop_evidence"] = [
            {field: crop[field] for field in ("relative_path", "size", "sha256")} for crop in evidence
        ]
        write_progress(batch_directory, progress)
        cleanup_committed_work(progress, batch_directory, item)
        for warning in updated.warnings:
            print(f"Warning for {ascii(str(image_path))}: {warning_text(warning)}", file=sys.stderr)
        reporter.photo_finished(index + 1, updated.status, len(updated.records))
        processed += 1
    return processed


def all_results(progress: dict[str, Any], batch_directory: Path) -> list[ImageResult]:
    results: list[ImageResult] = []
    for item in progress["inputs"]:
        value = item.get("result")
        if isinstance(value, dict):
            results.append(deserialize_result(value, batch_directory))
    return results


def validate_sources_and_crops(progress: dict[str, Any], batch_directory: Path, snapshot: list[dict[str, Any]]) -> None:
    """Recheck every source and committed crop before exposing a catalogue."""

    validate_batch_entries(batch_directory)
    expected = progress.get("input_snapshot", [])
    if expected != snapshot_signature(batch_snapshot(expected, snapshot)):
        raise ManagedBatchError(f"Input photos changed before publication for {batch_directory}")
    expected_paths: set[Path] = set()
    for item in progress["inputs"]:
        item_paths: set[Path] = set()
        for evidence in item.get("crop_evidence", []):
            validate_crop_evidence(batch_directory, evidence)
            evidence_path = crop_evidence_path(batch_directory, evidence)
            item_paths.add(evidence_path)
            expected_paths.add(evidence_path)
        result_value = item.get("result")
        if isinstance(result_value, dict):
            result = deserialize_result(result_value, batch_directory)
            if result.status != item.get("state"):
                raise ManagedBatchError(f"Committed photo status disagrees with its result in {batch_directory}")
            record_paths = {record.box.image_file_path for record in result.records}
            if record_paths != item_paths:
                raise ManagedBatchError(f"Committed result and crop evidence disagree in {batch_directory}")
        elif item.get("state") != "failed" or not isinstance(item.get("error"), str):
            raise ManagedBatchError(f"Committed photo result is missing in {batch_directory}")

    box_images = batch_directory / BOX_IMAGES_DIRECTORY_NAME
    check_path(box_images, "Box Images directory")
    metadata = box_images.lstat()
    if not stat.S_ISDIR(metadata.st_mode):
        raise ManagedBatchError(f"Box Images path is not a directory: {box_images}")
    for path in box_images.iterdir():
        if path not in expected_paths and not harmless_extra_file(path, desktop_only=True):
            raise ManagedBatchError(f"Unexpected file in managed Box Images directory: {path}")


def validate_completed_catalogue(details: dict[str, Any], batch_directory: Path) -> None:
    """Prove a terminal catalogue still matches its recorded publication."""

    record_count = details.get("catalogue_records")
    publication = details.get("publication")
    if not isinstance(record_count, int) or record_count < 0 or not isinstance(publication, dict):
        raise ManagedBatchError(f"Run Details has invalid catalogue publication metadata: {batch_directory}")
    catalogue = batch_directory / catalogue_file_name(details)
    for name in PENDING_CATALOGUE_NAMES:
        if path_entry_exists(batch_directory / name):
            raise ManagedBatchError(f"Completed batch still contains a pending catalogue: {batch_directory / name}")
    if record_count == 0:
        if path_entry_exists(catalogue):
            raise ManagedBatchError(f"Zero-row completed batch unexpectedly contains a catalogue: {catalogue}")
        if publication.get("phase") != "not_published" or publication.get("record_count") != 0:
            raise ManagedBatchError(f"Zero-row publication metadata is invalid in {batch_directory}")
        return

    digest = publication.get("sha256")
    if (
        publication.get("phase") != "published"
        or publication.get("record_count") != record_count
        or not isinstance(digest, str)
    ):
        raise ManagedBatchError(f"Published catalogue metadata is invalid in {batch_directory}")
    actual = fingerprint_file(catalogue, "Published catalogue")
    if actual["size"] == 0:
        raise ManagedBatchError(f"Published catalogue is empty: {catalogue}")
    if actual["sha256"] != digest:
        details["catalogue_modified"] = True
        print(f"Catalogue changed after completion; preserved your file: {catalogue}", file=sys.stderr)


def file_digest(path: Path, purpose: str) -> str:
    return str(fingerprint_file(path, purpose)["sha256"])


def stage_catalogue(
    arguments: argparse.Namespace,
    progress: dict[str, Any],
    batch_directory: Path,
    results: list[ImageResult],
) -> tuple[Path, int, str]:
    """Save the catalogue and its digest before its public name appears."""

    pending = batch_directory / pending_catalogue_name(progress)
    for name in PENDING_CATALOGUE_NAMES:
        if name != pending.name and path_entry_exists(batch_directory / name):
            raise ManagedBatchError(f"Unexpected pending catalogue format: {batch_directory / name}")
    publication = progress.setdefault("publication", {})
    known_layouts = (OUTPUT_COLUMNS, LEGACY_OUTPUT_COLUMNS, DATE_OUTPUT_COLUMNS, REVIEW_DATE_OUTPUT_COLUMNS)
    stored_columns = publication.get("output_columns")
    if stored_columns is not None and stored_columns not in known_layouts:
        raise ManagedBatchError(f"Unsupported pending catalogue columns in {batch_directory}")
    columns = stored_columns or OUTPUT_COLUMNS
    publication.update(phase="staging", pending_catalogue=pending.name)
    progress["phase"] = "staging"
    write_progress(batch_directory, progress)

    # A restart may find a staged catalogue. Rebuild from committed records and
    # compare bytes rather than assuming the leftover file is trustworthy.
    if path_entry_exists(pending):
        check_path(pending, "Pending catalogue")
        verification = batch_directory / f".uma-bct-catalogue-verify-{uuid.uuid4().hex}{pending.suffix}"
        try:
            pending_digest = file_digest(pending, "Pending catalogue")
            for columns in ((stored_columns,) if stored_columns is not None else known_layouts):
                records = export_catalogue(
                    results, verification, minimum_confidence=arguments.min_transcription_confidence,
                    replace_existing=False, detect_missing_boxes=getattr(arguments, "detect_missing_boxes", True),
                    trusted_crop_links=True, output_columns=columns,
                )
                matches = file_digest(verification, "Expected pending catalogue") == pending_digest
                verification.unlink()
                if matches:
                    break
            else:
                raise ManagedBatchError(f"Pending catalogue does not match the committed results: {pending}")
        finally:
            try:
                verification.unlink(missing_ok=True)
            except BaseException:
                raise
    else:
        records = export_catalogue(
            results,
            pending,
            minimum_confidence=arguments.min_transcription_confidence,
            replace_existing=False,
            detect_missing_boxes=getattr(arguments, "detect_missing_boxes", True),
            trusted_crop_links=True,
            output_columns=columns,
        )
    publication["output_columns"] = list(columns)
    digest = file_digest(pending, "Pending catalogue")
    publication.update(phase="publishing", sha256=digest, record_count=len(records))
    progress["phase"] = "publishing"
    write_progress(batch_directory, progress)
    return pending, len(records), digest


def publish_recorded_catalogue(progress: dict[str, Any], batch_directory: Path) -> tuple[int, str]:
    """Complete or verify the exact catalogue publication saved in the checkpoint."""

    publication = progress.get("publication", {})
    pending_name = publication.get("pending_catalogue", publication.get("pending_csv"))
    digest = publication.get("sha256")
    record_count = publication.get("record_count")
    if not isinstance(pending_name, str) or not isinstance(digest, str) or not isinstance(record_count, int):
        raise ManagedBatchError(f"Publishing checkpoint is incomplete in {batch_directory}")
    pending = safe_relative_batch_path(batch_directory, pending_name, "Pending catalogue")
    if pending.parent != batch_directory or pending.name != pending_catalogue_name(progress):
        raise ManagedBatchError(f"Pending catalogue path is invalid in {batch_directory}")
    catalogue = batch_directory / catalogue_file_name(progress)
    if path_entry_exists(catalogue):
        if file_digest(catalogue, "Published catalogue") != digest:
            raise ManagedBatchError(f"Published catalogue does not match its checkpoint: {catalogue}")
        if path_entry_exists(pending):
            if file_digest(pending, "Pending catalogue") != digest:
                raise ManagedBatchError(f"Pending catalogue does not match its checkpoint: {pending}")
            pending.unlink()
    else:
        if not path_entry_exists(pending) or file_digest(pending, "Pending catalogue") != digest:
            raise ManagedBatchError(f"The recorded pending catalogue is missing or changed: {pending}")
        rename_new_file(pending, catalogue, existing_message=f"Catalogue already exists: {catalogue}")
    if file_digest(catalogue, "Published catalogue") != digest:
        raise ManagedBatchError(f"Published catalogue changed during publication: {catalogue}")
    return record_count, digest


def failure_count(progress: dict[str, Any]) -> int:
    failed_photos = sum(item.get("state") == "failed" for item in progress.get("inputs", []))
    stage_failures = progress.get("run_failures", [])
    return failed_photos + (len(stage_failures) if isinstance(stage_failures, list) else 0)


def committed_record_count(value: dict[str, Any]) -> int:
    return sum(
        len(item.get("result", {}).get("records", []))
        for item in value.get("inputs", [])
        if isinstance(item, dict)
        and item.get("state") in TERMINAL_IMAGE_STATES
        and isinstance(item.get("result"), dict)
    )


def final_status(progress: dict[str, Any], record_count: int) -> str:
    states = [item["state"] for item in progress["inputs"]]
    imperfect = any(state != "complete" for state in states) or bool(progress.get("run_failures"))
    if record_count == 0:
        return "failed"
    return "partial" if imperfect else "complete"


def details_payload(
    arguments: argparse.Namespace,
    progress: dict[str, Any],
    batch_directory: Path,
    record_count: int,
) -> dict[str, Any]:
    results = all_results(progress, batch_directory)
    status = final_status(progress, record_count)
    catalogue = batch_directory / catalogue_file_name(progress)
    failures: list[dict[str, str]] = list(progress.get("run_failures", []))
    for item in progress["inputs"]:
        if item["state"] == "failed":
            failures.append(
                {
                    "source_image": item["source_path"],
                    "error": item.get("error", "No usable catalogue records were produced for this photo."),
                }
            )
    states = [item["state"] for item in progress["inputs"]]
    return {
        "schema_version": MANAGED_SCHEMA_VERSION,
        "application": progress["application"],
        "phase": "final",
        "status": status,
        "created_at": progress["created_at"],
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "batch_directory": str(batch_directory),
        "configuration": progress["configuration"],
        "file_type": progress.get("file_type", "csv"),
        "verification": progress.get("verification", {"mode": "never", "decision": False}),
        "locations": {
            "input": progress["input_directory"],
            "output": str(catalogue) if record_count else None,
            "box_images": str(batch_directory / BOX_IMAGES_DIRECTORY_NAME),
        },
        "input_directory": progress["input_directory"],
        "input_snapshot": progress["input_snapshot"],
        "inputs": progress["inputs"],
        "model": progress["model"],
        "endpoint_history": progress.get("endpoint_history", []),
        "images_total": len(states),
        "images_complete": states.count("complete"),
        "images_partial": states.count("partial"),
        "images_failed": states.count("failed"),
        "images": [result.to_dict() for result in results],
        "failures": failures,
        "failure_count": len(failures),
        "catalogue_output": str(catalogue) if record_count else None,
        "catalogue_records": record_count,
        "run_manifest": str(batch_directory / DETAILS_FILE_NAME),
        "previous_results": [],
        "resume_count": progress["resume_count"],
        "run_failures": progress.get("run_failures", []),
        "publication": progress["publication"],
    }


def promote_details(batch_directory: Path, details: dict[str, Any]) -> None:
    progress_path = batch_directory / PROGRESS_FILE_NAME
    details_path = batch_directory / DETAILS_FILE_NAME
    if path_entry_exists(details_path):
        raise ManagedBatchError(f"Run Details already exists and was not overwritten: {details_path}")
    write_json_atomic(progress_path, details)
    rename_new_file(progress_path, details_path, existing_message=f"Run Details already exists: {details_path}")


def finalize_batch(arguments: argparse.Namespace, progress: dict[str, Any],
    batch_directory: Path) -> dict[str, Any]:

    # Publication begins only after every photo has a terminal outcome and
    # the original inputs plus their review crops still match the checkpoint.
    if any(item["state"] not in TERMINAL_IMAGE_STATES for item in progress["inputs"]):
        raise ManagedBatchError(f"Cannot publish {batch_directory} while photos remain unattempted.")
    input_path = Path(progress["input_directory"])
    originals = {item["relative_name"] for item in progress["input_snapshot"]}
    # Only hash original inputs. An added photo may still be copying, and
    # its bytes are deliberately outside this batch's publication boundary.
    current_snapshot = rediscover_input_snapshot(input_path, names=originals)
    known = set(progress.get("observed_input_names", originals))
    if input_path.is_dir():
        deferred = sum(path.name not in known and path.suffix.casefold() in IMAGE_EXTENSIONS
                       for path in input_path.iterdir())
        if deferred:
            print(f"{deferred} additional photo(s) left for the next launch.", file=sys.stderr)
    validate_sources_and_crops(progress, batch_directory, current_snapshot)
    results = all_results(progress, batch_directory)
    record_count = sum(len(result.records) for result in results)

    if record_count:
        if progress.get("phase") in {"publishing", "finalizing", "final"}:
            record_count, digest = publish_recorded_catalogue(progress, batch_directory)
        else:
            pending, record_count, digest = stage_catalogue(arguments, progress, batch_directory, results)
            record_count, digest = publish_recorded_catalogue(progress, batch_directory)
        progress["publication"].update(phase="published", catalogue=catalogue_file_name(progress))
        progress["catalogue_output"] = str(batch_directory / catalogue_file_name(progress))
        progress["catalogue_records"] = record_count
    else:
        for name in PENDING_CATALOGUE_NAMES:
            if path_entry_exists(batch_directory / name):
                raise ManagedBatchError(f"A zero-row batch has an unexpected pending catalogue: {batch_directory / name}")
        for name in (CATALOGUE_FILE_NAME, "Catalogue.xlsx"):
            if path_entry_exists(batch_directory / name):
                raise ManagedBatchError(f"A zero-row batch has an unexpected catalogue: {batch_directory / name}")
        progress["publication"] = {"phase": "not_published", "record_count": 0}
        progress["catalogue_output"] = None
        progress["catalogue_records"] = 0

    # Renaming progress to details is the final authority change. Until then
    # the run remains resumable with its publication phase recorded on disk.
    progress["phase"] = "finalizing"
    write_progress(batch_directory, progress)
    details = details_payload(arguments, progress, batch_directory, record_count)
    working_root = batch_directory / WORKING_DIRECTORY_NAME
    try:
        working_root.rmdir()
    except FileNotFoundError:
        pass
    except OSError as error:
        raise ManagedBatchError(f"Managed work directory could not be cleaned: {warning_text(error)}") from error
    promote_details(batch_directory, details)
    return details


def verify_managed_results(arguments: argparse.Namespace, progress: dict[str, Any], batch_directory: Path,
                           pipeline: Any, reporter: TerminalProgress, request_timeout: float | None) -> None:
    """Keep first-pass photo commits independent of resumable verification."""

    state = progress.setdefault("verification", {"mode": "never", "decision": False})
    if state.get("complete"):
        return
    if state.get("decision") is None:
        progress["phase"] = "awaiting_verification"
        write_progress(batch_directory, progress)
        reporter.close()
        state["decision"] = choose_verification(arguments, state["mode"], has_records=bool(committed_record_count(progress)))
        write_progress(batch_directory, progress)
    if not state["decision"]:
        state["complete"] = True
        write_progress(batch_directory, progress)
        return
    pending = [item for item in progress["inputs"] if item.get("result") and any(
        record.get("verification", {}).get("status") not in {"complete", "failed"}
        for record in item["result"]["records"]
    )]
    def load_transcriber() -> Any:
        nonlocal pipeline
        if pipeline is None:
            candidate, metadata = build_managed_pipeline(arguments, batch_directory, request_timeout)
            validate_model_resume(progress, metadata, batch_directory)
            progress["model"] = metadata
            endpoint = metadata.get("api_base") or metadata["ollama_host"]
            progress.setdefault("endpoint_history", []).append(endpoint)
            write_progress(batch_directory, progress)
            pipeline = candidate
        return pipeline.transcriber

    def load_date_detector() -> Any:
        if pipeline is None:
            name = selected_date_detector(arguments)
            if name in {"none", "regex"}:
                return DATE_DETECTORS[name](None)
            # A missing model-based extraction follows the same lazy identity check.
            load_transcriber()
        return getattr(pipeline, "date_detector", None)

    progress["phase"] = "verifying"
    write_progress(batch_directory, progress)
    for item in pending:
        for evidence in item.get("crop_evidence", []):
            validate_crop_evidence(batch_directory, evidence)

        def checkpoint(result: ImageResult, item: dict[str, Any] = item) -> None:
            item["result"] = serialize_result(result, batch_directory)
            item["state"] = result.status
            write_progress(batch_directory, progress)

        result = verify_image(deserialize_result(item["result"], batch_directory), None,
                              on_stage=reporter.phase, checkpoint=checkpoint, get_transcriber=load_transcriber,
                              get_date_detector=load_date_detector)
        checkpoint(result)
    state["complete"] = True
    write_progress(batch_directory, progress)


def append_stage_failure(progress: dict[str, Any], stage: str, error: Exception | str) -> None:
    failure = {"stage": stage, "error": warning_text(error)}
    failures = progress.setdefault("run_failures", [])
    if failure not in failures:
        failures.append(failure)


def managed_public_summary(value: dict[str, Any], *, batch_directory: Path | None, resumed: bool,
    images_reused: int, images_processed_this_start: int, status: str | None = None,
    trust_catalogue_output: bool = True) -> dict[str, Any]:

    inputs = value.get("inputs", [])
    states = [item.get("state") for item in inputs if isinstance(item, dict)]
    record_count = value.get("catalogue_records")
    if not isinstance(record_count, int):
        record_count = committed_record_count(value)
    failures = value.get("failures")
    counted_failures = len(failures) if isinstance(failures, list) else failure_count(value)
    actual_status = status or value.get("status") or "failed"
    details_path = batch_directory / DETAILS_FILE_NAME if batch_directory else None
    progress_path = batch_directory / PROGRESS_FILE_NAME if batch_directory else None
    manifest = value.get("run_manifest")
    if batch_directory:
        manifest = str(details_path if details_path and path_entry_exists(details_path) else progress_path)
    # A failure before publication can leave a checkpoint mentioning a target
    # path. Report it as published only when the caller has verified it.
    catalogue_output = value.get("catalogue_output") if trust_catalogue_output else None
    return {
        "schema_version": MANAGED_SCHEMA_VERSION,
        "status": actual_status,
        "images_total": len(inputs),
        "images_processed": sum(state in TERMINAL_IMAGE_STATES for state in states),
        "images_complete": states.count("complete"),
        "images_partial": states.count("partial"),
        "images_failed": states.count("failed"),
        "catalogue_output": catalogue_output,
        "catalogue_records": record_count,
        "failure_count": counted_failures,
        "model": value.get("model", {}).get("model"),
        "run_manifest": manifest,
        "previous_results": value.get("previous_results", []),
        "batch_directory": str(batch_directory) if batch_directory else value.get("batch_directory"),
        "catalogue_modified": bool(value.get("catalogue_modified")),
        "resumed": resumed,
        "images_reused": images_reused,
        "images_processed_this_start": images_processed_this_start,
    }


@contextmanager
def managed_interruption_handlers() -> Iterator[None]:
    watched: list[tuple[signal.Signals, Any]] = []

    def interrupted(signum: int, frame: Any) -> None:
        if signum == getattr(signal, "SIGTERM", object()):
            raise ManagedInterruption(143)
        if signum == getattr(signal, "SIGHUP", object()):
            raise ManagedInterruption(129)
        raise ManagedInterruption(130)

    for name in ("SIGINT", "SIGTERM", "SIGHUP", "SIGBREAK"):
        signum = getattr(signal, name, None)
        if signum is None:
            continue
        try:
            previous = signal.signal(signum, interrupted)
        except (OSError, ValueError):
            continue
        watched.append((signum, previous))
    try:
        yield
    finally:
        for signum, previous in watched:
            signal.signal(signum, previous)


def already_complete_summary(details: dict[str, Any], batch: Path) -> dict[str, Any]:
    summary = managed_public_summary(
        details,
        batch_directory=batch,
        resumed=False,
        images_reused=len(details.get("inputs", [])),
        images_processed_this_start=0,
    )
    summary["already_complete"] = True
    return summary


def completed_history_summary(completed: list[tuple[Path, dict[str, Any]]],
                              snapshot: list[dict[str, Any]]) -> dict[str, Any]:
    """Count current photo outcomes separately from whole published catalogues.

    catalogue_records and catalogue_record_counts retain the publication's
    recorded row counts, including missing-box suggestions and historical
    photos no longer selected. They do not recount subsequent CSV edits.
    current_input_records counts only saved transcription records for current
    photos, once each, without synthesizing missing-box suggestions.
    """

    remaining = {input_identity(item) for item in snapshot}
    summaries = []
    for batch, details in completed:
        inputs = [item for item in details["inputs"] if input_identity(item) in remaining]
        if not inputs:
            continue
        remaining.difference_update(input_identity(item) for item in inputs)
        value = {**details, "inputs": inputs}
        current_records = committed_record_count(value)
        value["status"] = final_status(value, current_records)
        # Per-photo failures from removed or superseded inputs do not apply.
        # Run-level failures still apply to the contributing batch.
        value.pop("failures", None)
        value["catalogue_output"] = str(batch / catalogue_file_name(details)) if details.get("catalogue_records") else None
        item_summary = already_complete_summary(value, batch)
        item_summary["current_input_records"] = current_records
        summaries.append(item_summary)

    summary = dict(summaries[0])
    summary["catalogue_outputs"] = [item["catalogue_output"] for item in summaries if item["catalogue_output"]]
    summary["catalogue_record_counts"] = {
        item["catalogue_output"]: item["catalogue_records"] for item in summaries if item["catalogue_output"]
    }
    summary["batch_directories"] = [item["batch_directory"] for item in summaries]
    summary["run_manifests"] = [item["run_manifest"] for item in summaries]
    if len(summaries) > 1:
        for field in ("images_total", "images_processed", "images_complete", "images_partial", "images_failed",
                      "images_reused", "catalogue_records", "current_input_records", "failure_count"):
            summary[field] = sum(item[field] for item in summaries)
        summary["status"] = ("failed" if not summary["current_input_records"] else
                             "partial" if any(item["status"] != "complete" for item in summaries) else "complete")
        summary["catalogue_modified"] = any(item["catalogue_modified"] for item in summaries)
        for field in ("catalogue_output", "batch_directory", "run_manifest"):
            summary[field] = None
        if len({item["model"] for item in summaries}) > 1:
            summary["model"] = None
        summary["previous_results"] = summary["catalogue_outputs"]
    return summary


def processed_since_start(progress: dict[str, Any] | None, processed: int, reused: int) -> int:
    if progress is None:
        return processed
    terminal = sum(item.get("state") in TERMINAL_IMAGE_STATES for item in progress.get("inputs", []))
    return max(processed, terminal - reused)


def interrupted_summary(arguments: argparse.Namespace, images: list[Path], progress: dict[str, Any] | None,
    batch_directory: Path | None, resumed: bool, reused: int, processed: int) -> dict[str, Any]:

    if progress is not None and batch_directory is not None:
        progress["interrupted_phase"] = progress.get("phase", "processing")
        progress["phase"] = "interrupted"
        if can_write_progress(batch_directory):
            try:
                write_progress(batch_directory, progress)
            except Exception as error:
                print(f"Could not record interruption: {warning_text(error)}", file=sys.stderr)
    value = progress or {
        "model": selected_model_metadata(arguments),
        "inputs": [{"state": "pending"} for image in images],
    }
    return managed_public_summary(
        value,
        batch_directory=batch_directory,
        resumed=resumed,
        images_reused=reused,
        images_processed_this_start=processed,
        status="interrupted",
        trust_catalogue_output=False,
    )


def run_managed_batch(arguments: argparse.Namespace, images: list[Path], results_root: Path) -> int:
    """Run or resume one managed batch and show its final outcome."""

    with TerminalProgress(len(images)) as reporter:
        return _run_managed_batch(arguments, images, results_root, reporter)


def _run_managed_batch(arguments: argparse.Namespace, images: list[Path], results_root: Path,
    reporter: TerminalProgress) -> int:
    """Keep checkpoint, publication, and interruption handling together."""

    lock = ResultsRootLock(results_root)
    batch_directory: Path | None = None
    progress: dict[str, Any] | None = None
    resumed = False
    reused = 0
    processed = 0
    exit_code = 2
    summary: dict[str, Any]
    interruption_context = managed_interruption_handlers()
    interruption_context.__enter__()
    try:
        reporter.phase("Checking input photos and saved progress")
        lock.acquire()
        request_timeout = arguments.request_timeout
        if request_timeout is None:
            request_timeout = timeout_from_environment()
        unfinished, _completed = managed_batches(results_root)
        if unfinished:
            expected_names = {item["relative_name"] for item in unfinished[0][1].get("input_snapshot", [])}
            selected_images = [image for image in images if image.name in expected_names]
        else:
            selected_images = images
        snapshot = input_snapshot(selected_images, arguments.path_to_inputs)
        batch_directory, progress, resumed, completed = select_batch(
            arguments, results_root, snapshot, request_timeout
        )
        # A completed set can be reused only after its source, crop, and catalogue
        # evidence are checked again; a stale path is not enough to trust it.
        if completed is not None:
            summary = completed_history_summary(completed, snapshot)
            if len(completed) == 1:
                batch_directory = completed[0][0]
            print(
                "This photo set has saved outcomes in:\n"
                + "\n".join(summary["batch_directories"]) + "\n"
                "Use --reprocess to create another batch.",
                file=sys.stderr,
            )
            for catalogue in summary["catalogue_outputs"]:
                print(f"Saved catalogue: {catalogue}", file=sys.stderr)
            exit_code = 0 if summary["status"] == "complete" else 1
        else:
            assert batch_directory is not None and progress is not None
            images = [Path(item["source_path"]) for item in progress["inputs"]]
            reporter.set_total(len(images))
            if resumed:
                reused = prepare_resume(progress, batch_directory)
                reporter.reused(reused)
                progress["resume_count"] = int(progress.get("resume_count", 0)) + 1
                if progress.get("phase") == "interrupted":
                    progress["phase"] = progress.pop("interrupted_phase", "processing")
                write_progress(batch_directory, progress)

            # A resumed photo set must also use the same model identity and
            # endpoint class before it adds new readings to prior ones.
            pipeline = None
            if any(item["state"] not in TERMINAL_IMAGE_STATES for item in progress["inputs"]):
                reporter.phase("Checking the selected model")
                pipeline, model_metadata = build_managed_pipeline(arguments, batch_directory, request_timeout)
                if isinstance(pipeline, BoxTranscriptionPipeline):
                    pipeline.on_stage = reporter.phase
                validate_model_resume(progress, model_metadata, batch_directory)
                progress["model"] = model_metadata
                endpoint = model_metadata.get("api_base") or model_metadata["ollama_host"]
                progress.setdefault("endpoint_history", []).append(endpoint)
                write_progress(batch_directory, progress)

            if progress.get("phase") not in {"publishing", "finalizing", "final"}:
                processed = process_pending_images(arguments, images, progress, batch_directory, pipeline, reporter)
            if progress.get("phase") not in {"staging", "publishing", "finalizing", "final"}:
                verify_managed_results(arguments, progress, batch_directory, pipeline, reporter, request_timeout)
            reporter.phase("Finalizing results")
            details = finalize_batch(arguments, progress, batch_directory)
            summary = managed_public_summary(
                details,
                batch_directory=batch_directory,
                resumed=resumed,
                images_reused=reused,
                images_processed_this_start=processed,
            )
            exit_code = 0 if details["status"] == "complete" else 1
    except (ManagedInterruption, KeyboardInterrupt) as interruption:
        exit_code = interruption.exit_code if isinstance(interruption, ManagedInterruption) else 130
        processed = processed_since_start(progress, processed, reused)
        summary = interrupted_summary(
            arguments, images, progress, batch_directory, resumed, reused, processed
        )
    except Exception as error:
        if batch_directory is None and isinstance(error, ManagedBatchSelectionError):
            batch_directory = error.batch_directory
        processed = processed_since_start(progress, processed, reused)
        message = warning_text(error)
        print(f"Managed batch failed: {message}", file=sys.stderr)
        if progress is not None and batch_directory is not None and can_write_progress(batch_directory):
            append_stage_failure(progress, "managed_batch", error)
            try:
                write_progress(batch_directory, progress)
            except Exception as checkpoint_error:
                print(f"Checkpoint write failed: {warning_text(checkpoint_error)}", file=sys.stderr)
        summary_value = progress or {
            "model": selected_model_metadata(arguments),
            "inputs": [{"state": "pending"} for image in images],
            "run_failures": [{"stage": "managed_batch", "error": message}],
        }
        summary = managed_public_summary(
            summary_value,
            batch_directory=batch_directory,
            resumed=resumed,
            images_reused=reused,
            images_processed_this_start=processed,
            status="partial" if progress and committed_record_count(progress) else "failed",
            trust_catalogue_output=False,
        )
        summary["error"] = message
        exit_code = 2
    finally:
        interruption_context.__exit__(None, None, None)
        try:
            lock.release()
        except Exception as error:
            message = warning_text(error)
            print(f"Results lock cleanup failed: {message}", file=sys.stderr)
            if "summary" in locals():
                summary["failure_count"] = int(summary.get("failure_count", 0)) + 1
                summary["status"] = "partial" if summary.get("catalogue_records") else "failed"
                summary["lock_cleanup_error"] = message
            exit_code = 2

    reporter.close()
    print_run_summary(summary, human=not getattr(arguments, "json", False), elapsed_seconds=reporter.elapsed_seconds())
    return exit_code
