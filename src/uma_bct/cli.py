"""Command-line entry point for the archive-box transcription pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import stat
import sys
import tempfile
import uuid
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .adapters.ollama_client import (
    DEFAULT_MODEL,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    TIMEOUT_ENVIRONMENT_VARIABLE,
    timeout_from_environment,
)
from .bootstrap import DETECTORS, TRANSCRIBERS, build_pipeline
from .constants import IMAGE_EXTENSIONS
from .export import SUPPORTED_OUTPUT_EXTENSIONS, export_catalogue
from .pipeline import ImageResult

GENERATED_STEM_MAX_BYTES = 120


def package_version() -> str:
    try:
        return version("uma-bct-cli")
    except PackageNotFoundError:
        return "0.1.0"


def bounded_float(value: str) -> float:
    try:
        confidence = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number between 0 and 1") from error
    if not 0.0 <= confidence <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return confidence


def bounded_percentage(value: str) -> int:
    try:
        confidence = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "must be an integer between 0 and 100"
        ) from error
    if not 0 <= confidence <= 100:
        raise argparse.ArgumentTypeError("must be between 0 and 100")
    return confidence


def positive_float(value: str) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive number") from error
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect and transcribe UMA archive boxes using local components."
    )
    parser.add_argument(
        "-i",
        "--path-to-inputs",
        type=Path,
        default=Path("inputs"),
        help="input image or directory of images (default: %(default)s)",
    )
    parser.add_argument(
        "-o",
        "--path-to-outputs",
        "--path-to-output",
        dest="path_to_outputs",
        type=Path,
        default=Path("outputs.csv"),
        help="catalogue path ending in .csv or .xlsx (default: %(default)s)",
    )
    parser.add_argument(
        "-b",
        "--path-to-box-images",
        "--work-dir",
        dest="path_to_box_images",
        type=Path,
        default=Path("outputs") / "crops",
        help="root directory for review crops (default: %(default)s)",
    )
    parser.add_argument(
        "-d",
        "--detector",
        choices=tuple(DETECTORS),
        default="qwen",
        help="box detector to use (default: %(default)s)",
    )
    parser.add_argument(
        "-t",
        "--transcriber",
        choices=tuple(TRANSCRIBERS),
        default="qwen",
        help="text transcriber to use (default: %(default)s)",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=DEFAULT_MODEL,
        help="locally installed Ollama model tag (default: %(default)s)",
    )
    parser.add_argument(
        "-c",
        "--min-confidence",
        type=bounded_percentage,
        default=80,
        metavar="INTEGER",
        help="transcription review threshold from 0 to 100 (default: %(default)s)",
    )
    parser.add_argument(
        "--min-detector-confidence",
        type=bounded_float,
        default=0.60,
        metavar="FLOAT",
        help="OpenCV detector threshold from 0 to 1 (default: %(default).2f)",
    )
    parser.add_argument(
        "--request-timeout",
        type=positive_float,
        default=None,
        metavar="SECONDS",
        help=(
            "Ollama request timeout; defaults to "
            f"${TIMEOUT_ENVIRONMENT_VARIABLE} or {DEFAULT_REQUEST_TIMEOUT_SECONDS:g}s"
        ),
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="replace an existing catalogue output",
    )
    parser.add_argument("--version", action="version", version=package_version())
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_argument_parser().parse_args(argv)


def ensure_readable_image(path: Path) -> None:
    """Open an image once during preflight so permission failures are usage errors."""

    try:
        with path.open("rb") as image_file:
            image_file.read(1)
    except OSError as error:
        raise ValueError(f"Input image is not readable: {path}: {error}") from error


def discover_images(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.casefold() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {path.suffix or '<none>'}")
        ensure_readable_image(path)
        return [path]
    if not path.is_dir():
        raise ValueError(f"Input path does not exist: {path}")
    if not os.access(path, os.R_OK):
        raise ValueError(f"Input directory is not readable: {path}")
    try:
        candidates = tuple(path.iterdir())
    except OSError as error:
        raise ValueError(f"Input directory is not readable: {path}: {error}") from error

    images: list[Path] = []
    for candidate in candidates:
        if candidate.suffix.casefold() not in IMAGE_EXTENSIONS:
            continue
        try:
            is_supported_file = stat.S_ISREG(candidate.stat().st_mode)
        except OSError as error:
            raise ValueError(
                f"Could not inspect input path {candidate}: {error}"
            ) from error
        if is_supported_file:
            ensure_readable_image(candidate)
            images.append(candidate)
    return sorted(
        images, key=lambda candidate: (candidate.name.casefold(), candidate.name)
    )


def bounded_generated_stem(value: str) -> str:
    """Bound generated sibling names while keeping long stems collision-safe."""

    value = value.rstrip(" .") or "output"
    encoded = value.encode("utf-8")
    if len(encoded) <= GENERATED_STEM_MAX_BYTES:
        return value
    digest = hashlib.sha256(encoded).hexdigest()[:10]
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


def configuration(
    arguments: argparse.Namespace,
    work_directory: Path | None,
    request_timeout_seconds: float | None,
) -> dict[str, Any]:
    return {
        "detector": arguments.detector,
        "transcriber": arguments.transcriber,
        "model": arguments.model,
        "minimum_transcription_confidence": arguments.min_confidence,
        "minimum_detector_confidence": arguments.min_detector_confidence,
        "request_timeout_seconds": request_timeout_seconds,
        "input": str(arguments.path_to_inputs),
        "output": str(arguments.path_to_outputs),
        "box_images": str(work_directory) if work_directory else None,
    }


def public_summary(
    detailed_summary: dict[str, Any], *, image_count: int, model_name: str
) -> dict[str, Any]:
    images = detailed_summary["images"]
    complete_count = sum(item.get("status") == "complete" for item in images)
    partial_count = sum(item.get("status") == "partial" for item in images)
    return {
        "schema_version": detailed_summary["schema_version"],
        "status": detailed_summary["status"],
        "elapsed_seconds": detailed_summary["elapsed_seconds"],
        "images_total": image_count,
        "images_processed": len(images),
        "images_complete": complete_count,
        "images_partial": partial_count,
        "images_failed": image_count - complete_count - partial_count,
        "catalogue_output": detailed_summary["catalogue_output"],
        "catalogue_records": detailed_summary["catalogue_records"],
        "failure_count": len(detailed_summary["failures"]),
        "model": model_name,
        "run_manifest": detailed_summary["run_manifest"],
    }


def _run_batch(
    arguments: argparse.Namespace,
    images: list[Path],
    output_path: Path,
    normal_manifest_path: Path,
    *,
    preserve_existing_manifest_on_failure: bool,
) -> int:
    started_at = datetime.now(UTC)
    failures: list[dict[str, str]] = []
    results: list[ImageResult] = []
    work_directory: Path | None = None
    model_metadata: dict[str, str] = {"model": arguments.model}
    request_timeout_seconds: float | None = arguments.request_timeout

    try:
        if request_timeout_seconds is None:
            request_timeout_seconds = timeout_from_environment()
        pipeline = build_pipeline(
            detector_name=arguments.detector,
            transcriber_name=arguments.transcriber,
            model_name=arguments.model,
            work_directory=arguments.path_to_box_images,
            minimum_detector_confidence=arguments.min_detector_confidence,
            request_timeout_seconds=request_timeout_seconds,
        )
        model_metadata.update(getattr(pipeline, "model_metadata", {}))
        work_directory = run_directory(arguments.path_to_box_images)
        pipeline.work_directory = work_directory
    except Exception as error:
        failures.append({"stage": "preflight", "error": str(error)})
        pipeline = None

    if pipeline is not None:
        for image_path in images:
            print(f"Processing {image_path}", file=sys.stderr)
            try:
                result = pipeline.process(image_path)
            except Exception as error:  # Continue after one failed image.
                failures.append({"source_image": str(image_path), "error": str(error)})
                print(f"Failed {image_path}: {error}", file=sys.stderr)
                continue
            results.append(result)
            for warning in result.warnings:
                print(f"Warning for {image_path}: {warning}", file=sys.stderr)

    incomplete_images = [
        {
            "source_image": str(result.input_image.file_path),
            "status": getattr(result, "status", "complete"),
            "warnings": list(result.warnings),
        }
        for result in results
        if getattr(result, "status", "complete") != "complete"
    ]
    expected_records = sum(len(result.transcriptions) for result in results)
    record_count = 0
    catalogue_output: str | None = None
    catalogue_backup: Path | None = None
    if expected_records:
        try:
            catalogue_backup = backup_existing_catalogue(output_path)
            records = export_catalogue(
                results,
                output_path,
                minimum_confidence=arguments.min_confidence,
            )
            record_count = len(records)
            catalogue_output = str(output_path)
        except Exception as error:
            failures.append({"stage": "export", "error": str(error)})
            print(f"Export failed: {error}", file=sys.stderr)
            if catalogue_backup is not None:
                try:
                    catalogue_backup.unlink(missing_ok=True)
                except OSError as cleanup_error:
                    print(
                        f"Warning: could not remove catalogue backup "
                        f"{catalogue_backup}: {cleanup_error}",
                        file=sys.stderr,
                    )
                catalogue_backup = None
    elif pipeline is not None:
        failures.append(
            {
                "stage": "processing",
                "error": "No usable catalogue records were produced.",
            }
        )

    status = "complete"
    if failures or incomplete_images:
        status = "partial" if record_count else "failed"
    completed_at = datetime.now(UTC)
    actual_manifest_path = normal_manifest_path
    if preserve_existing_manifest_on_failure and record_count == 0:
        actual_manifest_path = failed_manifest_path(output_path)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": status,
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "elapsed_seconds": round((completed_at - started_at).total_seconds(), 3),
        "configuration": configuration(
            arguments,
            work_directory,
            request_timeout_seconds,
        ),
        "model": model_metadata,
        "images": [result.to_dict() for result in results],
        "incomplete_images": incomplete_images,
        "failures": failures,
        "catalogue_output": catalogue_output,
        "catalogue_records": record_count,
        "run_manifest": str(actual_manifest_path),
    }
    try:
        write_json_atomic(actual_manifest_path, summary)
    except Exception as error:
        summary["failures"].append({"stage": "manifest", "error": str(error)})
        print(f"Manifest write failed: {error}", file=sys.stderr)

        if catalogue_output is not None:
            try:
                restore_catalogue(output_path, catalogue_backup)
            except Exception as rollback_error:
                summary["status"] = "partial"
                summary["failures"].append(
                    {
                        "stage": "catalogue rollback",
                        "error": str(rollback_error),
                        "backup": str(catalogue_backup) if catalogue_backup else "",
                    }
                )
                print(f"Catalogue rollback failed: {rollback_error}", file=sys.stderr)
                # Preserve the private backup for manual recovery instead of
                # deleting the user's previous catalogue after a failed rollback.
                catalogue_backup = None
            else:
                catalogue_backup = None
                record_count = 0
                catalogue_output = None
                summary["status"] = "failed"
                summary["catalogue_output"] = None
                summary["catalogue_records"] = 0
        else:
            summary["status"] = "failed"

        recovery_manifest_path = failed_manifest_path(output_path)
        summary["run_manifest"] = str(recovery_manifest_path)
        try:
            write_json_atomic(recovery_manifest_path, summary)
        except Exception as recovery_error:
            summary["failures"].append(
                {"stage": "recovery manifest", "error": str(recovery_error)}
            )
            summary["run_manifest"] = None
            print(
                f"Recovery manifest write failed: {recovery_error}",
                file=sys.stderr,
            )
    else:
        if catalogue_backup is not None:
            try:
                catalogue_backup.unlink(missing_ok=True)
            except OSError as error:
                print(
                    f"Warning: could not remove catalogue backup "
                    f"{catalogue_backup}: {error}",
                    file=sys.stderr,
                )
            catalogue_backup = None
    finally:
        if catalogue_backup is not None:
            try:
                catalogue_backup.unlink(missing_ok=True)
            except OSError as error:
                print(
                    f"Warning: could not remove catalogue backup "
                    f"{catalogue_backup}: {error}",
                    file=sys.stderr,
                )

    print(
        json.dumps(
            public_summary(
                summary, image_count=len(images), model_name=arguments.model
            ),
            indent=2,
            ensure_ascii=False,
        )
    )
    if pipeline is None:
        # Connectivity, server-version, model-availability, and timeout-setting
        # failures are configuration/preflight errors rather than partial runs.
        return 2
    return 0 if summary["status"] == "complete" else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    arguments = parser.parse_args(argv)
    output_path = arguments.path_to_outputs
    run_manifest_path = manifest_path(output_path)
    if output_path.suffix.casefold() not in SUPPORTED_OUTPUT_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_OUTPUT_EXTENSIONS))
        parser.error(f"output path must end in one of: {supported}")
    try:
        images = discover_images(arguments.path_to_inputs)
    except ValueError as error:
        parser.error(str(error))
    if not images:
        parser.error(f"no supported images found in: {arguments.path_to_inputs}")
    try:
        reservation = acquire_output_lock(output_path)
    except ValueError as error:
        parser.error(str(error))
    try:
        existing_output = next(
            (path for path in (output_path, run_manifest_path) if path.exists()),
            None,
        )
        if existing_output is not None and not arguments.force:
            parser.error(
                f"output already exists; pass --force to replace it: {existing_output}"
            )
        return _run_batch(
            arguments,
            images,
            output_path,
            run_manifest_path,
            preserve_existing_manifest_on_failure=(
                arguments.force and (output_path.exists() or run_manifest_path.exists())
            ),
        )
    finally:
        release_output_lock(*reservation)
