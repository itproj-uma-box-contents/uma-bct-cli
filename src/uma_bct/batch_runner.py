"""Batch orchestration and recovery-aware result reporting."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .pipeline import ImageResult, warning_text


@dataclass(frozen=True)
class BatchOperations:
    """Injected operations that keep CLI compatibility shims patchable."""

    build_pipeline: Callable[..., Any]
    export_catalogue: Callable[..., Any]
    timeout_from_environment: Callable[[], float]
    run_directory: Callable[[Path], Path]
    failed_manifest_path: Callable[[Path], Path]
    backup_existing_catalogue: Callable[[Path], Path | None]
    restore_catalogue: Callable[[Path, Path | None], None]
    write_json_atomic: Callable[[Path, dict[str, Any]], None]


def configuration(arguments: argparse.Namespace, work_directory: Path | None,
    request_timeout_seconds: float | None) -> dict[str, Any]:
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


def public_summary(detailed_summary: dict[str, Any], *, image_count: int, model_name: str) -> dict[str, Any]:
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
        "catalogue_backup": detailed_summary.get("catalogue_backup"),
    }


def _process_images(pipeline: Any, images: list[Path], failures: list[dict[str, str]]) -> list[ImageResult]:
    results: list[ImageResult] = []
    for image_path in images:
        displayed_path = ascii(str(image_path))
        print(f"Processing {displayed_path}", file=sys.stderr)
        try:
            result = pipeline.process(image_path)
        except Exception as error:  # Continue after one failed image.
            message = warning_text(error)
            failures.append({"source_image": str(image_path), "error": message})
            print(f"Failed {displayed_path}: {message}", file=sys.stderr)
            continue
        results.append(result)
        for warning in result.warnings:
            message = warning_text(ValueError(warning))
            print(f"Warning for {displayed_path}: {message}", file=sys.stderr)
    return results


def write_recovery_manifest(output_path: Path, summary: dict[str, Any],
                            operations: BatchOperations) -> None:
    recovery_manifest_path = operations.failed_manifest_path(output_path)
    summary["run_manifest"] = str(recovery_manifest_path)
    try:
        operations.write_json_atomic(recovery_manifest_path, summary)
    except Exception as error:
        message = warning_text(error)
        summary["failures"].append({"stage": "recovery manifest", "error": message})
        summary["run_manifest"] = None
        print(f"Recovery manifest write failed: {message}", file=sys.stderr)


def handle_manifest_failure(*, error: Exception, summary: dict[str, Any], output_path: Path,
    catalogue_output: str | None, catalogue_backup: Path | None, operations: BatchOperations) -> None:
    message = warning_text(error)
    summary["failures"].append({"stage": "manifest", "error": message})
    print(f"Manifest write failed: {message}", file=sys.stderr)

    if catalogue_output is not None:
        try:
            operations.restore_catalogue(output_path, catalogue_backup)
        except Exception as rollback_error:
            rollback_message = warning_text(rollback_error)
            summary["status"] = "partial"
            summary["failures"].append(
                {
                    "stage": "catalogue rollback",
                    "error": rollback_message,
                    "backup": str(catalogue_backup) if catalogue_backup else "",
                }
            )
            retained_backup = (
                catalogue_backup
                if catalogue_backup is not None and catalogue_backup.exists()
                else None
            )
            summary["catalogue_backup"] = (
                str(retained_backup) if retained_backup else None
            )
            print(f"Catalogue rollback failed: {rollback_message}", file=sys.stderr)
            if retained_backup is not None:
                print(
                    f"Private catalogue backup retained for recovery: "
                    f"{ascii(str(retained_backup))}",
                    file=sys.stderr,
                )
        else:
            summary["status"] = "failed"
            summary["catalogue_output"] = None
            summary["catalogue_records"] = 0
    else:
        summary["status"] = "failed"
        if catalogue_backup is not None:
            try:
                catalogue_backup.unlink(missing_ok=True)
            except OSError as cleanup_error:
                cleanup_message = warning_text(cleanup_error)
                summary["catalogue_backup"] = str(catalogue_backup)
                summary["failures"].append(
                    {
                        "stage": "catalogue backup cleanup",
                        "error": cleanup_message,
                        "backup": str(catalogue_backup),
                    }
                )
                print(
                    f"Warning: could not remove catalogue backup "
                    f"{ascii(str(catalogue_backup))}: {cleanup_message}",
                    file=sys.stderr,
                )

    write_recovery_manifest(output_path, summary, operations)


def remove_catalogue_backup(*, catalogue_backup: Path | None, output_path: Path,
    summary: dict[str, Any], operations: BatchOperations) -> None:

    if catalogue_backup is None:
        return
    try:
        catalogue_backup.unlink(missing_ok=True)
    except OSError as error:
        message = warning_text(error)
        summary["catalogue_backup"] = str(catalogue_backup)
        summary["failures"].append(
            {
                "stage": "catalogue backup cleanup",
                "error": message,
                "backup": str(catalogue_backup),
            }
        )
        if summary["status"] == "complete":
            summary["status"] = "partial"
        print(
            f"Warning: could not remove catalogue backup "
            f"{ascii(str(catalogue_backup))}: {message}",
            file=sys.stderr,
        )
        write_recovery_manifest(output_path, summary, operations)


def run_batch(arguments: argparse.Namespace, images: list[Path], output_path: Path,
    normal_manifest_path: Path, *, preserve_existing_manifest_on_failure: bool, 
    operations: BatchOperations) -> int:
    """
    Process one image batch and emit its single public JSON summary.
    """

    started_at = datetime.now(UTC)
    failures: list[dict[str, str]] = []
    work_directory: Path | None = None
    model_metadata: dict[str, str] = {"model": arguments.model}
    request_timeout_seconds: float | None = arguments.request_timeout

    try:
        if request_timeout_seconds is None:
            request_timeout_seconds = operations.timeout_from_environment()
        pipeline = operations.build_pipeline(
            detector_name=arguments.detector,
            transcriber_name=arguments.transcriber,
            model_name=arguments.model,
            work_directory=arguments.path_to_box_images,
            minimum_detector_confidence=arguments.min_detector_confidence,
            request_timeout_seconds=request_timeout_seconds,
        )
        model_metadata.update(getattr(pipeline, "model_metadata", {}))
        # Invalid model setup should not leave behind an empty crop-run directory.
        work_directory = operations.run_directory(arguments.path_to_box_images)
        pipeline.work_directory = work_directory
    except Exception as error:
        failures.append({"stage": "preflight", "error": warning_text(error)})
        pipeline = None

    results = (
        _process_images(pipeline, images, failures) if pipeline is not None else []
    )
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
            catalogue_backup = operations.backup_existing_catalogue(output_path)
            records = operations.export_catalogue(
                results,
                output_path,
                minimum_confidence=arguments.min_confidence,
            )
            record_count = len(records)
            catalogue_output = str(output_path)
        except Exception as error:
            message = warning_text(error)
            failures.append({"stage": "export", "error": message})
            print(f"Export failed: {message}", file=sys.stderr)
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
        actual_manifest_path = operations.failed_manifest_path(output_path)
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
        "catalogue_backup": None,
    }

    try:
        operations.write_json_atomic(actual_manifest_path, summary)
    except Exception as error:
        handle_manifest_failure(
            error=error,
            summary=summary,
            output_path=output_path,
            catalogue_output=catalogue_output,
            catalogue_backup=catalogue_backup,
            operations=operations,
        )
    else:
        remove_catalogue_backup(
            catalogue_backup=catalogue_backup,
            output_path=output_path,
            summary=summary,
            operations=operations,
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
        # Connectivity, version, model, and timeout failures are setup errors.
        return 2
    return 0 if summary["status"] == "complete" else 1
