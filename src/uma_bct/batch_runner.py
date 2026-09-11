"""Run a batch and report what happened.

One bad image does not stop the rest. Errors are saved in the run result, and
the terminal only gets a short summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .adapters.ollama_client import timeout_from_environment
from .bootstrap import build_pipeline
from .export import export_catalogue
from .path_safety import check_path
from .pipeline import ImageResult, warning_text
from .run_artifacts import (
    CatalogueLock,
    backup_existing_results,
    failed_manifest_path,
    rollback_catalogue,
    run_directory,
    write_json_atomic,
)


@dataclass(frozen=True)
class ExportState:
    record_count: int = 0
    catalogue_output: str | None = None
    previous_results: tuple[Path, ...] = ()
    previous_catalogue: Path | None = None
    replace_manifest: bool = False
    wrote_catalogue: bool = False


def configuration(
    arguments: argparse.Namespace, work_directory: Path | None, request_timeout_seconds: float | None
) -> dict[str, Any]:
    return {
        "detector": arguments.detector,
        "transcriber": arguments.transcriber,
        "model": arguments.model,
        "minimum_transcription_confidence": arguments.min_transcription_confidence,
        "minimum_detector_confidence": arguments.min_detector_confidence,
        "request_timeout_seconds": request_timeout_seconds,
        "input": str(arguments.path_to_inputs),
        "output": str(arguments.path_to_outputs),
        "box_images": str(work_directory) if work_directory else None,
    }


def process_images(pipeline: Any, images: list[Path], failures: list[dict[str, str]]) -> list[ImageResult]:
    results: list[ImageResult] = []
    for image_path in images:
        print(f"Processing {ascii(str(image_path))}", file=sys.stderr)
        try:
            result = pipeline.process(image_path)
        except Exception as error:
            message = warning_text(error)
            failures.append({"source_image": str(image_path), "error": message})
            print(f"Failed {ascii(str(image_path))}: {message}", file=sys.stderr)
            continue
        results.append(result)
        for warning in result.warnings:
            print(f"Warning for {ascii(str(image_path))}: {warning_text(warning)}", file=sys.stderr)
    return results


def public_summary(summary: dict[str, Any], *, image_count: int, model_name: str) -> dict[str, Any]:
    images = summary["images"]
    result = {
        "schema_version": summary["schema_version"],
        "status": summary["status"],
        "images_total": image_count,
        "images_processed": len(images),
        "images_complete": sum(item["status"] == "complete" for item in images),
        "images_partial": sum(item["status"] == "partial" for item in images),
        "images_failed": image_count - sum(item["status"] in {"complete", "partial"} for item in images),
        "catalogue_output": summary["catalogue_output"],
        "catalogue_records": summary["catalogue_records"],
        "failure_count": len(summary["failures"]),
        "model": model_name,
        "run_manifest": summary["run_manifest"],
        "previous_results": summary["previous_results"],
    }
    if recovery_path := summary.get("recovery_path"):
        result["recovery_path"] = recovery_path
    return result


def prepare_pipeline(arguments: argparse.Namespace, failures: list[dict[str, str]]
) -> tuple[Any | None, Path | None, float | None, dict[str, str]]:

    request_timeout = arguments.request_timeout
    metadata = {"model": arguments.model}
    try:
        if request_timeout is None:
            request_timeout = timeout_from_environment()
        pipeline = build_pipeline(
            detector_name=arguments.detector,
            transcriber_name=arguments.transcriber,
            model_name=arguments.model,
            work_directory=arguments.path_to_box_images,
            minimum_detector_confidence=arguments.min_detector_confidence,
            request_timeout_seconds=request_timeout,
        )
        metadata.update(pipeline.model_metadata)
        work_directory = run_directory(arguments.path_to_box_images)
        pipeline.work_directory = work_directory
        return pipeline, work_directory, request_timeout, metadata
    except Exception as error:
        message = warning_text(error)
        failures.append({"stage": "preflight", "error": message})
        print(f"Preflight failed: {message}", file=sys.stderr)
        return None, None, request_timeout, metadata


def export_results(arguments: argparse.Namespace, results: list[ImageResult], output_path: Path,
                   manifest_path: Path, failures: list[dict[str, str]]) -> ExportState:
                   
    if not sum(len(result.records) for result in results):
        failures.append({"stage": "processing", "error": "No usable catalogue records were produced."})
        return ExportState()
    previous: tuple[Path, ...] = ()
    try:
        # Check again because model processing can take a long time.
        if not arguments.force:
            for path in (output_path, manifest_path):
                check_path(path, "Output")
                if path.exists():
                    raise FileExistsError(f"Output already exists: {path}")
        previous = backup_existing_results((output_path, manifest_path)) if arguments.force else ()
        previous_catalogue = next((path for path in previous if path.suffix.casefold() == ".csv"), None)
        replace_manifest = any(path.name.endswith(".run.json") for path in previous)
        records = export_catalogue(
            results,
            output_path,
            minimum_confidence=arguments.min_transcription_confidence,
            replace_existing=previous_catalogue is not None,
        )
        return ExportState(len(records), str(output_path), previous, previous_catalogue, replace_manifest, True)
    except Exception as error:
        message = warning_text(error)
        failures.append({"stage": "export", "error": message})
        print(f"Export failed: {message}", file=sys.stderr)
        return ExportState(previous_results=previous)


def write_manifest(
    manifest: Path, summary: dict[str, Any], failures: list[dict[str, str]], *, replace_existing: bool
) -> bool:
    try:
        write_json_atomic(manifest, summary, replace_existing=replace_existing)
    except Exception as error:
        message = warning_text(error)
        failures.append({"stage": "manifest", "error": message})
        summary["status"] = "partial" if summary["catalogue_output"] else "failed"
        summary["run_manifest"] = None
        print(f"Manifest write failed: {message}", file=sys.stderr)
        return False
    return True


def emit_summary(summary: dict[str, Any], image_count: int, model_name: str) -> None:
    print(json.dumps(public_summary(summary, image_count=image_count, model_name=model_name), indent=2, ensure_ascii=False))


def lock_failure_summary(arguments: argparse.Namespace, error: Exception) -> dict[str, Any]:
    message = warning_text(error)
    print(f"Lock failed: {message}", file=sys.stderr)
    return {
        "schema_version": 1,
        "status": "failed",
        "configuration": configuration(arguments, None, arguments.request_timeout),
        "model": {"model": arguments.model},
        "images": [],
        "failures": [{"stage": "lock", "error": message}],
        "catalogue_output": None,
        "catalogue_records": 0,
        "run_manifest": None,
        "previous_results": [],
    }


def recover_manifest_failure(
    output_path: Path,
    summary: dict[str, Any],
    export_state: ExportState,
    failures: list[dict[str, str]],
) -> None:
    try:
        rollback_catalogue(output_path, export_state.previous_catalogue)
    except Exception as error:
        message = warning_text(error)
        recovery_path = export_state.previous_catalogue or output_path
        failures.append({"stage": "rollback", "error": message})
        summary["status"] = "partial"
        summary["recovery_path"] = str(recovery_path)
        print(f"Rollback failed: {message}. Recovery file retained at {recovery_path}.", file=sys.stderr)
    else:
        summary["status"] = "failed"
        summary["catalogue_output"] = None
        summary["catalogue_records"] = 0

    recovery_manifest = failed_manifest_path(output_path)
    summary["run_manifest"] = str(recovery_manifest)
    write_manifest(recovery_manifest, summary, failures, replace_existing=False)


def locked_batch_summary(
    arguments: argparse.Namespace,
    images: list[Path],
    output_path: Path,
    normal_manifest_path: Path,
    *,
    preserve_existing_manifest_on_failure: bool,
) -> tuple[dict[str, Any], bool]:
    failures: list[dict[str, str]] = []
    pipeline, work_directory, request_timeout, model_metadata = prepare_pipeline(arguments, failures)
    results = process_images(pipeline, images, failures) if pipeline else []
    export_state = (
        export_results(arguments, results, output_path, normal_manifest_path, failures) if pipeline else ExportState()
    )

    incomplete = any(result.status != "complete" for result in results)
    status = "complete"
    if failures or incomplete:
        status = "partial" if export_state.record_count else "failed"

    manifest = (
        failed_manifest_path(output_path)
        if preserve_existing_manifest_on_failure and not export_state.record_count
        else normal_manifest_path
    )
    summary = {
        "schema_version": 1,
        "status": status,
        "configuration": configuration(arguments, work_directory, request_timeout),
        "model": model_metadata,
        "images": [result.to_dict() for result in results],
        "failures": failures,
        "catalogue_output": export_state.catalogue_output,
        "catalogue_records": export_state.record_count,
        "run_manifest": str(manifest),
        "previous_results": [str(path) for path in export_state.previous_results],
    }
    manifest_written = write_manifest(manifest, summary, failures, replace_existing=export_state.replace_manifest)
    if not manifest_written and export_state.wrote_catalogue:
        recover_manifest_failure(output_path, summary, export_state, failures)
    return summary, pipeline is not None


def run_batch(arguments: argparse.Namespace, images: list[Path], output_path: Path, normal_manifest_path: Path, *,
              preserve_existing_manifest_on_failure: bool) -> int:
    """Process a batch, write its files, and print a JSON summary."""

    lock = CatalogueLock(output_path)
    try:
        lock.acquire()
    except Exception as error:
        summary = lock_failure_summary(arguments, error)
        emit_summary(summary, len(images), arguments.model)
        return 2

    release_error: str | None = None
    try:
        summary, pipeline_ready = locked_batch_summary(
            arguments,
            images,
            output_path,
            normal_manifest_path,
            preserve_existing_manifest_on_failure=preserve_existing_manifest_on_failure,
        )
    finally:
        try:
            lock.release()
        except Exception as error:
            release_error = warning_text(error)

    if release_error is not None:
        summary["failures"].append({"stage": "lock_cleanup", "error": release_error})
        summary["status"] = "partial" if summary["catalogue_output"] else "failed"
        print(f"Lock cleanup failed: {release_error}", file=sys.stderr)
        recovery_manifest = failed_manifest_path(output_path)
        summary["run_manifest"] = str(recovery_manifest)
        write_manifest(recovery_manifest, summary, summary["failures"], replace_existing=False)

    emit_summary(summary, len(images), arguments.model)
    if release_error is not None:
        return 2
    if not pipeline_ready:
        return 2
    return 0 if summary["status"] == "complete" else 1
