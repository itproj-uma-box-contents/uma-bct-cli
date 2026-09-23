"""Run a batch and report what happened.

One bad image does not stop the rest. Errors are saved in the run result, and
the terminal only gets a short summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import signal
import sys
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .adapters.ollama_client import timeout_from_environment
from .atomic import rename_new_file
from .bootstrap import build_pipeline
from .export import export_catalogue
from .path_safety import check_path
from .pipeline import ImageResult, warning_text
from .run_artifacts import (
    CatalogueLock,
    backup_existing_results,
    failed_manifest_path,
    regular_file_sha256,
    replace_backed_file,
    rollback_owned_file,
    run_directory,
    write_json_atomic,
)
from .terminal import TerminalProgress, print_run_summary


# Keep the export state behavior together.
@dataclass
class ExportState:
    record_count: int = 0
    catalogue_output: str | None = None
    previous_results: tuple[Path, ...] = ()
    previous_catalogue: Path | None = None
    previous_manifest: Path | None = None
    catalogue_digest: str | None = None
    replace_manifest: bool = False
    wrote_catalogue: bool = False


@contextmanager
def direct_termination_handlers() -> Iterator[None]:
    """Turn termination signals into cleanup-aware exit codes."""

    watched: list[tuple[signal.Signals, Any]] = []

    # Handle the terminate step.
    def terminate(signum: int, _frame: Any) -> None:
        if signum == getattr(signal, "SIGTERM", object()):
            raise SystemExit(143)
        if signum == getattr(signal, "SIGHUP", object()):
            raise SystemExit(129)
        raise KeyboardInterrupt

    for name in ("SIGTERM", "SIGHUP", "SIGBREAK"):
        signum = getattr(signal, name, None)
        if signum is None:
            continue
        try:
            previous = signal.signal(signum, terminate)
        except (OSError, ValueError):
            continue
        watched.append((signum, previous))
    try:
        yield
    finally:
        for signum, previous in watched:
            signal.signal(signum, previous)


# Handle the configuration step.
def configuration(arguments: argparse.Namespace, work_directory: Path | None, request_timeout_seconds: float | None
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


# Process the images.
def process_images(pipeline: Any, images: list[Path], failures: list[dict[str, str]],
    progress: TerminalProgress) -> list[ImageResult]:
    results: list[ImageResult] = []
    for index, image_path in enumerate(images, start=1):
        progress.photo_started(index, image_path)
        try:
            result = pipeline.process(image_path)
        except Exception as error:
            message = warning_text(error)
            failures.append({"source_image": str(image_path), "error": message})
            print(f"Failed {ascii(str(image_path))}: {message}", file=sys.stderr)
            progress.photo_finished(index, "failed", 0)
            continue
        results.append(result)
        for warning in result.warnings:
            print(f"Warning for {ascii(str(image_path))}: {warning_text(warning)}", file=sys.stderr)
        progress.photo_finished(index, result.status, len(result.records))
    return results


# Build the public summary.
def public_summary(summary: dict[str, Any], *, image_count: int, model_name: str) -> dict[str, Any]:
    images = summary["images"]
    failed_results = sum(item["status"] == "failed" for item in images)
    source_failures = sum("source_image" in item for item in summary["failures"])
    stage_failures = sum("source_image" not in item for item in summary["failures"])
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
        "failure_count": failed_results + source_failures + stage_failures,
        "model": model_name,
        "run_manifest": summary["run_manifest"],
        "previous_results": summary["previous_results"],
    }
    if recovery_path := summary.get("recovery_path"):
        result["recovery_path"] = recovery_path
    return result


# Prepare the pipeline.
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


# Export the results.
def export_results(arguments: argparse.Namespace, results: list[ImageResult], output_path: Path,
                   manifest_path: Path, failures: list[dict[str, str]]) -> ExportState:
    return export_results_into(
        arguments, results, output_path, manifest_path, failures, ExportState())


# Export results through the active transaction.
def export_results_into(arguments: argparse.Namespace, results: list[ImageResult], output_path: Path,
    manifest_path: Path, failures: list[dict[str, str]], state: ExportState) -> ExportState:

    if not sum(len(result.records) for result in results):
        return state
    previous: tuple[Path, ...] = ()
    previous_catalogue: Path | None = None
    staging = output_path.with_name(f".uma-bct-catalogue-publish-{uuid.uuid4().hex}.csv")
    publication_started = False
    try:
        # Check again because model processing can take a long time.
        if not arguments.force:
            for path in (output_path, manifest_path):
                check_path(path, "Output")
                if path.exists():
                    raise FileExistsError(f"Output already exists: {path}")
        records = export_catalogue(
            results,
            staging,
            minimum_confidence=arguments.min_transcription_confidence,
            replace_existing=False,
        )
        previous = backup_existing_results((output_path, manifest_path)) if arguments.force else ()
        previous_catalogue = next((path for path in previous if path.suffix.casefold() == ".csv"), None)
        previous_manifest = next((path for path in previous if path.name.endswith(".run.json")), None)
        replace_manifest = previous_manifest is not None
        state.previous_results = previous
        state.previous_catalogue = previous_catalogue
        state.previous_manifest = previous_manifest
        state.replace_manifest = replace_manifest
        publication_started = True
        state.record_count = len(records)
        state.catalogue_output = str(output_path)
        state.wrote_catalogue = True
        state.catalogue_digest = regular_file_sha256(staging, "Staged catalogue output")
        if previous_catalogue is not None:
            replace_backed_file(staging, output_path, previous_catalogue, "Catalogue output")
        else:
            rename_new_file(staging, output_path, existing_message=f"Catalogue output already exists: {output_path}")
        staging = None
        return state

    except BaseException as error:
        published_or_uncertain = publication_started and (staging is None or not staging.exists())
        if published_or_uncertain:
            try:
                rollback_exported_catalogue(output_path, state)
                state.record_count = 0
                state.catalogue_output = None
                state.wrote_catalogue = False
            except Exception as rollback_error:
                print(
                    f"Export rollback failed: {warning_text(rollback_error)}. Review {previous_catalogue or output_path}.",
                    file=sys.stderr,
                )
        if not isinstance(error, Exception):
            raise
        message = warning_text(error)
        failures.append({"stage": "export", "error": message})
        print(f"Export failed: {message}", file=sys.stderr)
        state.record_count = 0
        state.catalogue_output = None
        state.wrote_catalogue = False
        return state

    finally:
        if staging is not None:
            staging.unlink(missing_ok=True)


# Write the manifest.
def write_manifest(manifest: Path, summary: dict[str, Any], failures: list[dict[str, str]], *, replace_existing: bool) -> bool:
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


# Rollback the exported catalogue.
def rollback_exported_catalogue(output_path: Path, state: ExportState) -> None:
    if not state.wrote_catalogue or state.catalogue_digest is None:
        return
    rollback_owned_file(output_path, state.catalogue_digest, state.previous_catalogue, "catalogue")


# Write the backed manifest.
def write_backed_manifest(manifest: Path, summary: dict[str, Any], backup: Path, failures: list[dict[str, str]]) -> bool:
    staging = manifest.with_name(f".uma-bct-json-publish-{uuid.uuid4().hex}.tmp")
    try:
        write_json_atomic(staging, summary, replace_existing=False)
        replace_backed_file(staging, manifest, backup, "Run manifest")
    except BaseException as error:
        if not isinstance(error, Exception):
            raise
        message = warning_text(error)
        failures.append({"stage": "manifest", "error": message})
        summary["status"] = "partial" if summary["catalogue_output"] else "failed"
        summary["run_manifest"] = None
        print(f"Manifest write failed: {message}", file=sys.stderr)
        return False
    finally:
        staging.unlink(missing_ok=True)
    return True


# Calculate the summary digest.
def summary_digest(summary: dict[str, Any]) -> str:
    content = (json.dumps(summary, indent=2, ensure_ascii=False) + "\n").encode()
    return hashlib.sha256(content).hexdigest()


# Replace the manifest if unchanged.
def replace_manifest_if_unchanged(path: Path, previous: dict[str, Any], updated: dict[str, Any]) -> None:
    expected = path.with_name(f".uma-bct-json-expected-{uuid.uuid4().hex}.tmp")
    staging = path.with_name(f".uma-bct-json-publish-{uuid.uuid4().hex}.tmp")
    try:
        write_json_atomic(expected, previous, replace_existing=False)
        write_json_atomic(staging, updated, replace_existing=False)
        replace_backed_file(staging, path, expected, "Run manifest")
    finally:
        expected.unlink(missing_ok=True)
        staging.unlink(missing_ok=True)


# Build the emit summary.
def emit_summary(summary: dict[str, Any], image_count: int, model_name: str,
    progress: TerminalProgress, *, human: bool) -> None:
    progress.close()
    print_run_summary(
        public_summary(summary, image_count=image_count, model_name=model_name),
        human=human,
        elapsed_seconds=progress.elapsed_seconds(),
        attempted=progress.completed,
    )


# Build the lock failure summary.
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


# Recover the manifest failure.
def recover_manifest_failure(output_path: Path, summary: dict[str, Any], export_state: ExportState,
    failures: list[dict[str, str]]) -> None:
    try:
        rollback_exported_catalogue(output_path, export_state)
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


# Build the locked batch summary.
def locked_batch_summary(arguments: argparse.Namespace, images: list[Path],
    output_path: Path, normal_manifest_path: Path, *,
    preserve_existing_manifest_on_failure: bool, progress: TerminalProgress) -> tuple[dict[str, Any], bool]:

    failures: list[dict[str, str]] = []
    progress.phase("Checking the local model...")
    pipeline, work_directory, request_timeout, model_metadata = prepare_pipeline(arguments, failures)
    results = process_images(pipeline, images, failures, progress) if pipeline else []
    progress.phase("Finalizing results..." if pipeline else "Saving run details...")
    export_state = ExportState()
    manifest: Path | None = None
    summary: dict[str, Any] | None = None
    try:
        if pipeline:
            export_results_into(arguments, results, output_path, normal_manifest_path, failures, export_state)

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
        if export_state.previous_manifest is not None:
            manifest_written = write_backed_manifest(
                manifest, summary, export_state.previous_manifest, failures
            )
        else:
            manifest_written = write_manifest(manifest, summary, failures, replace_existing=export_state.replace_manifest)
    except BaseException:
        if export_state.wrote_catalogue:
            try:
                rollback_exported_catalogue(output_path, export_state)
            except Exception as error:
                print(
                    f"Rollback after interruption failed: {warning_text(error)}. "
                    f"Review {export_state.previous_catalogue or output_path}.",
                    file=sys.stderr,
                )
        if manifest is not None and summary is not None:
            previous_manifest = export_state.previous_manifest if manifest == normal_manifest_path else None
            try:
                rollback_owned_file(manifest, summary_digest(summary), previous_manifest, "run manifest")
            except Exception as error:
                print(
                    f"Manifest rollback after interruption failed: {warning_text(error)}. Review {manifest}.",
                    file=sys.stderr,
                )
        raise
    if not manifest_written and export_state.wrote_catalogue:
        recover_manifest_failure(output_path, summary, export_state, failures)
    return summary, pipeline is not None


def run_batch(arguments: argparse.Namespace, images: list[Path], output_path: Path, normal_manifest_path: Path, *,
              preserve_existing_manifest_on_failure: bool) -> int:
    """Process a batch and report it in JSON or readable terminal text."""

    with TerminalProgress(len(images)) as progress:
        return _run_batch(
            arguments, images, output_path, normal_manifest_path,
            preserve_existing_manifest_on_failure=preserve_existing_manifest_on_failure,
            progress=progress,
        )


def _run_batch(arguments: argparse.Namespace, images: list[Path], output_path: Path, normal_manifest_path: Path, *,
               preserve_existing_manifest_on_failure: bool, progress: TerminalProgress) -> int:
    """Keep the batch's existing lock and publication behavior together."""

    lock = CatalogueLock(output_path)
    try:
        lock.acquire()
    except Exception as error:
        summary = lock_failure_summary(arguments, error)
        emit_summary(summary, len(images), arguments.model, progress, human=getattr(arguments, "human", False))
        return 2

    release_error: str | None = None
    try:
        with direct_termination_handlers():
            summary, pipeline_ready = locked_batch_summary(
                arguments,
                images,
                output_path,
                normal_manifest_path,
                preserve_existing_manifest_on_failure=preserve_existing_manifest_on_failure,
                progress=progress,
            )
    finally:
        try:
            lock.release()
        except Exception as error:
            release_error = warning_text(error)

    if release_error is not None:
        previous_summary = json.loads(json.dumps(summary))
        summary["failures"].append({"stage": "lock_cleanup", "error": release_error})
        summary["status"] = "partial" if summary["catalogue_output"] else "failed"
        print(f"Lock cleanup failed: {release_error}", file=sys.stderr)
        authoritative_manifest = Path(summary["run_manifest"]) if summary.get("run_manifest") else None
        if authoritative_manifest is not None:
            try:
                replace_manifest_if_unchanged(authoritative_manifest, previous_summary, summary)
            except Exception as error:
                message = warning_text(error)
                summary["failures"].append({"stage": "manifest", "error": message})
                summary["run_manifest"] = None
                print(f"Manifest was not updated after lock cleanup failed: {message}", file=sys.stderr)

    emit_summary(summary, len(images), arguments.model, progress, human=getattr(arguments, "human", False))
    if release_error is not None:
        return 2
    if not pipeline_ready:
        return 2
    return 0 if summary["status"] == "complete" else 1
