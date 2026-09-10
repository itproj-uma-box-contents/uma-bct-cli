"""Batch orchestration and compact result reporting.

``run_batch`` runs the whole CLI batch as a sequence of stages that each fail
independently rather than aborting the run: preflight (model/pipeline
setup), per-image processing, catalogue export, and manifest write each
record their own failures into a shared ``failures`` list instead of
raising, so one failed image or a failed export still allows every other
stage to report as much as it could complete. The overall ``status`` in the
emitted summary is therefore derived at the end from whether any failures or
incomplete images occurred, not tracked as the run progresses.
``public_summary`` intentionally re-shapes the full internal summary into a
smaller public one (counts instead of full per-image detail) for the
console; the full detail is only ever written to the run manifest file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .adapters.ollama_client import timeout_from_environment
from .bootstrap import build_pipeline
from .export import export_catalogue
from .path_safety import validate_path_hierarchy
from .pipeline import ImageResult, warning_text
from .run_artifacts import backup_existing_results, failed_manifest_path, run_directory, write_json_atomic


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
    return {
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


def prepare_pipeline(
    arguments: argparse.Namespace, failures: list[dict[str, str]]
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


def export_results(
    arguments: argparse.Namespace,
    results: list[ImageResult],
    output_path: Path,
    manifest_path: Path,
    failures: list[dict[str, str]],
) -> tuple[int, str | None, tuple[Path, ...]]:
    if not sum(len(result.records) for result in results):
        failures.append({"stage": "processing", "error": "No usable catalogue records were produced."})
        return 0, None, ()
    previous: tuple[Path, ...] = ()
    try:
        if not arguments.force:
            for path in (output_path, manifest_path):
                validate_path_hierarchy(path, "Output")
                if path.exists():
                    raise FileExistsError(f"Output already exists: {path}")
        previous = backup_existing_results((output_path, manifest_path)) if arguments.force else ()
        records = export_catalogue(
            results, output_path, minimum_confidence=arguments.min_transcription_confidence, replace_existing=arguments.force
        )
        return len(records), str(output_path), previous
    except Exception as error:
        message = warning_text(error)
        failures.append({"stage": "export", "error": message})
        print(f"Export failed: {message}", file=sys.stderr)
        return 0, None, previous


def write_manifest(
    manifest: Path, summary: dict[str, Any], failures: list[dict[str, str]], *, replace_existing: bool
) -> None:
    try:
        write_json_atomic(manifest, summary, replace_existing=replace_existing)
    except Exception as error:
        message = warning_text(error)
        failures.append({"stage": "manifest", "error": message})
        summary["status"] = "partial" if summary["catalogue_output"] else "failed"
        summary["run_manifest"] = None
        print(f"Manifest write failed: {message}", file=sys.stderr)


def emit_summary(summary: dict[str, Any], image_count: int, model_name: str) -> None:
    print(json.dumps(public_summary(summary, image_count=image_count, model_name=model_name), indent=2, ensure_ascii=False))


def run_batch(
    arguments: argparse.Namespace,
    images: list[Path],
    output_path: Path,
    normal_manifest_path: Path,
    *,
    preserve_existing_manifest_on_failure: bool,
) -> int:
    """Process one batch, write safe outputs, and emit one JSON summary."""

    failures: list[dict[str, str]] = []
    pipeline, work_directory, request_timeout, model_metadata = prepare_pipeline(arguments, failures)
    results = process_images(pipeline, images, failures) if pipeline else []
    if pipeline:
        record_count, catalogue_output, previous_results = export_results(
            arguments, results, output_path, normal_manifest_path, failures
        )
    else:
        record_count, catalogue_output, previous_results = 0, None, ()

    incomplete = any(result.status != "complete" for result in results)
    status = "complete"
    if failures or incomplete:
        status = "partial" if record_count else "failed"

    manifest = (
        failed_manifest_path(output_path)
        if preserve_existing_manifest_on_failure and not record_count
        else normal_manifest_path
    )
    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": status,
        "configuration": configuration(arguments, work_directory, request_timeout),
        "model": model_metadata,
        "images": [result.to_dict() for result in results],
        "failures": failures,
        "catalogue_output": catalogue_output,
        "catalogue_records": record_count,
        "run_manifest": str(manifest),
        "previous_results": [str(path) for path in previous_results],
    }
    write_manifest(manifest, summary, failures, replace_existing=arguments.force)
    emit_summary(summary, len(images), arguments.model)
    if pipeline is None:
        return 2
    return 0 if summary["status"] == "complete" else 1
