"""Command-line entry point for the minimal modular pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from constants import IMAGE_EXTENSIONS

from .bootstrap import DETECTORS, TRANSCRIBERS, build_pipeline
from .export import SUPPORTED_OUTPUT_EXTENSIONS, export_catalogue
from .pipeline import ImageResult


def confidence_value(value: str) -> float:
    try:
        confidence = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number between 0 and 1") from error
    if not 0.0 <= confidence <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return confidence


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect and transcribe UMA archive boxes with local components."
    )
    parser.add_argument(
        "-i",
        "--path-to-inputs",
        type=Path,
        required=True,
        help="an input image or a directory of images",
    )
    parser.add_argument(
        "-o",
        "--path-to-output",
        type=Path,
        required=True,
        help="catalogue output path ending in .csv or .xlsx",
    )
    parser.add_argument(
        "--detector",
        choices=tuple(DETECTORS),
        default="qwen",
        help="box detector to use (default: %(default)s)",
    )
    parser.add_argument(
        "--transcriber",
        choices=tuple(TRANSCRIBERS),
        default="qwen",
        help="text transcriber to use (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        default="qwen3-vl",
        help="locally installed Ollama model tag (default: %(default)s)",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("outputs") / "crops",
        help="directory for full-resolution box crops (default: %(default)s)",
    )
    parser.add_argument(
        "--min-confidence",
        type=confidence_value,
        default=0.60,
        metavar="FLOAT",
        help="minimum detector confidence from 0 to 1 (default: %(default).2f)",
    )
    return parser.parse_args(argv)


def discover_images(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.casefold() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {path.suffix or '<none>'}")
        return [path]
    if not path.is_dir():
        raise ValueError(f"Input path does not exist: {path}")
    return sorted(
        (
            candidate
            for candidate in path.iterdir()
            if candidate.is_file()
            and candidate.suffix.casefold() in IMAGE_EXTENSIONS
        ),
        key=lambda candidate: candidate.name.casefold(),
    )


def main(argv: list[str] | None = None) -> int:
    arguments = parse_args(argv)
    if arguments.path_to_output.suffix.casefold() not in SUPPORTED_OUTPUT_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_OUTPUT_EXTENSIONS))
        raise SystemExit(f"Output path must end in one of: {supported}")
    try:
        images = discover_images(arguments.path_to_inputs)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if not images:
        raise SystemExit(f"No supported images found in: {arguments.path_to_inputs}")

    pipeline = build_pipeline(
        detector_name=arguments.detector,
        transcriber_name=arguments.transcriber,
        model_name=arguments.model,
        work_directory=arguments.work_dir,
        minimum_detector_confidence=arguments.min_confidence,
    )
    results: list[ImageResult] = []
    failures: list[dict[str, str]] = []
    for image_path in images:
        try:
            results.append(pipeline.process(image_path))
        except Exception as error:  # Continue a batch after one bad image/model call.
            failures.append({"source_image": str(image_path), "error": str(error)})

    records = export_catalogue(results, arguments.path_to_output)
    print(
        json.dumps(
            {
                "images": [result.to_dict() for result in results],
                "failures": failures,
                "catalogue_output": str(arguments.path_to_output),
                "catalogue_records": len(records),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 1 if failures else 0
