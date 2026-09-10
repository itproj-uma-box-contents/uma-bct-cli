"""Command-line entry point for archive-box transcription.

Most of this module's complexity is defensive path validation rather than
argument parsing: ``discover_images``, ``validate_output_target``, and the
checks in ``main`` all call ``path_safety.redirected_path_kind`` to reject
symlinks and directory junctions before touching a path, since this tool
handles private archive photographs and a redirected input/output path could
otherwise be used to read or write outside the intended directories. ``main``
also rejects a review-crop root that is the same as, or a parent of, the
catalogue output path, so a run could never accidentally treat generated
crops as catalogue input on a later run. ``SafeArgumentParser`` exists only
to route argparse's own error messages through ``warning_text``, keeping
error output single-line and stderr-safe.
"""

from __future__ import annotations

import argparse
import math
import os
import stat
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .adapters.ollama_client import DEFAULT_MODEL, DEFAULT_REQUEST_TIMEOUT_SECONDS, TIMEOUT_ENVIRONMENT_VARIABLE
from .batch_runner import run_batch
from .bootstrap import DETECTORS, TRANSCRIBERS
from .constants import IMAGE_EXTENSIONS
from .path_safety import redirected_path_kind, validate_path_hierarchy
from .pipeline import warning_text
from .run_artifacts import manifest_path
from .setup_command import run_setup


class SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        super().error(warning_text(message))


def package_version() -> str:
    try:
        return version("uma-bct-cli")
    except PackageNotFoundError:
        return "0.1.0"


def bounded_float(value: str) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be between 0 and 1") from error
    if not 0 <= result <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return result


def bounded_percentage(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be between 0 and 100") from error
    if not 0 <= result <= 100:
        raise argparse.ArgumentTypeError("must be between 0 and 100")
    return result


def positive_float(value: str) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive number") from error
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return result


def add_path_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-i",
        "--path-to-inputs",
        type=Path,
        default=Path("inputs"),
        help="input image or flat image directory (default: %(default)s)",
    )
    parser.add_argument(
        "-o",
        "--path-to-outputs",
        "--path-to-output",
        dest="path_to_outputs",
        type=Path,
        default=Path("outputs/catalogue.csv"),
        help="catalogue CSV path (default: %(default)s)",
    )
    parser.add_argument(
        "-b",
        "--path-to-box-images",
        "--work-dir",
        dest="path_to_box_images",
        type=Path,
        default=None,
        help="review-crop root (default: crops beside the catalogue)",
    )


def add_processing_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-d", "--detector", choices=tuple(DETECTORS), default="qwen", help="primary box detector (default: %(default)s)"
    )
    parser.add_argument(
        "-t", "--transcriber", choices=tuple(TRANSCRIBERS), default="qwen", help="text transcriber (default: %(default)s)"
    )
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "-c",
        "--min-transcription-confidence",
        dest="min_transcription_confidence",
        type=bounded_percentage,
        default=80,
        metavar="INTEGER",
        help="review threshold from 0 to 100 (default: %(default)s)",
    )
    parser.add_argument(
        "--min-confidence",
        "--min-detector-confidence",
        dest="min_detector_confidence",
        type=bounded_float,
        default=0.60,
        metavar="FLOAT",
        help="OpenCV threshold from 0 to 1 (default: %(default).2f)",
    )
    parser.add_argument(
        "--request-timeout",
        type=positive_float,
        default=None,
        metavar="SECONDS",
        help=(f"Ollama timeout; defaults to ${TIMEOUT_ENVIRONMENT_VARIABLE} or {DEFAULT_REQUEST_TIMEOUT_SECONDS:g}s"),
    )
    parser.add_argument("-f", "--force", action="store_true", help="replace an existing catalogue")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = SafeArgumentParser(
        prog="uma-bct",
        description="Detect and transcribe UMA archive boxes using local components.",
        epilog=("Shortest workflow: run 'uma-bct setup', add photos to inputs/, then run 'uma-bct'."),
    )
    add_path_arguments(parser)
    add_processing_arguments(parser)
    parser.add_argument("--version", action="version", version=package_version())
    subcommands = parser.add_subparsers(dest="command", metavar="COMMAND")
    subcommands.add_parser("setup", help="prepare folders and the pinned model")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_argument_parser().parse_args(argv)


def path_metadata(path: Path, purpose: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        raise ValueError(f"{purpose} does not exist: {path}") from None
    except OSError as error:
        raise ValueError(f"Could not inspect {purpose} {path}: {error}") from error
    if kind := redirected_path_kind(path, metadata):
        raise ValueError(f"{purpose} must not be a {kind}: {path}")
    return metadata


def ensure_readable(path: Path) -> None:
    try:
        with path.open("rb") as image:
            image.read(1)
    except OSError as error:
        raise ValueError(f"Input image is not readable: {path}: {error}") from error


def discover_images(path: Path) -> list[Path]:
    metadata = path_metadata(path, "Input path")
    if stat.S_ISREG(metadata.st_mode):
        if path.suffix.casefold() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {path.suffix or '<none>'}")
        ensure_readable(path)
        return [path]
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"Input path is not a file or directory: {path}")
    try:
        candidates = tuple(path.iterdir())
    except OSError as error:
        raise ValueError(f"Input directory is not readable: {path}: {error}") from error

    images: list[Path] = []
    for candidate in candidates:
        if candidate.suffix.casefold() not in IMAGE_EXTENSIONS:
            continue
        metadata = path_metadata(candidate, "Input image")
        if stat.S_ISREG(metadata.st_mode):
            ensure_readable(candidate)
            images.append(candidate)
    return sorted(images, key=lambda item: (item.name.casefold(), item.name))


def validate_output_target(path: Path, *, directory: bool, purpose: str) -> None:
    validate_path_hierarchy(path, purpose)

    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return
    except OSError as error:
        raise ValueError(f"Could not inspect {purpose} {path}: {error}") from error
    if kind := redirected_path_kind(path, metadata):
        raise ValueError(f"{purpose} must not be a {kind}: {path}")
    valid = stat.S_ISDIR(metadata.st_mode) if directory else stat.S_ISREG(metadata.st_mode)
    if not valid:
        expected = "directory" if directory else "regular file"
        raise ValueError(f"{purpose} must be a {expected}: {path}")


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    raw_arguments = list(sys.argv[1:] if argv is None else argv)
    arguments = parser.parse_args(raw_arguments)
    if arguments.command == "setup":
        if raw_arguments[0] != "setup":
            parser.error("setup must be the first argument")
        return run_setup()

    output_path = arguments.path_to_outputs
    if output_path.suffix.casefold() != ".csv":
        parser.error("output path must end in .csv")
    crop_root = arguments.path_to_box_images or output_path.parent / "crops"
    arguments.path_to_box_images = crop_root
    run_manifest = manifest_path(output_path)
    try:
        validate_output_target(output_path, directory=False, purpose="Catalogue output")
        validate_output_target(run_manifest, directory=False, purpose="Run manifest")
        validate_output_target(crop_root, directory=True, purpose="Review-crop root")
        output_absolute = Path(os.path.abspath(output_path))
        crop_absolute = Path(os.path.abspath(crop_root))
        if crop_absolute == output_absolute or crop_absolute in output_absolute.parents:
            raise ValueError("catalogue output must not be inside the review-crop directory")
        images = discover_images(arguments.path_to_inputs)
    except ValueError as error:
        parser.error(str(error))
    if not images:
        parser.error(f"no supported images found in {arguments.path_to_inputs}")

    existing = next((path for path in (output_path, run_manifest) if path.exists()), None)
    if existing is not None and not arguments.force:
        parser.error(f"output already exists; pass --force to replace it: {existing}")
    return run_batch(
        arguments,
        images,
        output_path,
        run_manifest,
        preserve_existing_manifest_on_failure=bool(arguments.force and existing),
    )
