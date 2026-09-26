"""Read CLI options and start a transcription run.

Input and output paths are checked before model work starts. Linked folders,
unsafe output layouts, and unsupported files are rejected early.
"""

from __future__ import annotations

import argparse
import json
import math
import ntpath
import os
import stat
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .adapters.ollama_client import DEFAULT_MODEL, DEFAULT_REQUEST_TIMEOUT_SECONDS, TIMEOUT_ENVIRONMENT_VARIABLE
from .batch_runner import run_batch
from .bootstrap import DETECTORS, TRANSCRIBERS
from .constants import IMAGE_EXTENSIONS
from .keep_awake import keep_awake
from .managed_batch import run_managed_batch
from .path_safety import check_path, redirected_path_kind
from .pipeline import warning_text
from .run_artifacts import manifest_path
from .setup_command import run_setup
from .verification import MODES, configured_mode


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
    parser.add_argument(
        "--results-directory",
        type=Path,
        default=None,
        help="create or resume a readable managed batch in this directory",
    )


def add_processing_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--verification", choices=MODES, default=None,
                        help="second reading: ask, always, or never (default: saved preference or ask)")
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
    parser.add_argument(
        "--no-missing-box",
        "--no-missing-box-detection",
        dest="detect_missing_boxes",
        action="store_false",
        help="do not insert rows for missing unit numbers",
    )
    parser.set_defaults(detect_missing_boxes=True)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = SafeArgumentParser(
        prog="uma-bct",
        description="Detect and transcribe UMA archive boxes using local components.",
        epilog=("Shortest workflow: run 'uma-bct setup', add photos to inputs/, then run 'uma-bct'."),
    )
    add_path_arguments(parser)
    add_processing_arguments(parser)
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="create another managed batch for a photo set that is already complete",
    )
    output_format = parser.add_mutually_exclusive_group()
    output_format.add_argument("--json", action="store_true", help="write the final transcription result as JSON")
    output_format.add_argument("--human", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--version", action="version", version=package_version())
    subcommands = parser.add_subparsers(dest="command", metavar="COMMAND")
    setup = subcommands.add_parser("setup", help="prepare folders and the pinned model")
    setup.add_argument("--path-to-inputs", type=Path, default=None, help="directory to prepare for input photos")
    setup.add_argument("--results-directory", type=Path, default=None, help="directory to prepare for managed results")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_argument_parser().parse_args(argv)


def path_metadata(path: Path, purpose: str) -> os.stat_result:
    try:
        check_path(path, purpose)
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


def discover_images(path: Path, *, check_readable: bool = True) -> list[Path]:
    metadata = path_metadata(path, "Input path")
    if stat.S_ISREG(metadata.st_mode):
        if path.suffix.casefold() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {path.suffix or '<none>'}")
        if check_readable:
            ensure_readable(path)
        return [path]
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"Input path is not a file or directory: {path}")
    try:
        candidates = tuple(path.iterdir())
    except OSError as error:
        raise ValueError(f"Input directory is not readable: {path}: {error}") from error

    # Sort only after validating each direct child. Stable photo order also
    # makes managed checkpoints and later human review predictable.
    images: list[Path] = []
    for candidate in candidates:
        if candidate.suffix.casefold() not in IMAGE_EXTENSIONS:
            continue
        metadata = path_metadata(candidate, "Input image")
        if stat.S_ISREG(metadata.st_mode):
            if check_readable:
                ensure_readable(candidate)
            images.append(candidate)
    return sorted(images, key=lambda item: (item.name.casefold(), item.name))


def validate_output_target(path: Path, *, directory: bool, purpose: str) -> None:
    check_path(path, purpose)

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


def validate_output_layout(catalogue: str | Path, manifest: str | Path, crop_root: str | Path) -> None:
    """Keep crop directories separate from published files on one Windows drive."""

    raw_paths = tuple(os.fspath(path) for path in (catalogue, manifest, crop_root))
    absolute_paths = tuple(os.path.realpath(path) for path in raw_paths)
    windows_drives = {drive.casefold() for path in (*raw_paths, *absolute_paths) if (drive := ntpath.splitdrive(path)[0])}
    if len(windows_drives) > 1:
        raise ValueError("catalogue, manifest, and review-crop paths must use the same Windows drive")

    catalogue_absolute, manifest_absolute, crop_absolute = absolute_paths
    for artifact, purpose in ((catalogue_absolute, "catalogue output"), (manifest_absolute, "run manifest")):
        try:
            common = os.path.commonpath((artifact, crop_absolute))
        except ValueError:
            raise ValueError("catalogue, manifest, and review-crop paths must use the same Windows drive") from None
        common = os.path.normcase(os.path.normpath(common))
        artifact_normalized = os.path.normcase(os.path.normpath(artifact))
        crop_normalized = os.path.normcase(os.path.normpath(crop_absolute))
        if common == crop_normalized:
            if purpose == "catalogue output":
                raise ValueError("catalogue output must not be inside the review-crop directory")
            raise ValueError("run manifest must not be inside the review-crop directory")
        if common == artifact_normalized:
            raise ValueError(f"review-crop root must not be inside the {purpose} path")


def option_was_supplied(arguments: list[str], names: tuple[str, ...]) -> bool:
    """Check raw arguments because argparse cannot tell an explicit default from an omitted option."""

    for argument in arguments:
        if argument in names:
            return True
        for name in names:
            if name.startswith("--") and argument.startswith(f"{name}="):
                return True
            if len(name) == 2 and argument.startswith(name) and len(argument) > len(name):
                return True
    return False


def validate_managed_layout(input_path: Path, results_directory: Path) -> None:
    input_absolute = os.path.realpath(input_path)
    results_absolute = os.path.realpath(results_directory)
    try:
        common = os.path.commonpath((input_absolute, results_absolute))
    except ValueError:
        return
    if common == results_absolute or (input_path.is_dir() and common == input_absolute):
        raise ValueError("input photos and managed Results must be in separate, non-nested directories")


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    raw_arguments = list(sys.argv[1:] if argv is None else argv)
    arguments = parser.parse_args(raw_arguments)
    if arguments.command == "setup":
        if raw_arguments[0] != "setup":
            parser.error("setup must be the first argument")
        setup_arguments: dict[str, Path] = {}
        if arguments.path_to_inputs is not None:
            setup_arguments["input_directory"] = arguments.path_to_inputs
        if arguments.results_directory is not None:
            setup_arguments["results_directory"] = arguments.results_directory
        # Keep the computer active while the local model is checked or downloaded.
        with keep_awake():
            return run_setup(**setup_arguments)

    if arguments.reprocess and arguments.results_directory is None:
        parser.error("--reprocess requires --results-directory")
    try:
        configured_mode(arguments)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    if arguments.results_directory is not None:
        # Managed batches own their catalogue and crop paths. Mixing direct
        # output flags with that layout would make resume ambiguous.
        conflicting = (
            option_was_supplied(raw_arguments, ("-o", "--path-to-outputs", "--path-to-output"))
            or option_was_supplied(raw_arguments, ("-b", "--path-to-box-images", "--work-dir"))
            or option_was_supplied(raw_arguments, ("-f", "--force"))
        )
        if conflicting:
            parser.error("--results-directory cannot be combined with -o, -b, or --force")
        try:
            validate_output_target(arguments.results_directory, directory=True, purpose="Results directory")
            images = discover_images(arguments.path_to_inputs, check_readable=False)
            validate_managed_layout(arguments.path_to_inputs, arguments.results_directory)
        except ValueError as error:
            parser.error(str(error))
        if not images:
            parser.error(f"no supported images found in {arguments.path_to_inputs}")
        # Resume and publication are part of the same protected CLI job.
        with keep_awake():
            return run_managed_batch(arguments, images, arguments.results_directory)

    output_path = arguments.path_to_outputs
    if output_path.suffix.casefold() != ".csv":
        parser.error("output path must end in .csv")
    crop_root = arguments.path_to_box_images or output_path.parent / "crops"
    arguments.path_to_box_images = crop_root
    run_manifest = manifest_path(output_path)
    review_path = output_path.with_suffix(".review.html")
    # Check every output before reading photos or starting the model.
    try:
        validate_output_target(output_path, directory=False, purpose="Catalogue output")
        validate_output_target(run_manifest, directory=False, purpose="Run manifest")
        validate_output_target(review_path, directory=False, purpose="Review output")
        validate_output_target(crop_root, directory=True, purpose="Review-crop root")
        validate_output_layout(output_path, run_manifest, crop_root)
        validate_output_layout(review_path, run_manifest, crop_root)
        images = discover_images(arguments.path_to_inputs)
    except ValueError as error:
        parser.error(str(error))
    if not images:
        parser.error(f"no supported images found in {arguments.path_to_inputs}")

    existing = next((path for path in (output_path, run_manifest, review_path) if path.exists()), None)
    if existing is not None and not arguments.force:
        parser.error(f"output already exists; pass --force to replace it: {existing}")
    try:
        with keep_awake():
            return run_batch(
                arguments,
                images,
                output_path,
                run_manifest,
                preserve_existing_manifest_on_failure=bool(arguments.force and existing),
            )
    except KeyboardInterrupt:
        exit_code = 130
    except SystemExit as error:
        if error.code not in {129, 143}:
            raise
        exit_code = error.code
    # Batch cleanup has already run. Do not imply that a new catalogue was
    # committed, or fabricate per-photo counts after an interrupted direct run.
    message = "Run interrupted. Check stderr for recovery paths; use --results-directory for resumable batches."
    if arguments.json:
        print(json.dumps({"schema_version": 1, "status": "interrupted", "exit_code": exit_code,
                          "catalogue_output": None, "message": message}))
    else:
        print(message, file=sys.stderr)
    return exit_code
