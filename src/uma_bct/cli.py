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
from .constants import DEFAULT_MIN_DETECTOR_CONFIDENCE, DEFAULT_MIN_TRANSCRIPTION_CONFIDENCE, IMAGE_EXTENSIONS
from .keep_awake import keep_awake
from .managed_batch import run_managed_batch
from .model_config import hosted_model_from_environment
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
        metavar="PATH",
        help="read one photo or a folder of photos; subfolders are not scanned (default: %(default)s)",
    )
    parser.add_argument(
        "-o",
        "--path-to-outputs",
        "--path-to-output",
        dest="path_to_outputs",
        type=Path,
        default=None,
        metavar="PATH",
        help="direct catalogue .csv or .xlsx filename (default: outputs/catalogue.csv, or outputs/catalogue.xlsx for Excel)",
    )
    parser.add_argument(
        "--format", "--file-type", dest="file_type", type=str.casefold, choices=("excel", "csv"), default=None,
        help="save CSV or Excel .xlsx; names ignore case (default: CSV, or the output filename's extension)",
    )
    parser.add_argument(
        "-b",
        "--path-to-box-images",
        "--work-dir",
        dest="path_to_box_images",
        type=Path,
        default=None,
        metavar="PATH",
        help="direct output folder for cropped box images (default: crops beside the catalogue)",
    )
    parser.add_argument(
        "--results-directory",
        type=Path,
        default=None,
        metavar="PATH",
        help="save resumable Batch folders here; cannot be combined with -o, -b, or --force",
    )


def add_processing_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--verification", choices=MODES, default=None,
                        help="compare a blind second reading: never (default), always, or ask (interactive only)")
    parser.add_argument(
        "--hosted", action="store_true", help="use a hosted model configured by UMA_BCT_API_KEY and UMA_BCT_API_MODEL"
    )
    parser.add_argument(
        "-d", "--detector", choices=tuple(DETECTORS), default="qwen",
        help="find boxes with the selected model (qwen) or experimental OpenCV (default: %(default)s)",
    )
    parser.add_argument(
        "-t", "--transcriber", choices=tuple(TRANSCRIBERS), default="qwen",
        help="read text with the selected local or hosted model; qwen is the only supported mode",
    )
    parser.add_argument(
        "--ai-dates", action="store_true",
        help="ask the selected model for dates in transcribed handwriting; adds requests (default: local text rules)",
    )
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL,
                        help="local Ollama model (default: %(default)s); hosted runs use UMA_BCT_API_MODEL")
    parser.add_argument(
        "-c",
        "--min-transcription-confidence",
        dest="min_transcription_confidence",
        type=bounded_percentage,
        default=DEFAULT_MIN_TRANSCRIPTION_CONFIDENCE,
        metavar="PERCENT",
        help="flag scores below this value; keep all rows (0-100; default: %(default)s)",
    )
    parser.add_argument(
        "--min-confidence",
        "--min-detector-confidence",
        dest="min_detector_confidence",
        type=bounded_float,
        default=DEFAULT_MIN_DETECTOR_CONFIDENCE,
        metavar="SCORE",
        help="minimum OpenCV detection score, including fallback detection (0-1; default: %(default).2f)",
    )
    parser.add_argument(
        "--request-timeout",
        type=positive_float,
        default=None,
        metavar="SECONDS",
        help=(f"positive timeout per model request, not the whole run; retries can add time "
              f"(default: {TIMEOUT_ENVIRONMENT_VARIABLE} or {DEFAULT_REQUEST_TIMEOUT_SECONDS:g}s)"),
    )
    parser.add_argument("-f", "--force", action="store_true",
                        help="replace direct output files after keeping backups; not for managed batches")
    parser.add_argument(
        "--no-missing-box",
        "--no-missing-box-detection",
        dest="detect_missing_boxes",
        action="store_false",
        help="omit suggested rows for gaps in unit numbers; keep detected boxes",
    )
    parser.set_defaults(detect_missing_boxes=True)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = SafeArgumentParser(
        prog="uma-bct",
        description=("Turn archive-box photos into a CSV or Excel catalogue for human review.\n"
                     "Local Ollama is the default; --hosted opts into a hosted model."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uma-bct setup\n"
            "  uma-bct --format excel\n"
            "  uma-bct --results-directory Results --ai-dates\n"
            "  uma-bct --results-directory Results --format excel --reprocess\n\n"
            "Add photos to inputs/ before processing. Options apply to new work;\n"
            "completed batches keep their saved results unless you use --reprocess.\n"
            "An unfinished batch resumes first and needs its original options."
        ),
        allow_abbrev=False,
    )
    add_path_arguments(parser)
    add_processing_arguments(parser)
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="create a separate batch for processed photos; requires --results-directory; unfinished batches resume first",
    )
    output_format = parser.add_mutually_exclusive_group()
    output_format.add_argument(
        "--json", action="store_true",
        help="write one run summary as JSON; progress goes to stderr; never prompt; use --format for the catalogue",
    )
    output_format.add_argument("--human", action="store_true", help="show the readable run summary (default)")
    parser.add_argument("--version", action="version", version=package_version())
    subcommands = parser.add_subparsers(dest="command", metavar="COMMAND")
    setup = subcommands.add_parser(
        "setup", help="prepare folders and check or download the local model", allow_abbrev=False,
        description="Prepare folders and check or download the local model. Always returns a JSON summary.",
        epilog="Put setup first: uma-bct setup --path-to-inputs Photos --results-directory Results",
    )
    setup.add_argument("--hosted", action="store_true", help="check the configured hosted model instead of local Ollama")
    setup.add_argument("-m", "--model", default=DEFAULT_MODEL, help="local model to prepare (default: %(default)s)")
    setup.add_argument("--path-to-inputs", type=Path, default=None, metavar="PATH",
                       help="photo folder to prepare (default: inputs/ in the current directory)")
    setup.add_argument("--results-directory", type=Path, default=None, metavar="PATH",
                       help="managed results folder to prepare (default: outputs/ in the current directory)")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_argument_parser()
    arguments = parser.parse_args(argv)
    if arguments.command != "setup":
        resolve_output_arguments(arguments, parser)
    return arguments


def resolve_output_arguments(arguments: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    suffix = ".xlsx" if arguments.file_type == "excel" else ".csv"
    if arguments.path_to_outputs is None:
        arguments.path_to_outputs = Path("outputs") / f"catalogue{suffix}"
    else:
        output_suffix = arguments.path_to_outputs.suffix.casefold()
        if output_suffix not in {".csv", ".xlsx"}:
            parser.error("output path must end in .xlsx or .csv")
        if arguments.file_type is not None and output_suffix != suffix:
            parser.error("--format/--file-type conflicts with the output filename's extension")


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
    hosted_model = None
    if arguments.hosted:
        try:
            hosted_model = hosted_model_from_environment()
        except ValueError as error:
            parser.error(str(error))
        if hosted_model is None:
            parser.error("UMA_BCT_API_KEY is required with --hosted.")
        from .adapters.hosted_client import HostedVisionModel

        try:
            HostedVisionModel(hosted_model)
        except ValueError as error:
            parser.error(str(error))
        if option_was_supplied(raw_arguments, ("-m", "--model")):
            parser.error("--model selects local Ollama; use UMA_BCT_API_MODEL for hosted processing.")
    arguments.hosted_model = hosted_model
    if arguments.command == "setup":
        if raw_arguments[0] != "setup":
            parser.error("setup must be the first argument")
        setup_arguments: dict[str, object] = {}
        # ZIP processing and its first-run setup must select the same model.
        # Omitted options retain the existing pinned-model setup behavior.
        if option_was_supplied(raw_arguments, ("-m", "--model")):
            setup_arguments["model_name"] = arguments.model
        if arguments.path_to_inputs is not None:
            setup_arguments["input_directory"] = arguments.path_to_inputs
        if arguments.results_directory is not None:
            setup_arguments["results_directory"] = arguments.results_directory
        # Keep the computer active while the local model is checked or downloaded.
        with keep_awake():
            if hosted_model is not None:
                setup_arguments["hosted_model"] = hosted_model
            return run_setup(**setup_arguments)

    if arguments.reprocess and arguments.results_directory is None:
        parser.error("--reprocess requires --results-directory")
    resolve_output_arguments(arguments, parser)
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
    crop_root = arguments.path_to_box_images or output_path.parent / "crops"
    arguments.path_to_box_images = crop_root
    run_manifest = manifest_path(output_path)
    output_paths = (output_path, run_manifest)
    # Check every output before reading photos or starting the model.
    try:
        validate_output_target(output_path, directory=False, purpose="Catalogue output")
        validate_output_target(run_manifest, directory=False, purpose="Run manifest")
        validate_output_target(crop_root, directory=True, purpose="Review-crop root")
        validate_output_layout(output_path, run_manifest, crop_root)
        images = discover_images(arguments.path_to_inputs)
    except ValueError as error:
        parser.error(str(error))
    if not images:
        parser.error(f"no supported images found in {arguments.path_to_inputs}")

    existing = next((path for path in output_paths if path.exists()), None)
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
