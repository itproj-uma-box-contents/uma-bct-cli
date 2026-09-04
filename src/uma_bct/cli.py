"""Command-line entry point for the archive-box transcription pipeline."""

from __future__ import annotations

import argparse
import math
import os
import stat
import sys
import unicodedata
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .adapters.ollama_client import (
    DEFAULT_MODEL,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    TIMEOUT_ENVIRONMENT_VARIABLE,
    timeout_from_environment,
)
from .batch_runner import BatchOperations, run_batch
from .batch_runner import configuration as configuration
from .batch_runner import public_summary as public_summary
from .bootstrap import DETECTORS, TRANSCRIBERS, build_pipeline
from .constants import IMAGE_EXTENSIONS
from .export import SUPPORTED_OUTPUT_EXTENSIONS, export_catalogue
from .path_safety import (
    redirected_component,
    redirected_path_kind,
    redirected_paths_label,
)
from .pipeline import ImageResult as ImageResult
from .pipeline import warning_text
from .run_artifacts import GENERATED_STEM_MAX_BYTES as GENERATED_STEM_MAX_BYTES
from .run_artifacts import (
    acquire_output_lock,
    backup_existing_catalogue,
    failed_manifest_path,
    manifest_path,
    release_output_lock,
    restore_catalogue,
    run_directory,
    write_json_atomic,
)
from .run_artifacts import bounded_generated_stem as bounded_generated_stem
from .run_artifacts import output_lock_path as output_lock_path
from .setup_command import run_setup


class SafeArgumentParser(argparse.ArgumentParser):
    """Prevent untrusted paths and argument values from controlling the terminal."""

    def error(self, message: str) -> None:
        super().error(warning_text(ValueError(message)))


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
    parser = SafeArgumentParser(
        prog="uma-bct",
        description="Detect and transcribe UMA archive boxes using local components.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Shortest workflow:
  1. Run: uma-bct setup
  2. Add photos to inputs/
  3. Run: uma-bct

Progress and warnings go to stderr. The final JSON result goes to stdout.""",
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
        default=Path("outputs") / "catalogue.csv",
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
    subcommands = parser.add_subparsers(dest="command", metavar="COMMAND")
    subcommands.add_parser(
        "setup",
        help="create inputs/ and outputs/, then download the pinned model",
        description=(
            "Prepare the current directory and download the pinned Ollama model "
            "when it is not already installed."
        ),
    )
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
    if isinstance(path, Path):
        try:
            redirected = redirected_component(path)
        except OSError as error:
            raise ValueError(f"Could not inspect input path {path}: {error}") from error
        if redirected is not None:
            component, kind = redirected
            raise ValueError(
                f"Input {redirected_paths_label(kind)} are not accepted; select "
                f"the real image or directory path explicitly: {component}"
            )
    try:
        input_metadata = path.lstat()
        redirected_kind = redirected_path_kind(path, input_metadata)
    except FileNotFoundError:
        raise ValueError(f"Input path does not exist: {path}") from None
    except OSError as error:
        raise ValueError(f"Could not inspect input path {path}: {error}") from error
    if redirected_kind is not None:
        raise ValueError(
            f"Input {redirected_paths_label(redirected_kind)} are not accepted; "
            f"select the real image or directory path explicitly: {path}"
        )
    input_mode = input_metadata.st_mode
    if stat.S_ISREG(input_mode):
        if path.suffix.casefold() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {path.suffix or '<none>'}")
        ensure_readable_image(path)
        return [path]
    if not stat.S_ISDIR(input_mode):
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
            candidate_metadata = candidate.lstat()
            redirected_kind = redirected_path_kind(candidate, candidate_metadata)
        except OSError as error:
            raise ValueError(
                f"Could not inspect input path {candidate}: {error}"
            ) from error
        if redirected_kind is not None:
            raise ValueError(
                f"Input {redirected_paths_label(redirected_kind)} are not accepted; "
                f"select the real image path explicitly: {candidate}"
            )
        candidate_mode = candidate_metadata.st_mode
        if stat.S_ISREG(candidate_mode):
            ensure_readable_image(candidate)
            images.append(candidate)
    return sorted(
        images, key=lambda candidate: (candidate.name.casefold(), candidate.name)
    )


def validate_output_layout(
    output_path: Path,
    run_manifest_path: Path,
    crop_root: Path,
) -> None:
    """Reject layouts that cannot safely contain file and crop outputs."""

    normalized_output = Path(os.path.abspath(output_path))
    normalized_manifest = Path(os.path.abspath(run_manifest_path))
    normalized_crop_root = Path(os.path.abspath(crop_root))
    windows_semantics = os.name == "nt" or sys.platform == "win32"
    case_insensitive = windows_semantics or sys.platform == "darwin"

    def comparison_parts(path: Path) -> tuple[str, ...]:
        parts = path.parts
        if windows_semantics:
            parts = tuple(part.rstrip(" .") for part in parts)
        if case_insensitive:
            return tuple(
                unicodedata.normalize("NFC", part).casefold() for part in parts
            )
        return parts

    crop_parts = comparison_parts(normalized_crop_root)

    for label, artifact_path in (
        ("catalogue output", normalized_output),
        ("run manifest", normalized_manifest),
    ):
        artifact_parts = comparison_parts(artifact_path)
        if crop_parts[: len(artifact_parts)] == artifact_parts:
            raise ValueError(
                f"review crop directory must not be inside the {label} file path: "
                f"{crop_root}"
            )

    try:
        os.path.relpath(normalized_crop_root, normalized_output.parent)
    except ValueError:
        raise ValueError(
            "catalogue output and review crop directory must be on the same drive"
        ) from None


def _run_batch(
    arguments: argparse.Namespace,
    images: list[Path],
    output_path: Path,
    normal_manifest_path: Path,
    *,
    preserve_existing_manifest_on_failure: bool,
) -> int:
    """Delegate a batch while preserving the legacy patchable CLI symbols."""

    operations = BatchOperations(
        build_pipeline=build_pipeline,
        export_catalogue=export_catalogue,
        timeout_from_environment=timeout_from_environment,
        run_directory=run_directory,
        failed_manifest_path=failed_manifest_path,
        backup_existing_catalogue=backup_existing_catalogue,
        restore_catalogue=restore_catalogue,
        write_json_atomic=write_json_atomic,
    )
    return run_batch(
        arguments,
        images,
        output_path,
        normal_manifest_path,
        preserve_existing_manifest_on_failure=preserve_existing_manifest_on_failure,
        operations=operations,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    raw_arguments = list(sys.argv[1:] if argv is None else argv)
    arguments = parser.parse_args(raw_arguments)
    if arguments.command == "setup":
        if not raw_arguments or raw_arguments[0] != "setup":
            parser.error(
                "setup must be the first argument and does not accept batch options"
            )
        return run_setup()
    output_path = arguments.path_to_outputs
    run_manifest_path = manifest_path(output_path)
    if output_path.suffix.casefold() not in SUPPORTED_OUTPUT_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_OUTPUT_EXTENSIONS))
        parser.error(f"output path must end in one of: {supported}")
    for purpose, root, must_be_regular_file in (
        ("catalogue output", output_path, True),
        ("run manifest", run_manifest_path, True),
        ("review crop", arguments.path_to_box_images, False),
    ):
        try:
            redirected = redirected_component(root)
        except OSError as error:
            parser.error(f"could not inspect {purpose} path {root}: {error}")
        if redirected is not None:
            component, kind = redirected
            parser.error(
                f"{purpose} paths must not use a {kind}: {component}. Select a "
                "real local directory instead."
            )
        normalized = Path(os.path.abspath(root))
        try:
            metadata = normalized.lstat()
        except FileNotFoundError:
            continue
        except OSError as error:
            parser.error(f"could not inspect {purpose} path {root}: {error}")
        if must_be_regular_file and not stat.S_ISREG(metadata.st_mode):
            parser.error(
                f"{purpose} must be a regular file when it already exists: {root}"
            )
        if not must_be_regular_file and not stat.S_ISDIR(metadata.st_mode):
            parser.error(
                f"{purpose} must be a directory when it already exists: {root}"
            )
    try:
        validate_output_layout(
            output_path,
            run_manifest_path,
            arguments.path_to_box_images,
        )
    except ValueError as error:
        parser.error(str(error))
    try:
        images = discover_images(arguments.path_to_inputs)
    except ValueError as error:
        if arguments.path_to_inputs == Path("inputs"):
            parser.error(
                f"{error} Run 'uma-bct setup', add photos to inputs/, then run "
                "'uma-bct'."
            )
        parser.error(str(error))
    if not images:
        supported = ", ".join(sorted(IMAGE_EXTENSIONS))
        parser.error(
            f"no supported images found in: {arguments.path_to_inputs}. Add photos "
            f"({supported}), then run 'uma-bct' again."
        )
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
