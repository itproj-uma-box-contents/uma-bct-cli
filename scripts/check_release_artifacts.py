#!/usr/bin/env python3
"""Fail when a wheel or sdist contains private/runtime-only UMA BCT data."""

from __future__ import annotations

import argparse
import sys
import tarfile
import zipfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

ARCHIVE_SUFFIXES = (".whl", ".tar.gz")
PRIVATE_DIRECTORIES = {
    ".ollama",
    "add_repository_photos",
    "archive box photos",
    "archive_box_photos",
    "input",
    "inputs",
    "model",
    "models",
    "output",
    "outputs",
}
PRIVATE_FILE_SUFFIXES = {
    ".avif",
    ".bmp",
    ".csv",
    ".gif",
    ".gguf",
    ".heic",
    ".heif",
    ".jpeg",
    ".jpg",
    ".onnx",
    ".png",
    ".pt",
    ".pth",
    ".safetensors",
    ".tif",
    ".tiff",
    ".webp",
    ".xls",
    ".xlsx",
}
PRIVATE_RUNTIME_PREFIXES = (
    ".uma-bct-json-",
    ".uma-bct-catalogue-",
)


def artifact_paths(arguments: Iterable[str]) -> list[Path]:
    artifacts: list[Path] = []
    for argument in arguments:
        candidate = Path(argument)
        if candidate.is_dir():
            artifacts.extend(
                path
                for path in sorted(candidate.iterdir())
                if path.is_file() and path.name.endswith(ARCHIVE_SUFFIXES)
            )
        else:
            artifacts.append(candidate)
    return artifacts


def member_names(artifact: Path) -> Iterable[str]:
    if artifact.name.endswith(".whl"):
        with zipfile.ZipFile(artifact) as archive:
            yield from archive.namelist()
        return
    if artifact.name.endswith(".tar.gz"):
        with tarfile.open(artifact, mode="r:gz") as archive:
            for member in archive.getmembers():
                if member.issym() or member.islnk():
                    yield f"{member.name} -> {member.linkname}"
                else:
                    yield member.name
        return
    raise ValueError(f"unsupported release artifact: {artifact}")


def unsafe_reason(member_name: str) -> str | None:
    normalized = member_name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts:
        return "unsafe archive path"

    folded_parts = tuple(part.casefold() for part in path.parts)
    private = PRIVATE_DIRECTORIES.intersection(folded_parts)
    if private:
        return f"private/runtime directory {sorted(private)[0]!r}"

    if path.name.casefold().endswith(".run.json"):
        return "generated run manifest"
    if path.name.casefold().startswith(PRIVATE_RUNTIME_PREFIXES):
        return "generated private runtime file"

    suffix = Path(path.name).suffix.casefold()
    if suffix in PRIVATE_FILE_SUFFIXES:
        return f"private image, catalogue, or model file ({suffix})"
    if " -> " in member_name:
        return "archive link"
    return None


def inspect_artifact(artifact: Path) -> list[str]:
    if not artifact.is_file():
        return [f"{artifact}: artifact does not exist"]
    try:
        findings = []
        for member_name in member_names(artifact):
            if reason := unsafe_reason(member_name):
                findings.append(f"{artifact}: {member_name!r}: {reason}")
        return findings
    except (OSError, tarfile.TarError, zipfile.BadZipFile, ValueError) as error:
        return [f"{artifact}: cannot inspect artifact: {error}"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reject private images, outputs, and model files in releases."
    )
    parser.add_argument("paths", nargs="+", help="Wheel/sdist files or directories")
    arguments = parser.parse_args(argv)

    artifacts = artifact_paths(arguments.paths)
    if not artifacts:
        print("No wheel or .tar.gz source distribution found.", file=sys.stderr)
        return 1

    findings = [
        finding for artifact in artifacts for finding in inspect_artifact(artifact)
    ]
    if findings:
        print("Release artifact privacy check failed:", file=sys.stderr)
        for finding in findings:
            print(f"- {finding}", file=sys.stderr)
        return 1

    print(f"Release artifact privacy check passed for {len(artifacts)} artifact(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
