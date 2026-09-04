#!/usr/bin/env python3
"""Fail when a release artifact contains private/runtime-only UMA BCT data."""

from __future__ import annotations

import argparse
import stat
import sys
import tarfile
import zipfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

ARCHIVE_SUFFIXES = (".whl", ".tar.gz", ".zip")
PRIVATE_DIRECTORIES = {
    ".uma-bct-runtime",
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
    if artifact.name.endswith((".whl", ".zip")):
        with zipfile.ZipFile(artifact) as archive:
            for member in archive.infolist():
                unix_mode = (member.external_attr >> 16) & 0xFFFF
                file_type = stat.S_IFMT(unix_mode)
                if member.create_system == 3 and file_type not in {
                    0,
                    stat.S_IFREG,
                    stat.S_IFDIR,
                }:
                    yield f"{member.filename} -> <zip link or special entry>"
                else:
                    yield member.filename
        return
    if artifact.name.endswith(".tar.gz"):
        with tarfile.open(artifact, mode="r:gz") as archive:
            for member in archive.getmembers():
                if member.issym() or member.islnk():
                    yield f"{member.name} -> {member.linkname}"
                elif not member.isfile() and not member.isdir():
                    yield f"{member.name} -> <tar special entry>"
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

    folded_name = path.name.casefold()
    if folded_name.endswith(".run.json") or ".run.json." in folded_name:
        return "generated run manifest"
    if folded_name.startswith(PRIVATE_RUNTIME_PREFIXES):
        return "generated private runtime file"

    private_suffix = next(
        (
            f".{part}"
            for part in folded_name.split(".")[1:]
            if f".{part}" in PRIVATE_FILE_SUFFIXES
        ),
        None,
    )
    if private_suffix is not None:
        return f"private image, catalogue, or model file ({private_suffix})"
    if " -> " in member_name:
        return "archive link or special entry"
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
        print("No supported release artifact found.", file=sys.stderr)
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
