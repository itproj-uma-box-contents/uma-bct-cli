#!/usr/bin/env python3
"""Reject private images, outputs, or model files in release archives.

This is the last automated privacy gate before a client ZIP ships (see the
build sequence in ``DEVELOPMENT.md``): it opens each built wheel/sdist/zip
and inspects every member name without ever extracting it, so a malformed or
oversized archive can't do anything to the filesystem during the check
itself. ``unsafe_reason`` flags an entry for any of several independent
reasons -- a path-traversal attempt, a non-regular (symlink/special) entry, a
private directory name like ``inputs`` or ``.uma-bct-runtime``, a generated
run manifest, or a private file extension (photos, models, spreadsheets) --
so a private path or file accidentally included during packaging fails the
build instead of shipping to a client.
"""

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
    ".ollama",
    ".uma-bct-runtime",
    "archive box photos",
    "archive_box_photos",
    "add_repository_photos",
    "input",
    "inputs",
    "model",
    "models",
    "output",
    "outputs",
}
PRIVATE_SUFFIXES = {
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


def artifact_paths(arguments: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for argument in arguments:
        candidate = Path(argument)
        if candidate.is_dir():
            paths.extend(
                path
                for path in sorted(candidate.iterdir())
                if path.is_file() and path.name.endswith(ARCHIVE_SUFFIXES)
            )
        else:
            paths.append(candidate)
    return paths


def archive_members(path: Path) -> Iterable[tuple[str, bool]]:
    if path.name.endswith((".whl", ".zip")):
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                mode = (member.external_attr >> 16) & 0xFFFF
                regular = member.create_system != 3 or stat.S_IFMT(mode) in {
                    0,
                    stat.S_IFREG,
                    stat.S_IFDIR,
                }
                yield member.filename, regular
        return
    if path.name.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as archive:
            for member in archive.getmembers():
                yield member.name, member.isfile() or member.isdir()
        return
    raise ValueError(f"Unsupported release artifact: {path}")


def unsafe_reason(name: str, regular: bool = True) -> str | None:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts:
        return "unsafe archive path"
    if not regular:
        return "archive link or special entry"
    parts = {part.casefold() for part in path.parts}
    if private := parts.intersection(PRIVATE_DIRECTORIES):
        return f"private directory {sorted(private)[0]!r}"
    folded = path.name.casefold()
    if folded.endswith(".run.json") or ".run.json." in folded:
        return "generated run manifest"
    if folded.startswith((".uma-bct-json-", ".uma-bct-catalogue-")):
        return "generated temporary output"
    for suffix in PRIVATE_SUFFIXES:
        if folded.endswith(suffix) or f"{suffix}." in folded:
            return f"private file type {suffix}"
    return None


def inspect_artifact(path: Path) -> list[str]:
    if not path.is_file():
        return [f"{path}: artifact does not exist"]
    try:
        return [
            f"{path}: {name!r}: {reason}"
            for name, regular in archive_members(path)
            if (reason := unsafe_reason(name, regular))
        ]
    except (OSError, tarfile.TarError, zipfile.BadZipFile, ValueError) as error:
        return [f"{path}: cannot inspect artifact: {error}"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+")
    artifacts = artifact_paths(parser.parse_args(argv).paths)
    findings = [
        finding for artifact in artifacts for finding in inspect_artifact(artifact)
    ]
    if not artifacts:
        findings.append("No supported release artifact found.")
    if findings:
        print("Release artifact privacy check failed:", file=sys.stderr)
        print("\n".join(f"- {finding}" for finding in findings), file=sys.stderr)
        return 1
    print(f"Release artifact privacy check passed for {len(artifacts)} artifact(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
