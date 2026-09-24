#!/usr/bin/env python3
"""Check release archives for private or unsafe files.

The check reads file names without extracting anything. It blocks photos,
models, generated outputs, links, and paths that could leave the archive.
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
    ".uma-bct-launcher.lock",
    ".working",
    ".uma-bct-runtime",
    "archive box photos",
    "archive_box_photos",
    "add_repository_photos",
    "add photos here",
    "box images",
    "input",
    "inputs",
    "model",
    "models",
    "output",
    "outputs",
    "results",
}
PRIVATE_SUFFIXES = {
    ".avif",
    ".bmp",
    ".csv",
    ".dng",
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


# Collect the artifact paths.
def artifact_paths(arguments: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for argument in arguments:
        candidate = Path(argument)
        if candidate.is_dir():
            paths.extend(
                path for path in sorted(candidate.iterdir()) if path.is_file() and path.name.endswith(ARCHIVE_SUFFIXES)
            )
        else:
            paths.append(candidate)
    return paths


# Handle the archive members step.
def archive_members(path: Path) -> Iterable[tuple[str, bool, bool, int, bool]]:
    if path.name.endswith((".whl", ".zip")):
        with zipfile.ZipFile(path) as archive:
            seen: set[str] = set()
            for member in archive.infolist():
                mode = (member.external_attr >> 16) & 0xFFFF
                regular = member.create_system != 3 or stat.S_IFMT(mode) in {0, stat.S_IFREG, stat.S_IFDIR}
                key = member.filename.replace("\\", "/").casefold().rstrip("/")
                duplicate = key in seen
                seen.add(key)
                yield member.filename, regular, member.is_dir(), member.file_size, duplicate
        return
    if path.name.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as archive:
            seen = set()
            for member in archive.getmembers():
                key = member.name.replace("\\", "/").casefold().rstrip("/")
                duplicate = key in seen
                seen.add(key)
                yield member.name, member.isfile() or member.isdir(), member.isdir(), member.size, duplicate
        return
    raise ValueError(f"Unsupported release artifact: {path}")


# Handle the unsafe reason step.
def unsafe_reason(
    name: str, regular: bool = True, is_directory: bool = False, size: int = 0, duplicate: bool = False
) -> str | None:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts:
        return "unsafe archive path"
    if duplicate:
        return "duplicate archive entry"
    if not regular:
        return "archive link or special entry"
    if is_directory and tuple(part.casefold() for part in path.parts) in {
        ("archive box transcription", "add photos here"),
        ("archive box transcription", "results"),
    } and size == 0:
        return None
    if is_directory and size:
        return "archive directory entry contains data"
    parts = {part.casefold() for part in path.parts}
    if any(
        part.startswith((".uma-bct-launcher-lock-", ".uma-bct-lock-release-"))
        for part in parts
    ):
        return "generated lock or lock-recovery path"
    if private := parts.intersection(PRIVATE_DIRECTORIES):
        return f"private directory {sorted(private)[0]!r}"
    folded = path.name.casefold()
    if folded.endswith(".uma-bct.lock") or folded == ".uma-bct-launcher.lock":
        return "generated run lock"
    if folded.startswith(".uma-bct-launcher-lock-"):
        return "generated launcher-lock recovery"
    if folded.endswith(".run.json") or ".run.json." in folded:
        return "generated run manifest"
    if folded in {"run progress.json", "run details.json"}:
        return "generated managed run manifest"
    if folded.startswith(
        (".uma-bct-json-", ".uma-bct-catalogue-", ".uma-bct-lock-release-", ".uma-bct-rollback-", ".uma-bct-backup-")
    ):
        return "generated temporary output"
    for suffix in PRIVATE_SUFFIXES:
        if folded.endswith(suffix) or f"{suffix}." in folded:
            return f"private file type {suffix}"
    return None


# Inspect the artifact.
def inspect_artifact(path: Path) -> list[str]:
    if not path.is_file():
        return [f"{path}: artifact does not exist"]
    try:
        return [
            f"{path}: {name!r}: {reason}"
            for name, regular, is_directory, size, duplicate in archive_members(path)
            if (reason := unsafe_reason(name, regular, is_directory, size, duplicate))
        ]
    except (OSError, tarfile.TarError, zipfile.BadZipFile, ValueError) as error:
        return [f"{path}: cannot inspect artifact: {error}"]


# Run this command.
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+")
    artifacts = artifact_paths(parser.parse_args(argv).paths)
    findings = [finding for artifact in artifacts for finding in inspect_artifact(artifact)]
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
