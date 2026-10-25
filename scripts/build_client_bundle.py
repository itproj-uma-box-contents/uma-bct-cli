#!/usr/bin/env python3
"""Build or verify the client-facing ZIP around one wheel."""

from __future__ import annotations

import argparse
import os
import stat
import tempfile
import zipfile
from pathlib import Path

from uma_bct.atomic import open_stable_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_NAME = "archive-box-transcription.zip"
BUNDLE_ROOT = "Archive Box Transcription"
FRIENDLY_INPUT_DIRECTORY = "Add Photos Here"
FRIENDLY_RESULTS_DIRECTORY = "Results"
CLIENT_FILES = (
    "README.md",
    "CONTRIBUTING.md",
    "AGENTS.md",
    "docs/usage.md",
    "docs/catalogue.md",
    "docs/hosted-models.md",
    "docs/architecture.md",
    "Run UMA BCT.cmd",
    "client-constraints.txt",
    "run-uma-bct.ps1",
    "run-uma-bct.sh",
    "scripts/extract_ollama_archive.py",
)


def find_wheel(directory: Path) -> Path:
    wheels = tuple(directory.glob("uma_bct_cli-*.whl"))
    if len(wheels) != 1:
        raise ValueError(f"Expected one UMA BCT wheel; found {len(wheels)}.")
    return wheels[0]


def bundle_sources(directory: Path, root: Path) -> tuple[Path, ...]:
    sources = tuple(root / name for name in CLIENT_FILES) + (find_wheel(directory),)
    for source in sources:
        boundary = directory if source.parent == directory else root
        validate_source_path(source, boundary)
    return sources


def validate_source_path(path: Path, boundary: Path) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ValueError(f"Missing or unreadable client file: {path}: {error}") from error
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise ValueError(f"Client file must be one unlinked regular file: {path}")
    # Checking only the file would miss a linked parent that redirects the
    # bundle reader into private folders outside the expected source tree.
    current = path.parent
    while True:
        try:
            parent_metadata = current.lstat()
        except OSError as error:
            raise ValueError(f"Could not inspect client-file parent {current}: {error}") from error
        if stat.S_ISLNK(parent_metadata.st_mode) or current.is_junction():
            raise ValueError(f"Client file must not be beneath a linked directory: {current}")
        if current == boundary:
            break
        if current == current.parent:
            raise ValueError(f"Client file is outside its expected source directory: {path}")
        current = current.parent
    return metadata


def read_source(path: Path, boundary: Path) -> tuple[bytes, os.stat_result]:
    def validator(candidate: Path) -> os.stat_result:
        return validate_source_path(candidate, boundary)

    with open_stable_file(path, "Client file", validator=validator) as (source, metadata):
        return source.read(), metadata


def source_boundary(source: Path, directory: Path, root: Path) -> Path:
    return directory if source.parent == directory else root


def bundle_member_name(source: Path, root: Path) -> str:
    """Keep guide links portable while retaining the existing launcher layout."""

    name = source.relative_to(root).as_posix() if source.is_relative_to(root / "docs") else source.name
    return f"{BUNDLE_ROOT}/{name}"


def source_entry(name: str, metadata: os.stat_result) -> zipfile.ZipInfo:
    information = zipfile.ZipInfo(name)
    information.create_system = 3
    information.external_attr = (stat.S_IFREG | stat.S_IMODE(metadata.st_mode)) << 16
    information.compress_type = zipfile.ZIP_DEFLATED
    return information


def validate_bundle_path(destination: Path, directory: Path, *, must_exist: bool) -> None:
    try:
        directory_metadata = directory.lstat()
    except OSError as error:
        raise ValueError(f"Bundle output directory is missing or unreadable: {directory}: {error}") from error
    if not stat.S_ISDIR(directory_metadata.st_mode) or stat.S_ISLNK(directory_metadata.st_mode) or directory.is_junction():
        raise ValueError(f"Bundle output directory must not be linked: {directory}")
    try:
        metadata = destination.lstat()
    except FileNotFoundError:
        if must_exist:
            raise ValueError(f"Client bundle does not exist: {destination}") from None
        return
    except OSError as error:
        raise ValueError(f"Could not inspect client bundle {destination}: {error}") from error
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise ValueError(f"Client bundle must be one unlinked regular file: {destination}")


def build_client_bundle(directory: Path, *, source_root: Path = PROJECT_ROOT) -> Path:
    destination = directory / BUNDLE_NAME
    validate_bundle_path(destination, directory, must_exist=False)
    temporary_path: Path | None = None
    # Build beside the destination and publish once the archive is flushed.
    # The two empty folders are intentional client workspace placeholders.
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", dir=directory, prefix=".archive-box-transcription-", suffix=".tmp", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
                for directory_name in (FRIENDLY_INPUT_DIRECTORY, FRIENDLY_RESULTS_DIRECTORY):
                    directory_entry = zipfile.ZipInfo(f"{BUNDLE_ROOT}/{directory_name}/")
                    directory_entry.create_system = 3
                    directory_entry.external_attr = (0o40700 << 16) | 0x10
                    archive.writestr(directory_entry, b"")
                for source in bundle_sources(directory, source_root):
                    content, metadata = read_source(source, source_boundary(source, directory, source_root))
                    name = bundle_member_name(source, source_root)
                    archive.writestr(source_entry(name, metadata), content)
            temporary.flush()
            os.fsync(temporary.fileno())
        validate_bundle_path(destination, directory, must_exist=False)
        os.replace(temporary_path, destination)
        temporary_path = None
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    verify_client_bundle(directory, source_root=source_root)
    return destination


def verify_client_bundle(directory: Path, *, source_root: Path = PROJECT_ROOT) -> None:
    validate_bundle_path(directory / BUNDLE_NAME, directory, must_exist=True)
    sources = bundle_sources(directory, source_root)
    expected = {
        bundle_member_name(source, source_root): read_source(
            source, source_boundary(source, directory, source_root)
        )[0]
        for source in sources
    }
    expected[f"{BUNDLE_ROOT}/{FRIENDLY_INPUT_DIRECTORY}/"] = b""
    expected[f"{BUNDLE_ROOT}/{FRIENDLY_RESULTS_DIRECTORY}/"] = b""
    # Compare the final ZIP against fresh reads of its exact allowed sources;
    # duplicate or extra entries must not ride along in a deliverable.
    try:
        with zipfile.ZipFile(directory / BUNDLE_NAME) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError("Client bundle contains duplicate entries.")
            if set(names) != set(expected):
                raise ValueError("Client bundle has unexpected or missing files.")
            entries = {item.filename: item for item in archive.infolist()}
            if any(entries[name].is_dir() != name.endswith("/") for name in expected):
                raise ValueError("Client bundle contains an entry with the wrong type.")
            for name, entry in entries.items():
                if entry.create_system != 3:
                    raise ValueError("Client bundle contains an entry with unverifiable platform metadata.")
                entry_type = stat.S_IFMT((entry.external_attr >> 16) & 0xFFFF)
                expected_type = stat.S_IFDIR if name.endswith("/") else stat.S_IFREG
                if entry_type != expected_type:
                    raise ValueError("Client bundle contains a link or special entry.")
            if any(entries[name].file_size != len(content) for name, content in expected.items()):
                raise ValueError("Client bundle contains an entry with the wrong size.")
            if any(archive.read(entries[name]) != content for name, content in expected.items()):
                raise ValueError("Client bundle content does not match its sources.")
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError(f"Could not verify client bundle: {error}") from error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--verify", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.verify:
        verify_client_bundle(arguments.directory)
        result = arguments.directory / BUNDLE_NAME
    else:
        result = build_client_bundle(arguments.directory)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
