#!/usr/bin/env python3
"""Build or verify the client-facing ZIP around one wheel."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_NAME = "uma-bct-client.zip"
BUNDLE_ROOT = "UMA-BCT"
CLIENT_FILES = (
    "README.md",
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
    missing = tuple(path for path in sources if not path.is_file())
    if missing:
        raise ValueError(f"Missing client file: {missing[0]}")
    return sources


def build_client_bundle(directory: Path, *, source_root: Path = PROJECT_ROOT) -> Path:
    destination = directory / BUNDLE_NAME
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for source in bundle_sources(directory, source_root):
            archive.write(source, f"{BUNDLE_ROOT}/{source.name}")
    verify_client_bundle(directory, source_root=source_root)
    return destination


def verify_client_bundle(directory: Path, *, source_root: Path = PROJECT_ROOT) -> None:
    sources = bundle_sources(directory, source_root)
    expected = {f"{BUNDLE_ROOT}/{source.name}": source.read_bytes() for source in sources}
    try:
        with zipfile.ZipFile(directory / BUNDLE_NAME) as archive:
            if set(archive.namelist()) != set(expected):
                raise ValueError("Client bundle has unexpected or missing files.")
            if any(archive.read(name) != content for name, content in expected.items()):
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
