#!/usr/bin/env python3
"""Safely extract a verified Ollama .tar.zst archive."""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path, PurePosixPath


def extract_archive(source: Path, destination: Path) -> None:
    """Extract files and safe internal links from a verified archive."""

    import zstandard

    destination.mkdir(mode=0o700, parents=True, exist_ok=True)
    with source.open("rb") as compressed:
        decompressor = zstandard.ZstdDecompressor()
        with decompressor.stream_reader(compressed) as stream:
            with tarfile.open(fileobj=stream, mode="r|") as archive:
                for member in archive:
                    path = PurePosixPath(member.name)
                    if path.is_absolute() or ".." in path.parts:
                        raise ValueError(f"Unsafe archive path: {member.name}")
                    if not (
                        member.isfile()
                        or member.isdir()
                        or member.issym()
                        or member.islnk()
                    ):
                        raise ValueError(f"Unsupported archive entry: {member.name}")
                    archive.extract(member, destination, filter="data")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    arguments = parser.parse_args(argv)
    extract_archive(arguments.source, arguments.destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
