#!/usr/bin/env python3
"""Safely unpack a verified Ollama .tar.zst archive.

The archive is streamed because it is large. Paths and links are checked here,
then Python's data filter checks them again before writing anything.
"""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path, PurePosixPath


def validate_link_target(member: tarfile.TarInfo) -> None:
    """Reject a link that points outside the archive."""

    target_text = member.linkname.replace("\\", "/")
    target = PurePosixPath(target_text)
    if not target_text or target.is_absolute():
        raise ValueError(f"Unsafe archive link target for {member.name!r}")

    parts = list(PurePosixPath(member.name).parent.parts) if member.issym() else []
    for part in target.parts:
        if part in {"", "."}:
            continue
        if part == "..":
            if not parts:
                raise ValueError(f"Unsafe archive link target for {member.name!r}")
            parts.pop()
        else:
            parts.append(part)


def extract_archive(source: Path, destination: Path) -> None:
    """Unpack files and safe links from a checked archive."""

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
                    if not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
                        raise ValueError(f"Unsupported archive entry: {member.name}")
                    if member.issym() or member.islnk():
                        # Check the link text here; tarfile checks the resolved path again below.
                        validate_link_target(member)
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
