#!/usr/bin/env python3
"""Safely extract a verified Ollama .tar.zst archive.

The archive is streamed member-by-member (``mode="r|"``) rather than loaded
into memory, since the Ollama runtime archive is large. Every member path and
link target is checked for an absolute path or archive-root escape before
extraction (``extract_archive``), and ``filter="data"`` independently resolves
targets against the destination while stripping inherited ownership and
permission metadata. Only files, directories, and safe internal links are
accepted.
"""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path, PurePosixPath


def _validate_link_target(member: tarfile.TarInfo) -> None:
    """Reject links whose normalized archive-relative target escapes the root."""

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
                    if member.issym() or member.islnk():
                        _validate_link_target(member)
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
