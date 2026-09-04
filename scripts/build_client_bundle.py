#!/usr/bin/env python3
"""Build the client-facing launcher bundle around one verified wheel."""

from __future__ import annotations

import argparse
import stat
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_DIRECTORY = "UMA-BCT"
BUNDLE_NAME = "uma-bct-client.zip"
CLIENT_FILES = (
    "DEVELOPMENT.md",
    "README.md",
    "Run UMA BCT.cmd",
    "client-constraints.txt",
    "run-uma-bct.ps1",
    "run-uma-bct.sh",
    "scripts/extract_ollama_archive.py",
)


def _wheel(distribution_directory: Path) -> Path:
    wheels = sorted(distribution_directory.glob("uma_bct_cli-*.whl"))
    if len(wheels) != 1:
        raise ValueError(
            f"Expected exactly one wheel in {distribution_directory}; "
            f"found {len(wheels)}."
        )
    return wheels[0]


def _write_file(archive: zipfile.ZipFile, source: Path, archive_name: str) -> None:
    """Write one regular file with stable cross-platform permissions."""

    permissions = 0o755 if source.name.endswith(".sh") else 0o644
    member = zipfile.ZipInfo(archive_name)
    member.create_system = 3
    member.external_attr = (stat.S_IFREG | permissions) << 16
    member.compress_type = zipfile.ZIP_DEFLATED
    archive.writestr(member, source.read_bytes())


def build_client_bundle(
    distribution_directory: Path,
    *,
    source_root: Path = PROJECT_ROOT,
) -> Path:
    """Package the launchers, client guide, and exactly one wheel."""

    wheel = _wheel(distribution_directory)

    sources = [source_root / name for name in CLIENT_FILES]
    missing = [path.name for path in sources if not path.is_file()]
    if missing:
        raise ValueError(f"Missing client bundle file(s): {', '.join(missing)}")

    destination = distribution_directory / BUNDLE_NAME
    with zipfile.ZipFile(
        destination,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for source in (*sources, wheel):
            _write_file(archive, source, f"{BUNDLE_DIRECTORY}/{source.name}")
    verify_client_bundle(distribution_directory, source_root=source_root)
    return destination


def verify_client_bundle(
    distribution_directory: Path,
    *,
    source_root: Path = PROJECT_ROOT,
) -> None:
    """Verify exact bundle contents against the source files and wheel."""

    wheel = _wheel(distribution_directory)
    bundle = distribution_directory / BUNDLE_NAME
    sources = [*(source_root / name for name in CLIENT_FILES), wheel]
    expected_names = [f"{BUNDLE_DIRECTORY}/{source.name}" for source in sources]
    try:
        with zipfile.ZipFile(bundle) as archive:
            if sorted(archive.namelist()) != sorted(expected_names):
                raise ValueError("Client bundle has unexpected or missing entries.")
            for source, archive_name in zip(sources, expected_names, strict=True):
                member = archive.getinfo(archive_name)
                mode = (member.external_attr >> 16) & 0xFFFF
                if not stat.S_ISREG(mode):
                    raise ValueError(
                        f"Client bundle entry is not a regular file: {archive_name}"
                    )
                if archive.read(member) != source.read_bytes():
                    raise ValueError(
                        f"Client bundle entry does not match: {archive_name}"
                    )
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError(f"Could not verify client bundle: {error}") from error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("distribution_directory", type=Path)
    parser.add_argument("--verify", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.verify:
        verify_client_bundle(arguments.distribution_directory)
        print(arguments.distribution_directory / BUNDLE_NAME)
    else:
        print(build_client_bundle(arguments.distribution_directory))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
