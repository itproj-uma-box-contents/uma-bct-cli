"""Basic rejection of paths that redirect private input or output.

``redirected_path_kind`` identifies one path entry. ``check_path`` walks the
whole path so a missing output cannot hide a linked parent. These checks stop
private reads or writes from being sent somewhere unexpected.
"""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

MACOS_ROOT_ALIASES = frozenset({Path("/etc"), Path("/tmp"), Path("/var")})


def redirected_path_kind(path: Path, metadata: os.stat_result | None = None) -> str | None:
    metadata = metadata or path.lstat()
    if stat.S_ISLNK(metadata.st_mode):
        return "symbolic link"
    attributes = getattr(metadata, "st_file_attributes", 0)
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if reparse and attributes & reparse:
        return "directory junction" if path.is_junction() else "reparse point"
    if path.is_junction():
        return "directory junction"
    return None


def check_path(path: Path, purpose: str) -> None:
    """Reject a linked path or parent."""

    checked_path = path.absolute()
    # Walk back to the root because the final path may not exist yet.
    for part in (checked_path, *checked_path.parents):
        try:
            metadata = part.lstat()
        except FileNotFoundError:
            continue
        except OSError as error:
            raise ValueError(f"Could not inspect {purpose} {checked_path}: {error}") from error
        try:
            kind = redirected_path_kind(part, metadata)
        except OSError as error:
            raise ValueError(f"Could not inspect {purpose} {checked_path}: {error}") from error
        if kind:
            # macOS exposes root-owned aliases such as /var; unprivileged users cannot redirect them.
            root_owned_alias = (
                sys.platform == "darwin"
                and part in MACOS_ROOT_ALIASES
                and kind == "symbolic link"
                and getattr(metadata, "st_uid", -1) == 0
            )
            if root_owned_alias:
                continue
            subject = purpose if part == checked_path else f"{purpose} parent"
            raise ValueError(f"{subject} must not be a {kind}: {part}")
