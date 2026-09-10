"""Basic rejection of paths that redirect private input or output.

``redirected_path_kind`` identifies one path entry. ``check_path`` walks the
whole path so a missing output cannot hide a linked parent. These checks stop
private reads or writes from being sent somewhere unexpected.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path


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

    checked_path = Path(os.path.abspath(path))
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
                kind == "symbolic link" and part.parent == Path(part.anchor) and getattr(metadata, "st_uid", -1) == 0
            )
            if root_owned_alias:
                continue
            subject = purpose if part == checked_path else f"{purpose} parent"
            raise ValueError(f"{subject} must not be a {kind}: {part}")
