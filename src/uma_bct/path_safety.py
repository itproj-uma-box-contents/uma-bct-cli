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


def path_entry_exists(path: Path) -> bool:
    """Treat a path as occupied unless its entry is definitely missing.

    Use lstat so dangling links count as occupied. Other inspection errors
    also count as occupied to avoid overwriting an entry we cannot inspect.
    """

    try:
        path.lstat()
    except FileNotFoundError:
        return False
    except OSError:
        return True
    return True


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
    """Reject redirection in every existing component of a private path."""

    checked_path = path.absolute()
    # A new output file has no inode yet, but one of its parents may already
    # redirect elsewhere. Inspect the lexical path all the way to the root.
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
