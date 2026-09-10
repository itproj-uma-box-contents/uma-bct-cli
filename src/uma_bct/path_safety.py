"""Basic rejection of paths that redirect private input or output.

``redirected_path_kind`` identifies one path entry, while
``redirected_path_component`` walks every existing entry in a hierarchy so a
not-yet-created output cannot hide a redirected parent. A symlink or, on
Windows, a directory junction/reparse point could silently redirect private
reads or writes; these helpers let callers reject that redirection before I/O.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path


def redirected_path_kind(
    path: Path, metadata: os.stat_result | None = None
) -> str | None:
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


def redirected_path_component(path: Path) -> tuple[Path, str] | None:
    """Return the first existing redirected component in a path hierarchy."""

    for component in (path, *path.parents):
        try:
            metadata = component.lstat()
        except FileNotFoundError:
            continue
        if kind := redirected_path_kind(component, metadata):
            return component, kind
    return None


def validate_path_hierarchy(path: Path, purpose: str) -> None:
    """Reject a redirected leaf or parent with a useful error message."""

    try:
        redirected = redirected_path_component(path)
    except OSError as error:
        raise ValueError(f"Could not inspect {purpose} {path}: {error}") from error
    if redirected is None:
        return
    component, kind = redirected
    subject = purpose if component == path else f"{purpose} parent"
    raise ValueError(f"{subject} must not be a {kind}: {component}")
