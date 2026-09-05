"""Basic rejection of paths that redirect private input or output."""

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
