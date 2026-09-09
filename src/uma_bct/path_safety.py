"""Basic rejection of paths that redirect private input or output.

``redirected_path_kind`` is used everywhere a path supplied via CLI
arguments is first touched (``cli.py``, ``setup_command.py``). A symlink or,
on Windows, a directory junction/reparse point could silently redirect reads
or writes outside an input/output directory the user believes is private;
this module gives every caller one place to detect that redirection and
refuse it before any I/O happens.
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
