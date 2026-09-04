"""Cross-platform checks for paths that redirect filesystem access."""

from __future__ import annotations

import os
import stat
from pathlib import Path


def redirected_path_kind(path: Path, metadata: os.stat_result) -> str | None:
    """Identify links and Windows reparse paths that redirect file access."""

    if stat.S_ISLNK(metadata.st_mode):
        return "symbolic link"

    file_attributes = getattr(metadata, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if (
        isinstance(file_attributes, int)
        and reparse_flag
        and file_attributes & reparse_flag
    ):
        return "directory junction" if path.is_junction() else "Windows reparse point"
    if path.is_junction():
        return "directory junction"
    return None


def redirected_paths_label(kind: str) -> str:
    """Return the plural label used in input-policy errors."""

    return {
        "symbolic link": "symbolic links",
        "directory junction": "directory junctions",
        "Windows reparse point": "Windows reparse points",
    }[kind]


def redirected_component(path: Path) -> tuple[Path, str] | None:
    """Find an existing path component that redirects to another location."""

    uncollapsed = path if path.is_absolute() else Path.cwd() / path
    seen: set[Path] = set()
    for prefix in (uncollapsed, *uncollapsed.parents):
        # Normalizing each prefix exposes links after missing/.. components.
        for candidate in (prefix, Path(os.path.abspath(prefix))):
            if candidate in seen:
                continue
            seen.add(candidate)
            try:
                metadata = candidate.lstat()
            except FileNotFoundError:
                continue
            kind = redirected_path_kind(candidate, metadata)
            if kind is not None:
                # POSIX systems may provide root-owned aliases such as /tmp and /var.
                # They are outside user control and safe to treat as system paths.
                filesystem_root = Path(candidate.anchor)
                if (
                    kind == "symbolic link"
                    and os.name != "nt"
                    and candidate.is_absolute()
                    and candidate.parent == filesystem_root
                    and metadata.st_uid == 0
                ):
                    continue
                return candidate, kind
    return None
