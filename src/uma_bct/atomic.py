"""Small cross-platform helpers for atomic file publication."""

from __future__ import annotations

import ctypes
import errno
import os
import sys
from pathlib import Path

AT_FDCWD = -100
LINUX_RENAME_NOREPLACE = 1
MACOS_RENAME_EXCL = 0x00000004


def _raise_rename_error(destination: Path) -> None:
    error_number = ctypes.get_errno()
    raise OSError(error_number, os.strerror(error_number), str(destination))


def _rename_exclusive(source: Path, destination: Path) -> None:
    """Atomically move a file only when the destination does not exist."""

    if os.name == "nt":
        source.rename(destination)
        return

    library = ctypes.CDLL(None, use_errno=True)
    encoded_source = os.fsencode(source)
    encoded_destination = os.fsencode(destination)
    if sys.platform == "darwin":
        rename = library.renamex_np
        rename.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
        rename.restype = ctypes.c_int
        result = rename(encoded_source, encoded_destination, MACOS_RENAME_EXCL)
    elif sys.platform.startswith("linux"):
        rename = library.renameat2
        rename.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
        rename.restype = ctypes.c_int
        result = rename(AT_FDCWD, encoded_source, AT_FDCWD, encoded_destination, LINUX_RENAME_NOREPLACE)
    else:
        raise OSError(errno.ENOTSUP, "Atomic no-clobber publication is unsupported on this platform.")
    if result != 0:
        _raise_rename_error(destination)


def rename_new_file(source: Path, destination: Path, *, existing_message: str) -> None:
    """Publish a complete file without replacing a late-arriving destination."""

    try:
        _rename_exclusive(source, destination)
    except FileExistsError:
        raise FileExistsError(existing_message) from None
