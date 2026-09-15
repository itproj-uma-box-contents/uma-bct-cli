"""Small cross-platform helpers for atomic file publication."""

from __future__ import annotations

import ctypes
import errno
import os
import stat
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO

from .path_safety import check_path

AT_FDCWD = -100
LINUX_RENAME_NOREPLACE = 1
MACOS_RENAME_EXCL = 0x00000004


def file_identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns


@contextmanager
def open_stable_file(path: Path, purpose: str, *, validator: Callable[[Path], os.stat_result] | None = None,
    error_type: type[Exception] = ValueError) -> Iterator[tuple[BinaryIO, os.stat_result]]:
    """Open one regular file and reject links or changes during the read."""

    def inspect() -> os.stat_result:
        if validator is not None:
            return validator(path)
        check_path(path, purpose)
        return path.lstat()

    before = inspect()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise error_type(f"{purpose} must be one unlinked regular file: {path}")
    expected = file_identity(before)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        source = os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise
    try:
        opened = os.fstat(descriptor)
        if file_identity(opened) != expected or opened.st_nlink != 1:
            raise error_type(f"{purpose} changed while it was opened: {path}")
        yield source, before
        if source.tell() != before.st_size:
            raise error_type(f"{purpose} changed while it was being read: {path}")
        after_open = os.fstat(descriptor)
        after_path = inspect()
        if file_identity(after_open) != expected or file_identity(after_path) != expected or after_path.st_nlink != 1:
            raise error_type(f"{purpose} changed while it was being read: {path}")
    finally:
        source.close()


def raise_rename_error(destination: Path) -> None:
    error_number = ctypes.get_errno()
    raise OSError(error_number, os.strerror(error_number), str(destination))


def rename_exclusive(source: Path, destination: Path) -> None:
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
        raise_rename_error(destination)


def rename_new_file(source: Path, destination: Path, *, existing_message: str) -> None:
    """Publish a complete file without replacing a late-arriving destination."""

    try:
        rename_exclusive(source, destination)
    except FileExistsError:
        raise FileExistsError(existing_message) from None
