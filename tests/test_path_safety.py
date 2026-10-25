"""Path occupancy checks must preserve linked and inaccessible entries."""

import errno
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from uma_bct.path_safety import path_entry_exists


class PathEntryExistsTest(unittest.TestCase):
    def test_existing_files_and_directories_are_occupied(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            file = root / "catalogue.csv"
            file.write_text("synthetic catalogue", encoding="utf-8")
            for path in (root, file):
                with self.subTest(path=path):
                    self.assertTrue(path_entry_exists(path))

    def test_missing_entry_is_available(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertFalse(path_entry_exists(Path(directory) / "missing.csv"))

    def test_dangling_symbolic_link_is_occupied(self):
        with tempfile.TemporaryDirectory() as directory:
            link = Path(directory) / "catalogue.csv"
            try:
                link.symlink_to("missing.csv")
            except (OSError, NotImplementedError) as error:
                self.skipTest(str(error))
            self.assertTrue(path_entry_exists(link))

    def test_inspection_errors_do_not_authorize_overwriting(self):
        path = Path("uninspectable.csv")
        for error in (PermissionError(errno.EACCES, "denied"), NotADirectoryError(), OSError(errno.EIO, "I/O error")):
            with self.subTest(error=type(error).__name__), patch.object(Path, "lstat", side_effect=error):
                self.assertTrue(path_entry_exists(path))
