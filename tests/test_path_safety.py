import stat
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from uma_bct.path_safety import redirected_component


class PathSafetyTest(unittest.TestCase):
    def test_relative_root_owned_symlink_is_not_a_trusted_system_alias(self):
        metadata = SimpleNamespace(st_mode=stat.S_IFLNK, st_uid=0)

        with patch("pathlib.Path.lstat", return_value=metadata):
            redirected = redirected_component(Path("outputs"))

        self.assertEqual(redirected, (Path("outputs"), "symbolic link"))


if __name__ == "__main__":
    unittest.main()
