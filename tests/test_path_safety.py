import os
import stat
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from uma_bct.path_safety import redirected_component, redirected_path_kind


class PathSafetyTest(unittest.TestCase):
    def test_windows_reparse_points_are_rejected_even_when_not_junctions(self):
        path = Mock()
        path.is_junction.return_value = False
        reparse_flag = 0x400
        metadata = SimpleNamespace(
            st_mode=stat.S_IFDIR,
            st_file_attributes=reparse_flag,
        )

        with patch(
            "uma_bct.path_safety.stat.FILE_ATTRIBUTE_REPARSE_POINT",
            reparse_flag,
            create=True,
        ):
            self.assertEqual(
                redirected_path_kind(path, metadata),
                "Windows reparse point",
            )

    def test_relative_root_owned_symlink_is_not_a_trusted_system_alias(self):
        metadata = SimpleNamespace(st_mode=stat.S_IFLNK, st_uid=0)

        with patch("pathlib.Path.lstat", return_value=metadata):
            redirected = redirected_component(Path("outputs"))

        self.assertEqual(
            redirected,
            (Path(os.path.abspath("outputs")), "symbolic link"),
        )


if __name__ == "__main__":
    unittest.main()
