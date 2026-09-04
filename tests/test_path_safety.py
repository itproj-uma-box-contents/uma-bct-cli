import os
import stat
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from uma_bct.path_safety import redirected_component, redirected_path_kind


class PathSafetyTest(unittest.TestCase):
    def test_safe_parent_directory_references_are_supported(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "current").mkdir()
            (root / "output").mkdir()

            self.assertIsNone(redirected_component(root / "current" / ".." / "output"))

    def test_symlink_before_parent_directory_reference_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            target = root / "target"
            linked = root / "linked"
            target.mkdir()
            try:
                linked.symlink_to(target, target_is_directory=True)
            except OSError as error:
                self.skipTest(f"symbolic links are unavailable: {error}")

            self.assertEqual(
                redirected_component(linked / ".." / "output"),
                (linked, "symbolic link"),
            )

    def test_missing_component_cannot_hide_a_later_symlink(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            target = root / "target" / "child"
            linked = root / "linked"
            target.mkdir(parents=True)
            try:
                linked.symlink_to(target, target_is_directory=True)
            except OSError as error:
                self.skipTest(f"symbolic links are unavailable: {error}")

            path = root / "missing" / ".." / "linked" / ".." / "output"

            redirected = redirected_component(path)
            self.assertIsNotNone(redirected)
            component, kind = redirected
            self.assertEqual(
                (Path(os.path.abspath(component)), kind),
                (linked, "symbolic link"),
            )
            self.assertFalse((root / "missing").exists())

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
