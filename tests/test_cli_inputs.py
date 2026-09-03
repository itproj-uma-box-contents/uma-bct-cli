import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from uma_bct.cli import discover_images


class CliInputTest(unittest.TestCase):
    def test_discovers_case_variant_filenames_in_deterministic_order(self):
        directory = Mock()
        directory.lstat.return_value.st_mode = stat.S_IFDIR
        directory.is_junction.return_value = False
        lower = Mock(name="lower")
        lower.lstat.return_value.st_mode = stat.S_IFREG
        lower.is_junction.return_value = False
        lower.suffix = ".jpg"
        lower.name = "photo.jpg"
        upper = Mock(name="upper")
        upper.lstat.return_value.st_mode = stat.S_IFREG
        upper.is_junction.return_value = False
        upper.suffix = ".JPG"
        upper.name = "Photo.JPG"
        directory.iterdir.return_value = iter((lower, upper))

        with (
            patch("uma_bct.cli.os.access", return_value=True),
            patch("uma_bct.cli.ensure_readable_image"),
        ):
            discovered = discover_images(directory)

        self.assertEqual(discovered, [upper, lower])

    def test_directory_enumeration_failure_is_a_configuration_error(self):
        directory = Mock()
        directory.lstat.return_value.st_mode = stat.S_IFDIR
        directory.is_junction.return_value = False
        directory.iterdir.side_effect = PermissionError("denied")

        with (
            patch("uma_bct.cli.os.access", return_value=True),
            self.assertRaisesRegex(ValueError, "Input directory is not readable"),
        ):
            discover_images(directory)

    def test_each_discovered_image_is_preflighted_for_readability(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_path = root / "shelf.jpg"
            image_path.touch()

            with (
                patch(
                    "uma_bct.cli.ensure_readable_image",
                    side_effect=ValueError("Input image is not readable"),
                ),
                self.assertRaisesRegex(ValueError, "not readable"),
            ):
                discover_images(root)

    def test_disappearing_directory_entry_is_a_configuration_error(self):
        directory = Mock()
        directory.lstat.return_value.st_mode = stat.S_IFDIR
        directory.is_junction.return_value = False
        candidate = Mock()
        candidate.suffix = ".jpg"
        candidate.lstat.side_effect = FileNotFoundError("disappeared")
        directory.iterdir.return_value = iter((candidate,))

        with (
            patch("uma_bct.cli.os.access", return_value=True),
            self.assertRaisesRegex(ValueError, "Could not inspect input path"),
        ):
            discover_images(directory)

    def test_rejects_windows_directory_junction_inputs(self):
        directory = Mock()
        directory.lstat.return_value.st_mode = stat.S_IFDIR
        directory.is_junction.return_value = True

        with self.assertRaisesRegex(ValueError, "junctions are not accepted"):
            discover_images(directory)

    def test_rejects_symbolic_link_inputs(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            real_image = root / "real.jpg"
            linked_image = root / "linked.jpg"
            real_image.touch()
            try:
                linked_image.symlink_to(real_image)
            except OSError as error:
                self.skipTest(f"symbolic links are unavailable: {error}")

            with self.assertRaisesRegex(ValueError, "symbolic links are not accepted"):
                discover_images(linked_image)
            with self.assertRaisesRegex(ValueError, "symbolic links are not accepted"):
                discover_images(root)

    def test_rejects_an_input_reached_through_a_symbolic_linked_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            real_directory = root / "real"
            linked_directory = root / "linked"
            real_directory.mkdir()
            image_path = real_directory / "shelf.jpg"
            image_path.touch()
            try:
                linked_directory.symlink_to(real_directory, target_is_directory=True)
            except OSError as error:
                self.skipTest(f"symbolic links are unavailable: {error}")

            with self.assertRaisesRegex(ValueError, "symbolic links are not accepted"):
                discover_images(linked_directory / image_path.name)


if __name__ == "__main__":
    unittest.main()
