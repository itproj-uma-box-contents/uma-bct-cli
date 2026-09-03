import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from uma_bct.run_artifacts import (
    acquire_output_lock,
    backup_existing_catalogue,
    manifest_path,
    output_lock_path,
    release_output_lock,
    restore_catalogue,
    write_json_atomic,
)


class RunArtifactTest(unittest.TestCase):
    def test_release_removes_the_owned_output_lock(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "catalogue.csv"
            reservation = acquire_output_lock(output_path)

            release_output_lock(*reservation)

            self.assertFalse(output_lock_path(output_path).exists())

    def test_generated_output_siblings_are_bounded_and_collision_safe(self):
        first = Path(f"{'a' * 240}x.csv")
        second = Path(f"{'a' * 240}y.csv")

        first_manifest = manifest_path(first)
        second_manifest = manifest_path(second)

        self.assertLessEqual(len(first_manifest.name.encode("utf-8")), 255)
        self.assertLessEqual(len(output_lock_path(first).name.encode("utf-8")), 255)
        self.assertNotEqual(first_manifest, second_manifest)

    def test_normalized_output_stems_do_not_share_manifests_or_locks(self):
        ordinary = Path("foo.csv")
        trailing_space = Path("foo .csv")
        trailing_dot = Path("foo..csv")

        self.assertNotEqual(manifest_path(ordinary), manifest_path(trailing_space))
        self.assertNotEqual(
            output_lock_path(ordinary), output_lock_path(trailing_space)
        )
        self.assertNotEqual(manifest_path(ordinary), manifest_path(trailing_dot))
        self.assertNotEqual(output_lock_path(ordinary), output_lock_path(trailing_dot))

    def test_atomic_json_failure_preserves_existing_destination(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            destination = root / "catalogue.run.json"
            destination.write_text("keep me\n", encoding="utf-8")

            with self.assertRaises(TypeError):
                write_json_atomic(destination, {"invalid": object()})

            self.assertEqual(destination.read_text(encoding="utf-8"), "keep me\n")
            self.assertEqual(list(root.glob(".uma-bct-json-*.tmp")), [])

    def test_failed_lock_initialization_removes_only_its_new_lock(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "catalogue.csv"

            with (
                patch(
                    "uma_bct.run_artifacts.os.write",
                    side_effect=OSError("disk full"),
                ),
                self.assertRaisesRegex(ValueError, "could not reserve output"),
            ):
                acquire_output_lock(output_path)

            self.assertFalse(output_lock_path(output_path).exists())

    def test_release_does_not_remove_a_replacement_lock(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "catalogue.csv"
            descriptor, lock_path, identity = acquire_output_lock(output_path)

            replacement_identity = SimpleNamespace(
                st_dev=identity[0],
                st_ino=identity[1] + 1,
            )
            with patch("pathlib.Path.stat", return_value=replacement_identity):
                release_output_lock(descriptor, lock_path, identity)

            self.assertTrue(lock_path.is_file())

    def test_catalogue_backup_can_restore_the_previous_bytes(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "catalogue.csv"
            destination.write_bytes(b"previous catalogue\n")

            backup = backup_existing_catalogue(destination)
            self.assertIsNotNone(backup)
            destination.write_bytes(b"replacement catalogue\n")
            restore_catalogue(destination, backup)

            self.assertEqual(destination.read_bytes(), b"previous catalogue\n")
            self.assertFalse(backup.exists())


if __name__ == "__main__":
    unittest.main()
