"""Recovery must not mutate files reached through a redirected parent."""

import hashlib
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from uma_bct.managed_batch import remove_working_directory
from uma_bct.run_artifacts import restore_backup_no_clobber, rollback_owned_file


def redirect_directory(link: Path, target: Path) -> None:
    # Directory junctions exercise the Windows path guard without requiring
    # administrator privileges or Developer Mode for symlink creation.
    if os.name == "nt":
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True, capture_output=True)
    else:
        link.symlink_to(target, target_is_directory=True)


class RedirectedRecoveryTest(unittest.TestCase):
    def test_working_cleanup_rejects_a_redirected_parent_before_deleting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            batch, outside = root / "batch", root / "outside"
            batch.mkdir()
            work = outside / "image-0001-attempt-001-demo"
            work.mkdir(parents=True)
            notes = work / "reviewer-notes.txt"
            notes.write_text("keep these notes", encoding="utf-8")
            redirect_directory(batch / ".working", outside)

            with self.assertRaises(ValueError):
                remove_working_directory(batch, ".working/image-0001-attempt-001-demo")
            self.assertEqual(notes.read_text(encoding="utf-8"), "keep these notes")

    def test_rollback_rejects_a_redirected_output_parent_before_moving(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            outside = root / "outside"
            outside.mkdir()
            catalogue = outside / "catalogue.csv"
            catalogue.write_bytes(b"reviewer work")
            redirect_directory(root / "output", outside)

            with self.assertRaises(ValueError):
                rollback_owned_file(root / "output/catalogue.csv", hashlib.sha256(b"published").hexdigest(), None, "Catalogue")
            self.assertEqual(catalogue.read_bytes(), b"reviewer work")
            self.assertEqual([path.name for path in outside.iterdir()], ["catalogue.csv"])

    def test_backup_restore_does_not_create_a_file_under_a_redirected_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            outside = root / "outside"
            outside.mkdir()
            backup = root / "backup.csv"
            backup.write_bytes(b"previous catalogue")
            redirect_directory(root / "output", outside)

            with self.assertRaises(ValueError):
                restore_backup_no_clobber(backup, root / "output/catalogue.csv", "Catalogue")
            self.assertEqual(list(outside.iterdir()), [])
            self.assertEqual(backup.read_bytes(), b"previous catalogue")
