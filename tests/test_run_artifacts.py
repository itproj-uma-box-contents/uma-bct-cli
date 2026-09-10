import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from uma_bct.run_artifacts import CatalogueLock, catalogue_lock_path


class CatalogueLockTest(unittest.TestCase):
    def test_lock_is_exclusive_and_records_the_owner_pid(self):
        with tempfile.TemporaryDirectory() as directory:
            catalogue = Path(directory) / "outputs/catalogue.csv"
            first = CatalogueLock(catalogue)
            first.acquire()
            lock_path = catalogue_lock_path(catalogue)
            try:
                self.assertEqual(json.loads(lock_path.read_text(encoding="utf-8")), {"pid": os.getpid()})
                with self.assertRaisesRegex(RuntimeError, rf"PID {os.getpid()}.*remove .*\.uma-bct\.lock"):
                    CatalogueLock(catalogue).acquire()
            finally:
                first.release()
            self.assertFalse(lock_path.exists())
            self.assertEqual(list(lock_path.parent.glob(".uma-bct-lock-release-*")), [])

    def test_invalid_stale_lock_is_rejected_with_safe_recovery_instructions(self):
        with tempfile.TemporaryDirectory() as directory:
            catalogue = Path(directory) / "catalogue.csv"
            lock_path = catalogue_lock_path(catalogue)
            lock_path.write_text("not valid JSON\nprivate text", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, r"PID unavailable.*remove .*\.uma-bct\.lock") as raised:
                CatalogueLock(catalogue).acquire()

            self.assertNotIn("private text", str(raised.exception))
            self.assertTrue(lock_path.exists())

    def test_release_does_not_remove_a_replaced_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            catalogue = Path(directory) / "catalogue.csv"
            lock = CatalogueLock(catalogue)
            lock.acquire()
            lock_path = catalogue_lock_path(catalogue)
            original = lock_path.with_name("original.lock")
            lock_path.replace(original)
            lock_path.write_text('{"pid": 999}\n', encoding="utf-8")

            with patch("uma_bct.run_artifacts.os.replace") as replace:
                lock.release()

            replace.assert_not_called()
            self.assertTrue(lock_path.exists())
            self.assertEqual(json.loads(lock_path.read_text(encoding="utf-8")), {"pid": 999})
            self.assertTrue(original.exists())
            with self.assertRaisesRegex(RuntimeError, "PID 999"):
                CatalogueLock(catalogue).acquire()

    @unittest.skipIf(os.name == "nt", "FIFO fixture requires POSIX")
    def test_non_regular_stale_lock_does_not_block(self):
        with tempfile.TemporaryDirectory() as directory:
            catalogue = Path(directory) / "catalogue.csv"
            lock_path = catalogue_lock_path(catalogue)
            os.mkfifo(lock_path)

            with self.assertRaisesRegex(RuntimeError, "PID unavailable"):
                CatalogueLock(catalogue).acquire()

    def test_release_does_not_unlink_a_replacement_swapped_at_cleanup_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            catalogue = Path(directory) / "catalogue.csv"
            lock = CatalogueLock(catalogue)
            lock.acquire()
            lock_path = catalogue_lock_path(catalogue)
            original = lock_path.with_name("original.lock")
            real_replace = os.replace
            swapped = False

            def swap_before_replace(source, destination):
                nonlocal swapped
                if Path(source) == lock_path and not swapped:
                    real_replace(source, original)
                    lock_path.write_text('{"pid": 999}\n', encoding="utf-8")
                    swapped = True
                return real_replace(source, destination)

            with patch("uma_bct.run_artifacts.os.replace", side_effect=swap_before_replace):
                lock.release()

            self.assertTrue(swapped)
            self.assertEqual(json.loads(lock_path.read_text(encoding="utf-8")), {"pid": 999})
            self.assertEqual(list(lock_path.parent.glob(".uma-bct-lock-release-*")), [])


if __name__ == "__main__":
    unittest.main()
