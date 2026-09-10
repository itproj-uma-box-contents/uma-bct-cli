import json
import os
import tempfile
import unittest
from pathlib import Path

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

            lock.release()

            self.assertTrue(lock_path.exists())
            self.assertEqual(json.loads(lock_path.read_text(encoding="utf-8")), {"pid": 999})
            self.assertTrue(original.exists())


if __name__ == "__main__":
    unittest.main()
