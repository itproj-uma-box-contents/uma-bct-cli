import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from uma_bct.setup_command import ensure_directory, run_setup


class SetupTest(unittest.TestCase):
    def test_creates_private_work_folders_and_checks_the_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = Mock()
            stdout = io.StringIO()
            with (
                patch("uma_bct.setup_command.OllamaVisionModel", return_value=model),
                contextlib.redirect_stdout(stdout),
            ):
                status = run_setup(root=root, model_name="qwen-test")
            self.assertEqual(status, 0)
            self.assertTrue((root / "inputs").is_dir())
            self.assertTrue((root / "outputs").is_dir())
            self.assertEqual(json.loads(stdout.getvalue())["status"], "complete")
            model.ensure_model.assert_called_once()

    def test_setup_failure_is_json_and_stderr(self):
        with tempfile.TemporaryDirectory() as directory:
            stdout, stderr = io.StringIO(), io.StringIO()
            with (
                patch(
                    "uma_bct.setup_command.OllamaVisionModel",
                    side_effect=RuntimeError("server unavailable"),
                ),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                status = run_setup(root=Path(directory))
            self.assertEqual(status, 2)
            self.assertEqual(json.loads(stdout.getvalue())["status"], "failed")
            self.assertIn("server unavailable", stderr.getvalue())

    def test_rejects_a_redirected_setup_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target"
            linked = root / "linked"
            target.mkdir()
            try:
                linked.symlink_to(target, target_is_directory=True)
            except OSError as error:
                self.skipTest(str(error))
            with self.assertRaisesRegex(RuntimeError, "symbolic link"):
                ensure_directory(linked)


if __name__ == "__main__":
    unittest.main()
