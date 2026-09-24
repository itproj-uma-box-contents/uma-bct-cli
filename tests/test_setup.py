import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from uma_bct.run_artifacts import catalogue_lock_path
from uma_bct.setup_command import ensure_directory, run_setup


# Group the setup checks.
class SetupTest(unittest.TestCase):
    # Check that the code creates private work folders and checks the model.
    def test_creates_private_work_folders_and_checks_the_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = Mock()
            stdout = io.StringIO()
            with patch("uma_bct.setup_command.OllamaVisionModel", return_value=model), contextlib.redirect_stdout(stdout):
                status = run_setup(root=root, model_name="qwen-test")
            self.assertEqual(status, 0)
            self.assertTrue((root / "inputs").is_dir())
            self.assertTrue((root / "outputs").is_dir())
            self.assertEqual(json.loads(stdout.getvalue())["status"], "complete")
            model.ensure_model.assert_called_once()
            self.assertFalse(catalogue_lock_path(root / "outputs/catalogue.csv").exists())

    # Check that prepares custom friendly directories.
    def test_prepares_custom_friendly_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "Add Photos Here"
            results = root / "Results"
            with patch("uma_bct.setup_command.OllamaVisionModel"), contextlib.redirect_stdout(io.StringIO()):
                status = run_setup(input_directory=inputs, results_directory=results)
                repeated_status = run_setup(input_directory=inputs, results_directory=results)
            self.assertEqual(status, 0)
            self.assertEqual(repeated_status, 0)
            self.assertTrue(inputs.is_dir())
            self.assertTrue(results.is_dir())
            self.assertTrue((results / ".uma-bct.lock").is_file())

    # Check that setup failure is JSON and stderr.
    def test_setup_failure_is_json_and_stderr(self):
        with tempfile.TemporaryDirectory() as directory:
            stdout, stderr = io.StringIO(), io.StringIO()
            with (
                patch("uma_bct.setup_command.OllamaVisionModel", side_effect=RuntimeError("server unavailable")),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                status = run_setup(root=Path(directory))
            self.assertEqual(status, 2)
            self.assertEqual(json.loads(stdout.getvalue())["status"], "failed")
            self.assertIn("server unavailable", stderr.getvalue())
            self.assertFalse(catalogue_lock_path(Path(directory) / "outputs/catalogue.csv").exists())

    # Check that the code rejects a redirected setup directory.
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

    # Check that the code rejects a setup directory beneath a redirected parent.
    def test_rejects_a_setup_directory_beneath_a_redirected_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            outside = root / "outside"
            outside.mkdir()
            linked = root / "linked"
            try:
                linked.symlink_to(outside, target_is_directory=True)
            except OSError as error:
                self.skipTest(str(error))

            with self.assertRaisesRegex(RuntimeError, "parent must not be a symbolic link"):
                ensure_directory(linked / "inputs")

    # Check that the code rejects a setup root with redirected component before dot dot.
    def test_rejects_a_setup_root_with_redirected_component_before_dot_dot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            outside = root / "outside"
            (outside / "child").mkdir(parents=True)
            linked = root / "linked"
            try:
                linked.symlink_to(outside / "child", target_is_directory=True)
            except OSError as error:
                self.skipTest(str(error))
            model = Mock()
            stdout = io.StringIO()

            with (
                patch("uma_bct.setup_command.OllamaVisionModel", return_value=model),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                status = run_setup(root=linked / "..")

            self.assertEqual(status, 2)
            self.assertEqual(json.loads(stdout.getvalue())["status"], "failed")
            self.assertFalse((outside / "inputs").exists())
            model.ensure_model.assert_not_called()

    # Check that setup rechecks paths after directory creation.
    def test_setup_rechecks_paths_after_directory_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with (
                patch("uma_bct.setup_command.check_path") as check,
                patch("uma_bct.setup_command.OllamaVisionModel") as model,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(run_setup(root=root), 0)

            self.assertEqual(check.call_count, 4)
            model.return_value.ensure_model.assert_called_once()

    # Check that setup holds the catalogue lock during model work.
    def test_setup_holds_the_catalogue_lock_during_model_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            lock_path = catalogue_lock_path(root / "outputs/catalogue.csv")
            model = Mock()
            model.ensure_model.side_effect = lambda **_kwargs: self.assertTrue(lock_path.is_file())

            with (
                patch("uma_bct.setup_command.OllamaVisionModel", return_value=model),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(run_setup(root=root), 0)

            self.assertFalse(lock_path.exists())

    # Check that setup rejects an existing lock before model work.
    def test_setup_rejects_an_existing_lock_before_model_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            outputs = root / "outputs"
            outputs.mkdir()
            lock_path = catalogue_lock_path(outputs / "catalogue.csv")
            lock_path.write_text('{"pid": 1234}\n', encoding="utf-8")
            model = Mock()
            stdout = io.StringIO()

            with (
                patch("uma_bct.setup_command.OllamaVisionModel", return_value=model),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(run_setup(root=root), 2)

            self.assertEqual(json.loads(stdout.getvalue())["status"], "failed")
            self.assertTrue(lock_path.exists())
            model.ensure_model.assert_not_called()

    # Check that setup lock cleanup failure keeps JSON and returns two.
    def test_setup_lock_cleanup_failure_keeps_json_and_returns_two(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            stdout, stderr = io.StringIO(), io.StringIO()
            with (
                patch("uma_bct.setup_command.OllamaVisionModel"),
                patch("uma_bct.setup_command.CatalogueLock.release", side_effect=OSError("cleanup denied")),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                status = run_setup(root=root)

            self.assertEqual(status, 2)
            self.assertEqual(json.loads(stdout.getvalue())["status"], "failed")
            self.assertIn("Lock cleanup failed", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
