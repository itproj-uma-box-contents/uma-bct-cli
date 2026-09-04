import contextlib
import io
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from uma_bct.adapters.ollama_client import DEFAULT_MODEL
from uma_bct.setup_command import run_setup


class SetupCommandTest(unittest.TestCase):
    def test_creates_missing_directories_and_emits_one_json_result(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            model = Mock()
            model.ensure_model.return_value = {
                "model": DEFAULT_MODEL,
                "model_digest": "sha256:test",
                "ollama_version": "0.12.8",
                "ollama_host": "http://127.0.0.1:11434",
            }
            stdout = io.StringIO()

            with (
                patch("uma_bct.setup_command.OllamaVisionModel", return_value=model),
                contextlib.redirect_stdout(stdout),
            ):
                exit_code = run_setup(root=root)

            result = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertTrue((root / "inputs").is_dir())
            self.assertTrue((root / "outputs").is_dir())
            if os.name != "nt":
                self.assertEqual(stat.S_IMODE((root / "inputs").stat().st_mode), 0o700)
                self.assertEqual(
                    stat.S_IMODE((root / "outputs").stat().st_mode),
                    0o700,
                )
            self.assertEqual(
                result,
                {
                    "schema_version": 1,
                    "command": "setup",
                    "status": "complete",
                    "input_directory": str(root / "inputs"),
                    "output_directory": str(root / "outputs"),
                    "model": DEFAULT_MODEL,
                },
            )
            model.ensure_model.assert_called_once()

    def test_preserves_existing_directory_contents_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            inputs = root / "inputs"
            outputs = root / "outputs"
            inputs.mkdir()
            outputs.mkdir()
            input_marker = inputs / "keep.txt"
            output_marker = outputs / "keep.txt"
            input_marker.write_text("input", encoding="utf-8")
            output_marker.write_text("output", encoding="utf-8")
            model = Mock()
            model.ensure_model.return_value = {"model": DEFAULT_MODEL}

            with (
                patch("uma_bct.setup_command.OllamaVisionModel", return_value=model),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                first_exit = run_setup(root=root)
                second_exit = run_setup(root=root)

            self.assertEqual((first_exit, second_exit), (0, 0))
            self.assertEqual(input_marker.read_text(encoding="utf-8"), "input")
            self.assertEqual(output_marker.read_text(encoding="utf-8"), "output")
            self.assertEqual(model.ensure_model.call_count, 2)

    @unittest.skipIf(os.name == "nt", "POSIX permission bits are not available")
    def test_warns_without_changing_broad_existing_directory_permissions(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            inputs = root / "inputs"
            outputs = root / "outputs"
            inputs.mkdir(mode=0o755)
            outputs.mkdir(mode=0o700)
            inputs.chmod(0o755)
            model = Mock()
            model.ensure_model.return_value = {"model": DEFAULT_MODEL}
            stderr = io.StringIO()

            with (
                patch("uma_bct.setup_command.OllamaVisionModel", return_value=model),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(stderr),
            ):
                exit_code = run_setup(root=root)

            self.assertEqual(exit_code, 0)
            self.assertEqual(inputs.stat().st_mode & 0o777, 0o755)
            self.assertIn("mode 700", stderr.getvalue())

    def test_setup_failure_is_actionable_json_and_stderr(self):
        for message in (
            "Cannot connect to the local Ollama server. Start Ollama.",
            "Ollama 0.12.7 or newer is required; server reported 0.11.0.",
        ):
            with self.subTest(message=message):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    root = Path(temporary_directory)
                    model = Mock()
                    model.ensure_model.side_effect = RuntimeError(message)
                    stdout = io.StringIO()
                    stderr = io.StringIO()

                    with (
                        patch(
                            "uma_bct.setup_command.OllamaVisionModel",
                            return_value=model,
                        ),
                        contextlib.redirect_stdout(stdout),
                        contextlib.redirect_stderr(stderr),
                    ):
                        exit_code = run_setup(root=root)

                    result = json.loads(stdout.getvalue())
                    self.assertEqual(exit_code, 2)
                    self.assertEqual(result["status"], "failed")
                    self.assertEqual(result["error"], message)
                    self.assertIn(message, stderr.getvalue())
                    self.assertTrue((root / "inputs").is_dir())
                    self.assertTrue((root / "outputs").is_dir())

    def test_rejects_an_unsafe_endpoint_before_network_access(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            stdout = io.StringIO()
            with (
                patch.dict(
                    "os.environ",
                    {"OLLAMA_HOST": "http://models.example.test:11434"},
                    clear=True,
                ),
                patch(
                    "uma_bct.adapters.ollama_client.OllamaVisionModel._client"
                ) as client,
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                exit_code = run_setup(root=Path(temporary_directory))

            result = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 2)
            self.assertIn("sends archive image data off-device", result["error"])
            client.assert_not_called()

    def test_sanitizes_and_bounds_errors_in_json_and_stderr(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            model = Mock()
            model.ensure_model.side_effect = RuntimeError(
                "\x1b[31mfirst line\nsecond line " + "x" * 2200
            )
            stdout = io.StringIO()
            stderr = io.StringIO()

            with (
                patch("uma_bct.setup_command.OllamaVisionModel", return_value=model),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                exit_code = run_setup(root=Path(temporary_directory))

            error = json.loads(stdout.getvalue())["error"]
            self.assertEqual(exit_code, 2)
            self.assertLessEqual(len(error), 2000)
            self.assertNotIn("\x1b", error)
            self.assertNotIn("\n", error)
            self.assertNotIn("\x1b", stderr.getvalue())

    def test_progress_stays_on_stderr(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            model = Mock()

            def report_progress(*, progress_stream):
                print("Downloading model: 50%", file=progress_stream)
                return {"model": DEFAULT_MODEL}

            model.ensure_model.side_effect = report_progress
            stdout = io.StringIO()
            stderr = io.StringIO()

            with (
                patch("uma_bct.setup_command.OllamaVisionModel", return_value=model),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                exit_code = run_setup(root=Path(temporary_directory))

            self.assertEqual(exit_code, 0)
            self.assertEqual(json.loads(stdout.getvalue())["command"], "setup")
            self.assertNotIn("Downloading", stdout.getvalue())
            self.assertIn("Downloading model: 50%", stderr.getvalue())

    def test_rejects_existing_non_directory_without_replacing_it(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "inputs"
            input_path.write_text("keep me", encoding="utf-8")
            stdout = io.StringIO()

            with (
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                exit_code = run_setup(root=root)

            self.assertEqual(exit_code, 2)
            self.assertEqual(input_path.read_text(encoding="utf-8"), "keep me")
            self.assertIn("not a directory", json.loads(stdout.getvalue())["error"])

    def test_rejects_an_inaccessible_existing_directory_before_model_setup(self):
        for inaccessible_name in ("inputs", "outputs"):
            with self.subTest(inaccessible_name=inaccessible_name):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    root = Path(temporary_directory)
                    (root / "inputs").mkdir()
                    (root / "outputs").mkdir()
                    stdout = io.StringIO()

                    def accessible(path, denied=inaccessible_name):
                        if Path(path).name == denied:
                            raise RuntimeError(
                                "Setup directory must be readable, writable, "
                                f"and searchable: {path}"
                            )

                    with (
                        patch(
                            "uma_bct.setup_command.ensure_directory_access",
                            side_effect=accessible,
                        ),
                        patch("uma_bct.setup_command.OllamaVisionModel") as model,
                        contextlib.redirect_stdout(stdout),
                        contextlib.redirect_stderr(io.StringIO()),
                    ):
                        exit_code = run_setup(root=root)

                    result = json.loads(stdout.getvalue())
                    self.assertEqual(exit_code, 2)
                    self.assertEqual(result["status"], "failed")
                    self.assertIn(
                        "readable, writable, and searchable",
                        result["error"],
                    )
                    model.assert_not_called()

    def test_rejects_an_existing_symbolic_link_without_touching_its_target(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            target = root / "real-inputs"
            target.mkdir()
            marker = target / "keep.txt"
            marker.write_text("keep me", encoding="utf-8")
            try:
                (root / "inputs").symlink_to(target, target_is_directory=True)
            except OSError as error:
                self.skipTest(f"symbolic links are unavailable: {error}")
            stdout = io.StringIO()

            with (
                patch("uma_bct.setup_command.OllamaVisionModel") as model,
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                exit_code = run_setup(root=root)

            self.assertEqual(exit_code, 2)
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep me")
            self.assertIn("symbolic link", json.loads(stdout.getvalue())["error"])
            model.assert_not_called()

    def test_rejects_a_directory_junction_before_model_access(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "inputs").mkdir()
            stdout = io.StringIO()

            with (
                patch("pathlib.Path.is_junction", return_value=True),
                patch("uma_bct.setup_command.OllamaVisionModel") as model,
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                exit_code = run_setup(root=root)

            self.assertEqual(exit_code, 2)
            self.assertIn("directory junction", json.loads(stdout.getvalue())["error"])
            model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
