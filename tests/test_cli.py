import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from helpers import make_box, make_image, make_result, make_transcription

from uma_bct.batch_runner import run_batch
from uma_bct.cli import discover_images, main, parse_args
from uma_bct.domain import BoxResult


class FakeBatchPipeline:
    def __init__(self, failing_name=None, empty=False):
        self.failing_name = failing_name
        self.empty = empty
        self.work_directory = Path("crops")
        self.model_metadata = {
            "model": "qwen-test",
            "model_digest": "sha256:test",
            "ollama_host": "http://127.0.0.1:11434",
        }

    def process(self, path):
        if path.name == self.failing_name:
            raise RuntimeError("model failed")
        if self.empty:
            return make_result(status="failed")
        image = make_image(path)
        box = make_box(
            image=image,
            box_id=f"{path.stem}:001",
            crop_path=self.work_directory / path.stem / "box_001.png",
        )
        return make_result(BoxResult(box, make_transcription(box)))


def arguments(root: Path, *, force=False):
    return argparse.Namespace(
        detector="qwen",
        transcriber="qwen",
        model="qwen-test",
        min_confidence=80,
        min_detector_confidence=0.6,
        request_timeout=10.0,
        path_to_inputs=root / "inputs",
        path_to_outputs=root / "outputs/catalogue.csv",
        path_to_box_images=root / "outputs/crops",
        force=force,
    )


class CliTest(unittest.TestCase):
    def test_parser_keeps_defaults_and_existing_aliases(self):
        defaults = parse_args([])
        aliases = parse_args(["--path-to-output", "other.csv", "--work-dir", "review"])
        self.assertEqual(defaults.path_to_inputs, Path("inputs"))
        self.assertEqual(defaults.path_to_outputs, Path("outputs/catalogue.csv"))
        self.assertEqual(aliases.path_to_outputs, Path("other.csv"))
        self.assertEqual(aliases.path_to_box_images, Path("review"))

    def test_discovers_supported_images_in_stable_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "b.jpg").write_bytes(b"x")
            (root / "A.JPG").write_bytes(b"x")
            (root / "notes.txt").write_text("ignore", encoding="utf-8")
            self.assertEqual(
                [path.name for path in discover_images(root)],
                ["A.JPG", "b.jpg"],
            )

    def test_rejects_redirected_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jpg"
            linked = root / "linked.jpg"
            source.write_bytes(b"x")
            try:
                linked.symlink_to(source)
            except OSError as error:
                self.skipTest(str(error))
            with self.assertRaisesRegex(ValueError, "symbolic link"):
                discover_images(linked)

    def test_routes_setup_without_starting_a_batch(self):
        with patch("uma_bct.cli.run_setup", return_value=0) as setup:
            self.assertEqual(main(["setup"]), 0)
        setup.assert_called_once_with()

    def test_cli_rejects_xlsx_before_processing(self):
        with (
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as raised,
        ):
            main(["--path-to-outputs", "catalogue.xlsx"])
        self.assertEqual(raised.exception.code, 2)

    def test_cli_routes_a_valid_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "inputs"
            inputs.mkdir()
            (inputs / "shelf.jpg").write_bytes(b"x")
            output = root / "outputs/catalogue.csv"
            with patch("uma_bct.cli.run_batch", return_value=0) as batch:
                status = main(
                    [
                        "-i",
                        str(inputs),
                        "-o",
                        str(output),
                        "--work-dir",
                        str(root / "outputs/crops"),
                    ]
                )
            self.assertEqual(status, 0)
            self.assertEqual(batch.call_args.args[1], [inputs / "shelf.jpg"])

    def test_cli_rejects_catalogue_inside_review_crop_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "inputs"
            inputs.mkdir()
            (inputs / "shelf.jpg").write_bytes(b"x")
            crops = root / "outputs/crops"
            stderr = io.StringIO()
            with (
                contextlib.redirect_stderr(stderr),
                patch("uma_bct.cli.run_batch") as batch,
                self.assertRaises(SystemExit) as raised,
            ):
                main(
                    [
                        "-i",
                        str(inputs),
                        "-o",
                        str(crops / "catalogue.csv"),
                        "--work-dir",
                        str(crops),
                    ]
                )
            self.assertEqual(raised.exception.code, 2)
            self.assertIn(
                "catalogue output must not be inside the review-crop directory",
                stderr.getvalue(),
            )
            batch.assert_not_called()


class BatchRunnerTest(unittest.TestCase):
    def run_with(self, root, pipeline, images, *, force=False, preserve=False):
        args = arguments(root, force=force)
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            patch("uma_bct.batch_runner.build_pipeline", return_value=pipeline),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            status = run_batch(
                args,
                images,
                args.path_to_outputs,
                root / "outputs/catalogue.run.json",
                preserve_existing_manifest_on_failure=preserve,
            )
        return status, json.loads(stdout.getvalue()), stderr.getvalue()

    def test_writes_csv_manifest_and_clean_json_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "inputs/shelf.jpg"
            status, summary, progress = self.run_with(
                root, FakeBatchPipeline(), [image]
            )
            self.assertEqual(status, 0)
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["catalogue_records"], 1)
            self.assertTrue((root / "outputs/catalogue.csv").is_file())
            manifest = json.loads(
                (root / "outputs/catalogue.run.json").read_text(encoding="utf-8")
            )
            self.assertNotIn("started_at", manifest)
            self.assertNotIn("attempts", str(manifest))
            self.assertIn("Processing", progress)

    def test_continues_after_one_image_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = [root / "good.jpg", root / "bad.jpg"]
            status, summary, _progress = self.run_with(
                root, FakeBatchPipeline(failing_name="bad.jpg"), images
            )
            self.assertEqual(status, 1)
            self.assertEqual(summary["status"], "partial")
            self.assertEqual(summary["catalogue_records"], 1)
            self.assertEqual(summary["failure_count"], 1)

    def test_empty_forced_run_preserves_previous_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = root / "outputs"
            outputs.mkdir()
            catalogue = outputs / "catalogue.csv"
            manifest = outputs / "catalogue.run.json"
            catalogue.write_text("old catalogue", encoding="utf-8")
            manifest.write_text("old manifest", encoding="utf-8")
            status, summary, _progress = self.run_with(
                root,
                FakeBatchPipeline(empty=True),
                [root / "empty.jpg"],
                force=True,
                preserve=True,
            )
            self.assertEqual(status, 1)
            self.assertEqual(catalogue.read_text(encoding="utf-8"), "old catalogue")
            self.assertEqual(manifest.read_text(encoding="utf-8"), "old manifest")
            self.assertIn(".failed-", summary["run_manifest"])

    def test_forced_success_keeps_timestamped_previous_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = root / "outputs"
            outputs.mkdir()
            catalogue = outputs / "catalogue.csv"
            manifest = outputs / "catalogue.run.json"
            catalogue.write_text("old catalogue", encoding="utf-8")
            manifest.write_text("old manifest", encoding="utf-8")
            status, summary, _progress = self.run_with(
                root,
                FakeBatchPipeline(),
                [root / "shelf.jpg"],
                force=True,
            )
            self.assertEqual(status, 0)
            self.assertEqual(len(summary["previous_results"]), 2)
            for path in summary["previous_results"]:
                self.assertTrue(Path(path).is_file())


if __name__ == "__main__":
    unittest.main()
