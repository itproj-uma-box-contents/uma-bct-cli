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
from uma_bct.cli import discover_images, main, parse_args, validate_output_target
from uma_bct.domain import BoxResult
from uma_bct.path_safety import redirected_path_component
from uma_bct.run_artifacts import write_json_atomic


class FakeBatchPipeline:
    def __init__(self, failing_name=None, empty=False):
        self.failing_name = failing_name
        self.empty = empty
        self.work_directory = Path("crops")
        self.model_metadata = {"model": "qwen-test", "model_digest": "sha256:test", "ollama_host": "http://127.0.0.1:11434"}

    def process(self, path):
        if path.name == self.failing_name:
            raise RuntimeError("model failed")
        if self.empty:
            return make_result(status="failed")
        image = make_image(path)
        box = make_box(image=image, box_id=f"{path.stem}:001", crop_path=self.work_directory / path.stem / "box_001.png")
        return make_result(BoxResult(box, make_transcription(box)))


def arguments(root: Path, *, force=False):
    return argparse.Namespace(
        detector="qwen",
        transcriber="qwen",
        model="qwen-test",
        min_transcription_confidence=80,
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
        self.assertEqual(defaults.min_detector_confidence, 0.6)
        self.assertEqual(defaults.min_transcription_confidence, 80)
        self.assertEqual(aliases.path_to_outputs, Path("other.csv"))
        self.assertEqual(aliases.path_to_box_images, Path("review"))

    def test_confidence_options_keep_their_distinct_meanings(self):
        legacy_detector = parse_args(["--min-confidence", "0.7"])
        explicit_detector = parse_args(["--min-detector-confidence", "0.8"])
        short_transcription = parse_args(["-c", "70"])
        explicit_transcription = parse_args(["--min-transcription-confidence", "75"])

        self.assertEqual(legacy_detector.min_detector_confidence, 0.7)
        self.assertEqual(explicit_detector.min_detector_confidence, 0.8)
        self.assertEqual(short_transcription.min_transcription_confidence, 70)
        self.assertEqual(explicit_transcription.min_transcription_confidence, 75)

    def test_discovers_supported_images_in_stable_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "b.jpg").write_bytes(b"x")
            (root / "A.JPG").write_bytes(b"x")
            (root / "notes.txt").write_text("ignore", encoding="utf-8")
            self.assertEqual([path.name for path in discover_images(root)], ["A.JPG", "b.jpg"])

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

    def test_rejects_a_redirected_parent_of_a_new_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            inputs = root / "inputs"
            inputs.mkdir()
            (inputs / "shelf.jpg").write_bytes(b"x")
            outside = root / "outside"
            nested = outside / "nested"
            nested.mkdir(parents=True)
            redirected = root / "outputs"
            try:
                redirected.symlink_to(outside, target_is_directory=True)
            except OSError as error:
                self.skipTest(str(error))

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
                        str(redirected / "nested/catalogue.csv"),
                        "--work-dir",
                        str(root / "review-crops"),
                    ]
                )
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("parent must not be a symbolic link", stderr.getvalue())
            self.assertFalse((nested / "catalogue.csv").exists())
            self.assertFalse((nested / "catalogue.run.json").exists())
            batch.assert_not_called()

    def test_rejects_a_redirected_parent_of_a_new_crop_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            inputs = root / "inputs"
            inputs.mkdir()
            (inputs / "shelf.jpg").write_bytes(b"x")
            outside = root / "outside"
            nested = outside / "nested"
            nested.mkdir(parents=True)
            redirected = root / "review"
            try:
                redirected.symlink_to(outside, target_is_directory=True)
            except OSError as error:
                self.skipTest(str(error))

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
                        str(root / "outputs/catalogue.csv"),
                        "--work-dir",
                        str(redirected / "nested/crops"),
                    ]
                )
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("parent must not be a symbolic link", stderr.getvalue())
            self.assertFalse((nested / "crops").exists())
            batch.assert_not_called()

    def test_manifest_writer_rechecks_a_parent_swapped_after_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            outputs = root / "outputs"
            outputs.mkdir()
            manifest = outputs / "catalogue.run.json"
            validate_output_target(manifest, directory=False, purpose="Run manifest")

            original = root / "outputs-original"
            outside = root / "outside"
            outside.mkdir()
            outputs.rename(original)
            outputs.symlink_to(outside, target_is_directory=True)

            with self.assertRaisesRegex(ValueError, "parent must not be a symbolic"):
                write_json_atomic(manifest, {"status": "failed"})
            self.assertFalse((outside / "catalogue.run.json").exists())

    def test_manifest_writer_does_not_replace_a_new_unexpected_file(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory).resolve() / "catalogue.run.json"
            manifest.write_text('{"owner": "other process"}\n', encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "already exists"):
                write_json_atomic(manifest, {"status": "failed"}, replace_existing=False)
            self.assertEqual(manifest.read_text(encoding="utf-8"), '{"owner": "other process"}\n')

    def test_parent_reparse_detection_is_not_platform_dependent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            parent = root / "outputs"
            parent.mkdir()
            target = parent / "catalogue.csv"

            def fake_kind(path, metadata):
                return "reparse point" if path == parent else None

            with patch("uma_bct.path_safety.redirected_path_kind", side_effect=fake_kind):
                self.assertEqual(redirected_path_component(target), (parent, "reparse point"))

    def test_routes_setup_without_starting_a_batch(self):
        with patch("uma_bct.cli.run_setup", return_value=0) as setup:
            self.assertEqual(main(["setup"]), 0)
        setup.assert_called_once_with()

    def test_cli_rejects_xlsx_before_processing(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            main(["--path-to-outputs", "catalogue.xlsx"])
        self.assertEqual(raised.exception.code, 2)

    def test_cli_routes_a_valid_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            inputs = root / "inputs"
            inputs.mkdir()
            (inputs / "shelf.jpg").write_bytes(b"x")
            output = root / "outputs/catalogue.csv"
            with patch("uma_bct.cli.run_batch", return_value=0) as batch:
                status = main(["-i", str(inputs), "-o", str(output), "--work-dir", str(root / "outputs/crops")])
            self.assertEqual(status, 0)
            self.assertEqual(batch.call_args.args[1], [inputs / "shelf.jpg"])

    def test_cli_rejects_catalogue_inside_review_crop_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
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
                main(["-i", str(inputs), "-o", str(crops / "catalogue.csv"), "--work-dir", str(crops)])
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("catalogue output must not be inside the review-crop directory", stderr.getvalue())
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
            root = Path(directory).resolve()
            image = root / "inputs/shelf.jpg"
            status, summary, progress = self.run_with(root, FakeBatchPipeline(), [image])
            self.assertEqual(status, 0)
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["catalogue_records"], 1)
            self.assertTrue((root / "outputs/catalogue.csv").is_file())
            manifest = json.loads((root / "outputs/catalogue.run.json").read_text(encoding="utf-8"))
            self.assertNotIn("started_at", manifest)
            self.assertNotIn("attempts", str(manifest))
            self.assertEqual(manifest["configuration"]["minimum_transcription_confidence"], 80)
            self.assertEqual(manifest["configuration"]["minimum_detector_confidence"], 0.6)
            self.assertIn("Processing", progress)

    def test_continues_after_one_image_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            images = [root / "good.jpg", root / "bad.jpg"]
            status, summary, progress = self.run_with(root, FakeBatchPipeline(failing_name="bad.jpg"), images)
            self.assertEqual(status, 1)
            self.assertEqual(summary["status"], "partial")
            self.assertEqual(summary["catalogue_records"], 1)
            self.assertEqual(summary["failure_count"], 1)

    def test_empty_forced_run_preserves_previous_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            outputs = root / "outputs"
            outputs.mkdir()
            catalogue = outputs / "catalogue.csv"
            manifest = outputs / "catalogue.run.json"
            catalogue.write_text("old catalogue", encoding="utf-8")
            manifest.write_text("old manifest", encoding="utf-8")
            status, summary, progress = self.run_with(
                root, FakeBatchPipeline(empty=True), [root / "empty.jpg"], force=True, preserve=True
            )
            self.assertEqual(status, 1)
            self.assertEqual(catalogue.read_text(encoding="utf-8"), "old catalogue")
            self.assertEqual(manifest.read_text(encoding="utf-8"), "old manifest")
            self.assertIn(".failed-", summary["run_manifest"])

    def test_forced_success_keeps_timestamped_previous_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            outputs = root / "outputs"
            outputs.mkdir()
            catalogue = outputs / "catalogue.csv"
            manifest = outputs / "catalogue.run.json"
            catalogue.write_text("old catalogue", encoding="utf-8")
            manifest.write_text("old manifest", encoding="utf-8")
            status, summary, progress = self.run_with(root, FakeBatchPipeline(), [root / "shelf.jpg"], force=True)
            self.assertEqual(status, 0)
            self.assertEqual(len(summary["previous_results"]), 2)
            for path in summary["previous_results"]:
                self.assertTrue(Path(path).is_file())


if __name__ == "__main__":
    unittest.main()
