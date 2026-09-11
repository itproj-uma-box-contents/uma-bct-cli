import argparse
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from helpers import make_box, make_image, make_result, make_transcription

from uma_bct.atomic import rename_exclusive
from uma_bct.batch_runner import run_batch
from uma_bct.cli import discover_images, main, parse_args, validate_output_layout, validate_output_target
from uma_bct.domain import BoxResult
from uma_bct.path_safety import check_path
from uma_bct.run_artifacts import backup_existing_results, catalogue_lock_path, write_json_atomic


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

    def test_rejects_an_input_beneath_a_redirected_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            outside = root / "outside"
            outside.mkdir()
            (outside / "shelf.jpg").write_bytes(b"x")
            linked = root / "linked"
            try:
                linked.symlink_to(outside, target_is_directory=True)
            except OSError as error:
                self.skipTest(str(error))

            with self.assertRaisesRegex(ValueError, "parent must not be a symbolic link"):
                discover_images(linked / "shelf.jpg")

    def test_rejects_a_redirected_component_before_dot_dot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            outside = root / "outside"
            (outside / "escaped").mkdir(parents=True)
            (outside / "child").mkdir()
            (outside / "escaped/shelf.jpg").write_bytes(b"private")
            linked = root / "linked"
            try:
                linked.symlink_to(outside / "child", target_is_directory=True)
            except OSError as error:
                self.skipTest(str(error))

            with self.assertRaisesRegex(ValueError, "symbolic link"):
                discover_images(linked / "../escaped/shelf.jpg")
            with self.assertRaisesRegex(ValueError, "symbolic link"):
                write_json_atomic(linked / "../escaped/catalogue.run.json", {"status": "failed"})
            self.assertFalse((outside / "escaped/catalogue.run.json").exists())

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

    def test_manifest_writer_does_not_replace_a_file_appearing_at_the_publication_syscall(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory).resolve() / "catalogue.run.json"

            def late_destination(source, destination):
                destination.write_text('{"owner": "late process"}\n', encoding="utf-8")
                return rename_exclusive(source, destination)

            with (
                patch("uma_bct.atomic._rename_exclusive", side_effect=late_destination),
                self.assertRaisesRegex(FileExistsError, "already exists"),
            ):
                write_json_atomic(manifest, {"status": "failed"}, replace_existing=False)

            self.assertEqual(manifest.read_text(encoding="utf-8"), '{"owner": "late process"}\n')

    def test_parent_reparse_detection_is_not_platform_dependent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            parent = root / "outputs"
            parent.mkdir()
            target = parent / "catalogue.csv"

            def fake_kind(path, metadata):
                return "reparse point" if path == parent else None

            with patch("uma_bct.path_safety.redirected_path_kind", side_effect=fake_kind):
                with self.assertRaisesRegex(ValueError, "reparse point"):
                    check_path(target, "Catalogue output")

    def test_path_check_wraps_reparse_inspection_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory).resolve() / "catalogue.csv"
            with (
                patch("uma_bct.path_safety.redirected_path_kind", side_effect=OSError("inspection failed")),
                self.assertRaisesRegex(ValueError, "Could not inspect Catalogue output"),
            ):
                check_path(target, "Catalogue output")

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

    def test_cli_rejects_review_crops_inside_catalogue_or_manifest_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            inputs = root / "inputs"
            inputs.mkdir()
            (inputs / "shelf.jpg").write_bytes(b"x")
            output = root / "outputs/catalogue.csv"
            invalid_crops = (
                output / "crops",
                root / "outputs/catalogue.run.json",
                root / "outputs/catalogue.run.json/crops",
            )

            for crops in invalid_crops:
                stderr = io.StringIO()
                with (
                    self.subTest(crops=crops),
                    contextlib.redirect_stderr(stderr),
                    patch("uma_bct.cli.run_batch") as batch,
                    self.assertRaises(SystemExit) as raised,
                ):
                    main(["-i", str(inputs), "-o", str(output), "--work-dir", str(crops)])
                self.assertEqual(raised.exception.code, 2)
                self.assertIn("review-crop", stderr.getvalue())
                batch.assert_not_called()

    def test_rejects_cross_drive_windows_output_layouts(self):
        with self.assertRaisesRegex(ValueError, "same Windows drive"):
            validate_output_layout(r"C:\results\catalogue.csv", r"C:\results\catalogue.run.json", r"D:\review\crops")

    @unittest.skipUnless(sys.platform == "darwin", "macOS system alias test")
    def test_output_layout_compares_physical_macos_alias_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            lexical_root = Path(directory)
            physical_root = lexical_root.resolve()
            if lexical_root == physical_root:
                self.skipTest("temporary path has no system alias")
            with self.assertRaisesRegex(ValueError, "inside the review-crop"):
                validate_output_layout(
                    physical_root / "crops/catalogue.csv", physical_root / "catalogue.run.json", lexical_root / "crops"
                )


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
            self.assertNotIn("recovery_path", manifest)
            self.assertNotIn("recovery_path", summary)
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

    def test_preflight_failure_is_reported_once_and_releases_the_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            args = arguments(root)
            stdout, stderr = io.StringIO(), io.StringIO()
            with (
                patch("uma_bct.batch_runner.build_pipeline", side_effect=RuntimeError("not enough memory")),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                status = run_batch(
                    args,
                    [root / "inputs/shelf.jpg"],
                    args.path_to_outputs,
                    root / "outputs/catalogue.run.json",
                    preserve_existing_manifest_on_failure=False,
                )

            summary = json.loads(stdout.getvalue())
            self.assertEqual(status, 2)
            self.assertEqual(summary["status"], "failed")
            self.assertEqual(summary["failure_count"], 1)
            self.assertIn("Preflight failed: not enough memory", stderr.getvalue())
            self.assertFalse(catalogue_lock_path(args.path_to_outputs).exists())

    def test_existing_lock_rejects_the_batch_before_model_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            args = arguments(root)
            lock_path = catalogue_lock_path(args.path_to_outputs)
            lock_path.parent.mkdir(parents=True)
            lock_path.write_text('{"pid": 1234}\n', encoding="utf-8")
            pipeline_builder = Mock()
            stdout, stderr = io.StringIO(), io.StringIO()

            with (
                patch("uma_bct.batch_runner.build_pipeline", pipeline_builder),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                status = run_batch(
                    args,
                    [root / "inputs/shelf.jpg"],
                    args.path_to_outputs,
                    root / "outputs/catalogue.run.json",
                    preserve_existing_manifest_on_failure=False,
                )

            self.assertEqual(status, 2)
            self.assertEqual(json.loads(stdout.getvalue())["failure_count"], 1)
            self.assertIn("PID 1234", stderr.getvalue())
            self.assertTrue(lock_path.exists())
            pipeline_builder.assert_not_called()

    def test_manifest_failure_restores_old_csv_and_writes_a_failed_run_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            outputs = root / "outputs"
            outputs.mkdir()
            catalogue = outputs / "catalogue.csv"
            manifest = outputs / "catalogue.run.json"
            catalogue.write_text("old catalogue", encoding="utf-8")
            manifest.write_text("old manifest", encoding="utf-8")
            real_writer = write_json_atomic

            def fail_normal_manifest(destination, value, *, replace_existing=True):
                if destination == manifest:
                    raise OSError("manifest disk failure")
                return real_writer(destination, value, replace_existing=replace_existing)

            with patch("uma_bct.batch_runner.write_json_atomic", side_effect=fail_normal_manifest):
                status, summary, progress = self.run_with(root, FakeBatchPipeline(), [root / "shelf.jpg"], force=True)

            self.assertEqual(status, 1)
            self.assertEqual(summary["status"], "failed")
            self.assertIsNone(summary["catalogue_output"])
            self.assertEqual(catalogue.read_text(encoding="utf-8"), "old catalogue")
            self.assertEqual(manifest.read_text(encoding="utf-8"), "old manifest")
            self.assertIn(".failed-", summary["run_manifest"])
            self.assertTrue(Path(summary["run_manifest"]).is_file())
            self.assertEqual(len(summary["previous_results"]), 2)
            self.assertIn("Manifest write failed", progress)

    def test_manifest_failure_removes_a_new_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            manifest = root / "outputs/catalogue.run.json"
            real_writer = write_json_atomic

            def fail_normal_manifest(destination, value, *, replace_existing=True):
                if destination == manifest:
                    raise OSError("manifest disk failure")
                return real_writer(destination, value, replace_existing=replace_existing)

            with patch("uma_bct.batch_runner.write_json_atomic", side_effect=fail_normal_manifest):
                status, summary, progress = self.run_with(root, FakeBatchPipeline(), [root / "shelf.jpg"])

            self.assertEqual(status, 1)
            self.assertEqual(summary["status"], "failed")
            self.assertFalse((root / "outputs/catalogue.csv").exists())
            self.assertTrue(Path(summary["run_manifest"]).is_file())

    def test_rollback_failure_reports_partial_and_retains_the_recovery_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            outputs = root / "outputs"
            outputs.mkdir()
            catalogue = outputs / "catalogue.csv"
            manifest = outputs / "catalogue.run.json"
            catalogue.write_text("old catalogue", encoding="utf-8")
            manifest.write_text("old manifest", encoding="utf-8")
            real_writer = write_json_atomic

            def fail_normal_manifest(destination, value, *, replace_existing=True):
                if destination == manifest:
                    raise OSError("manifest disk failure")
                return real_writer(destination, value, replace_existing=replace_existing)

            with (
                patch("uma_bct.batch_runner.write_json_atomic", side_effect=fail_normal_manifest),
                patch("uma_bct.batch_runner.rollback_catalogue", side_effect=OSError("rollback disk failure")),
            ):
                status, summary, progress = self.run_with(root, FakeBatchPipeline(), [root / "shelf.jpg"], force=True)

            self.assertEqual(status, 1)
            self.assertEqual(summary["status"], "partial")
            self.assertTrue(Path(summary["recovery_path"]).is_file())
            self.assertIn(summary["recovery_path"], summary["previous_results"])
            self.assertIn("Rollback failed", progress)

    def test_new_csv_rollback_failure_reports_the_csv_as_the_recovery_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            manifest = root / "outputs/catalogue.run.json"
            real_writer = write_json_atomic

            def fail_normal_manifest(destination, value, *, replace_existing=True):
                if destination == manifest:
                    raise OSError("manifest disk failure")
                return real_writer(destination, value, replace_existing=replace_existing)

            with (
                patch("uma_bct.batch_runner.write_json_atomic", side_effect=fail_normal_manifest),
                patch("uma_bct.batch_runner.rollback_catalogue", side_effect=OSError("rollback disk failure")),
            ):
                status, summary, progress = self.run_with(root, FakeBatchPipeline(), [root / "shelf.jpg"])

            self.assertEqual(status, 1)
            self.assertEqual(summary["status"], "partial")
            self.assertEqual(summary["recovery_path"], str(root / "outputs/catalogue.csv"))
            self.assertTrue(Path(summary["recovery_path"]).is_file())

    def test_recovery_manifest_failure_keeps_stdout_valid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            outputs = root / "outputs"
            outputs.mkdir()
            catalogue = outputs / "catalogue.csv"
            catalogue.write_text("old catalogue", encoding="utf-8")

            with patch("uma_bct.batch_runner.write_json_atomic", side_effect=OSError("manifest disk failure")):
                status, summary, progress = self.run_with(root, FakeBatchPipeline(), [root / "shelf.jpg"], force=True)

            self.assertEqual(status, 1)
            self.assertEqual(summary["status"], "failed")
            self.assertIsNone(summary["run_manifest"])
            self.assertGreaterEqual(summary["failure_count"], 2)
            self.assertEqual(catalogue.read_text(encoding="utf-8"), "old catalogue")

    def test_lock_cleanup_failure_keeps_json_and_returns_lock_error_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with patch("uma_bct.batch_runner.CatalogueLock.release", side_effect=OSError("cleanup denied")):
                status, summary, progress = self.run_with(root, FakeBatchPipeline(), [root / "shelf.jpg"])

            self.assertEqual(status, 2)
            self.assertEqual(summary["status"], "partial")
            self.assertEqual(summary["failure_count"], 1)
            self.assertIn("Lock cleanup failed", progress)
            self.assertIn(".failed-", summary["run_manifest"])
            recovery_manifest = json.loads(Path(summary["run_manifest"]).read_text(encoding="utf-8"))
            self.assertEqual(recovery_manifest["status"], summary["status"])
            self.assertEqual(recovery_manifest["failures"], [{"stage": "lock_cleanup", "error": "cleanup denied"}])

    def test_preflight_and_lock_cleanup_failures_still_emit_one_json_document(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            args = arguments(root)
            stdout, stderr = io.StringIO(), io.StringIO()
            with (
                patch("uma_bct.batch_runner.build_pipeline", side_effect=RuntimeError("preflight denied")),
                patch("uma_bct.batch_runner.CatalogueLock.release", side_effect=OSError("cleanup denied")),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                status = run_batch(
                    args,
                    [root / "inputs/shelf.jpg"],
                    args.path_to_outputs,
                    root / "outputs/catalogue.run.json",
                    preserve_existing_manifest_on_failure=False,
                )

            summary = json.loads(stdout.getvalue())
            self.assertEqual(status, 2)
            self.assertEqual(summary["status"], "failed")
            self.assertEqual(summary["failure_count"], 2)
            self.assertIn("Lock cleanup failed", stderr.getvalue())

    def test_forced_backup_race_does_not_restore_a_manifest_as_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            outputs = root / "outputs"
            outputs.mkdir()
            catalogue = outputs / "catalogue.csv"
            manifest = outputs / "catalogue.run.json"
            catalogue.write_text("old catalogue", encoding="utf-8")
            manifest.write_text("old manifest", encoding="utf-8")
            real_writer = write_json_atomic

            def catalogue_disappears(paths):
                catalogue.unlink()
                return backup_existing_results(paths)

            def fail_normal_manifest(destination, value, *, replace_existing=True):
                if destination == manifest:
                    raise OSError("manifest disk failure")
                return real_writer(destination, value, replace_existing=replace_existing)

            with (
                patch("uma_bct.batch_runner.backup_existing_results", side_effect=catalogue_disappears),
                patch("uma_bct.batch_runner.write_json_atomic", side_effect=fail_normal_manifest),
            ):
                status, summary, progress = self.run_with(root, FakeBatchPipeline(), [root / "shelf.jpg"], force=True)

            self.assertEqual(status, 1)
            self.assertFalse(catalogue.exists())
            self.assertEqual(manifest.read_text(encoding="utf-8"), "old manifest")
            self.assertTrue(all(not path.casefold().endswith(".csv") for path in summary["previous_results"]))

    def test_forced_backup_race_restores_a_catalogue_that_appears(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            outputs = root / "outputs"
            outputs.mkdir()
            catalogue = outputs / "catalogue.csv"
            manifest = outputs / "catalogue.run.json"
            manifest.write_text("old manifest", encoding="utf-8")
            real_writer = write_json_atomic

            def catalogue_appears(paths):
                catalogue.write_text("raced catalogue", encoding="utf-8")
                return backup_existing_results(paths)

            def fail_normal_manifest(destination, value, *, replace_existing=True):
                if destination == manifest:
                    raise OSError("manifest disk failure")
                return real_writer(destination, value, replace_existing=replace_existing)

            with (
                patch("uma_bct.batch_runner.backup_existing_results", side_effect=catalogue_appears),
                patch("uma_bct.batch_runner.write_json_atomic", side_effect=fail_normal_manifest),
            ):
                status, summary, progress = self.run_with(root, FakeBatchPipeline(), [root / "shelf.jpg"], force=True)

            self.assertEqual(status, 1)
            self.assertEqual(catalogue.read_text(encoding="utf-8"), "raced catalogue")

    def test_catalogue_appearing_after_backup_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            outputs = root / "outputs"
            outputs.mkdir()
            catalogue = outputs / "catalogue.csv"
            manifest = outputs / "catalogue.run.json"
            manifest.write_text("old manifest", encoding="utf-8")

            def appear_after_backup(paths):
                backups = backup_existing_results(paths)
                catalogue.write_text("late catalogue", encoding="utf-8")
                return backups

            with patch("uma_bct.batch_runner.backup_existing_results", side_effect=appear_after_backup):
                status, summary, progress = self.run_with(root, FakeBatchPipeline(), [root / "shelf.jpg"], force=True)

            self.assertEqual(status, 1)
            self.assertEqual(catalogue.read_text(encoding="utf-8"), "late catalogue")
            self.assertIsNone(summary["catalogue_output"])

    def test_manifest_appearing_after_backup_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            outputs = root / "outputs"
            outputs.mkdir()
            catalogue = outputs / "catalogue.csv"
            manifest = outputs / "catalogue.run.json"

            def appear_after_backup(paths):
                backups = backup_existing_results(paths)
                manifest.write_text("late manifest", encoding="utf-8")
                return backups

            with patch("uma_bct.batch_runner.backup_existing_results", side_effect=appear_after_backup):
                status, summary, progress = self.run_with(root, FakeBatchPipeline(), [root / "shelf.jpg"], force=True)

            self.assertEqual(status, 1)
            self.assertEqual(manifest.read_text(encoding="utf-8"), "late manifest")
            self.assertFalse(catalogue.exists())
            self.assertIn(".failed-", summary["run_manifest"])


if __name__ == "__main__":
    unittest.main()
