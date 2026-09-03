import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import uma_bct.cli as cli_module
from uma_bct.cli import build_argument_parser, main, output_lock_path
from uma_bct.domain import (
    BoundingBox,
    Box,
    BoxText,
    BoxTranscription,
    InputImage,
    Label,
)
from uma_bct.pipeline import ImageResult


class FakePipeline:
    def __init__(self, warnings: tuple[str, ...] = ()) -> None:
        self.warnings = warnings

    def process(self, image_path: Path) -> ImageResult:
        input_image = InputImage(image_path, 1000, 500)
        box = Box(
            box_id="box-1",
            source_image=input_image,
            bounding_box=BoundingBox((10, 20), (110, 220)),
            image_file_path=image_path.parent / "crops" / "box-1.png",
            position="top-left",
            detector_confidence=0.9,
        )
        confidence = 92
        transcription = BoxTranscription(
            box_id=box.box_id,
            position=box.position,
            label=Label(
                box.box_id,
                "0001",
                ("1974.0056", "Unit 0001"),
                ("1974.0056",),
            ),
            text_lines=(
                BoxText(1, "1974.0056", "typed", confidence),
                BoxText(2, "Unit 0001", "typed", confidence),
                BoxText(1, "Correspondence", "handwritten", confidence),
            ),
            confidence=confidence,
        )
        return ImageResult(
            input_image=input_image,
            detector="fake-detector",
            transcriber="fake-transcriber",
            boxes=(box,),
            transcriptions=(transcription,),
            warnings=self.warnings,
        )


class PartiallyFailingPipeline(FakePipeline):
    def process(self, image_path: Path) -> ImageResult:
        if image_path.name == "b.jpg":
            raise RuntimeError("simulated image failure")
        return super().process(image_path)


class EmptyPipeline:
    def process(self, image_path: Path) -> ImageResult:
        return ImageResult(
            input_image=InputImage(image_path, 1000, 500),
            detector="qwen",
            transcriber="qwen",
            boxes=(),
            transcriptions=(),
            warnings=("no readable records",),
            status="failed",
        )


class CliExportTest(unittest.TestCase):
    def test_default_paths_support_the_shortest_successful_workflow(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            inputs = root / "inputs"
            inputs.mkdir()
            (inputs / "shelf.jpg").touch()
            stdout = io.StringIO()

            with (
                contextlib.chdir(root),
                patch("uma_bct.cli.build_pipeline", return_value=FakePipeline()),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                exit_code = main([])

            summary = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(summary["catalogue_output"], "outputs/catalogue.csv")
            self.assertTrue((root / "outputs" / "catalogue.csv").is_file())
            self.assertTrue((root / "outputs" / "catalogue.run.json").is_file())
            self.assertEqual(len(list((root / "outputs" / "crops").iterdir())), 1)

    def test_default_output_still_refuses_overwrite_unless_forced(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            inputs = root / "inputs"
            outputs = root / "outputs"
            inputs.mkdir()
            outputs.mkdir()
            (inputs / "shelf.jpg").touch()
            catalogue = outputs / "catalogue.csv"
            catalogue.write_text("keep me", encoding="utf-8")

            with contextlib.chdir(root), self.assertRaises(SystemExit) as raised:
                main([])

            self.assertEqual(raised.exception.code, 2)
            self.assertEqual(catalogue.read_text(encoding="utf-8"), "keep me")

            with (
                contextlib.chdir(root),
                patch("uma_bct.cli.build_pipeline", return_value=FakePipeline()),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                exit_code = main(["--force"])

            self.assertEqual(exit_code, 0)
            self.assertNotEqual(catalogue.read_text(encoding="utf-8"), "keep me")

    def test_missing_default_input_explains_the_short_workflow(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            stderr = io.StringIO()

            with (
                contextlib.chdir(temporary_directory),
                contextlib.redirect_stderr(stderr),
                self.assertRaises(SystemExit) as raised,
            ):
                main([])

            self.assertEqual(raised.exception.code, 2)
            self.assertIn("uma-bct setup", stderr.getvalue())
            self.assertIn("add photos", stderr.getvalue())

    def test_help_shows_setup_and_the_short_workflow_with_advanced_controls(self):
        help_text = build_argument_parser().format_help()

        self.assertIn("uma-bct setup", help_text)
        self.assertIn("uma-bct", help_text)
        self.assertIn("--detector", help_text)
        self.assertIn("--model", help_text)
        self.assertIn("--request-timeout", help_text)

    def test_setup_subcommand_routes_without_processing_inputs(self):
        with patch("uma_bct.cli.run_setup", return_value=0) as setup:
            exit_code = main(["setup"])

        self.assertEqual(exit_code, 0)
        setup.assert_called_once_with()

    def test_setup_rejects_batch_options_instead_of_ignoring_them(self):
        for arguments in (
            ["--force", "setup"],
            ["--model", "another-model", "setup"],
            ["--request-timeout", "1", "setup"],
        ):
            with (
                self.subTest(arguments=arguments),
                patch("uma_bct.cli.run_setup") as setup,
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as raised,
            ):
                main(arguments)

            self.assertEqual(raised.exception.code, 2)
            setup.assert_not_called()

    def test_output_argument_writes_catalogue_and_reports_record_count(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_path = root / "shelf.JPG"
            image_path.touch()
            output_path = root / "exports" / "catalogue.csv"
            stdout = io.StringIO()
            stderr = io.StringIO()

            with (
                patch("uma_bct.cli.build_pipeline", return_value=FakePipeline()),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                exit_code = main(["-i", str(image_path), "-o", str(output_path)])

            summary = json.loads(stdout.getvalue())
            written = pd.read_csv(output_path, keep_default_na=False)
            self.assertEqual(exit_code, 0)
            self.assertEqual(summary["catalogue_output"], str(output_path))
            self.assertEqual(summary["catalogue_records"], 1)
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(written.loc[0, "Node Title"], "1974.0056 Unit 0001")
            self.assertTrue((root / "exports" / "catalogue.run.json").is_file())
            self.assertEqual(json.loads(stdout.getvalue())["schema_version"], 1)
            self.assertNotIn("Correspondence", stdout.getvalue())
            self.assertIn("Processing", stderr.getvalue())

    def test_rejects_output_without_a_supported_extension_before_processing(self):
        with self.assertRaises(SystemExit) as raised:
            main(["-i", "unused.jpg", "-o", "catalogue.json"])

        self.assertEqual(raised.exception.code, 2)

    def test_rejects_redirected_output_and_crop_roots(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_path = root / "shelf.jpg"
            real_output = root / "real-output"
            linked_output = root / "linked-output"
            image_path.touch()
            real_output.mkdir()
            try:
                linked_output.symlink_to(real_output, target_is_directory=True)
            except OSError as error:
                self.skipTest(f"symbolic links are unavailable: {error}")

            cases = (
                ["-i", str(image_path), "-o", str(linked_output / "catalogue.csv")],
                [
                    "-i",
                    str(image_path),
                    "-o",
                    str(root / "catalogue.csv"),
                    "-b",
                    str(linked_output),
                ],
            )
            for arguments in cases:
                with (
                    self.subTest(arguments=arguments),
                    contextlib.redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit) as raised,
                ):
                    main(arguments)

                self.assertEqual(raised.exception.code, 2)

    def test_refuses_to_replace_existing_output_without_force(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_path = root / "shelf.jpg"
            output_path = root / "catalogue.csv"
            image_path.touch()
            output_path.write_text("keep me", encoding="utf-8")

            with self.assertRaises(SystemExit) as raised:
                main(["-i", str(image_path), "-o", str(output_path)])

            self.assertEqual(raised.exception.code, 2)
            self.assertEqual(output_path.read_text(encoding="utf-8"), "keep me")

    def test_refuses_to_replace_existing_manifest_without_force(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_path = root / "shelf.jpg"
            output_path = root / "catalogue.csv"
            manifest = root / "catalogue.run.json"
            image_path.touch()
            manifest.write_text("keep me", encoding="utf-8")

            with self.assertRaises(SystemExit) as raised:
                main(["-i", str(image_path), "-o", str(output_path)])

            self.assertEqual(raised.exception.code, 2)
            self.assertEqual(manifest.read_text(encoding="utf-8"), "keep me")

    def test_preflight_failure_writes_manifest_but_not_empty_catalogue(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_path = root / "shelf.jpg"
            output_path = root / "catalogue.csv"
            image_path.touch()
            stdout = io.StringIO()

            with (
                patch(
                    "uma_bct.cli.build_pipeline", side_effect=RuntimeError("no model")
                ),
                contextlib.redirect_stdout(stdout),
            ):
                exit_code = main(["-i", str(image_path), "-o", str(output_path)])

            summary = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 2)
            self.assertEqual(summary["status"], "failed")
            self.assertFalse(output_path.exists())
            manifest_path = root / "catalogue.run.json"
            self.assertTrue(manifest_path.is_file())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertIn("no model", manifest["failures"][0]["error"])

    def test_invalid_timeout_environment_is_a_json_preflight_failure(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_path = root / "shelf.jpg"
            output_path = root / "catalogue.csv"
            image_path.touch()
            stdout = io.StringIO()

            with (
                patch.dict(
                    "os.environ",
                    {"UMA_BCT_OLLAMA_TIMEOUT_SECONDS": "not-a-number"},
                ),
                patch("uma_bct.cli.build_pipeline") as build_pipeline,
                contextlib.redirect_stdout(stdout),
            ):
                exit_code = main(["-i", str(image_path), "-o", str(output_path)])

            summary = json.loads(stdout.getvalue())
            manifest = json.loads(
                Path(summary["run_manifest"]).read_text(encoding="utf-8")
            )
            self.assertEqual(exit_code, 2)
            self.assertEqual(summary["status"], "failed")
            self.assertIsNone(manifest["configuration"]["request_timeout_seconds"])
            self.assertFalse(output_path.exists())
            build_pipeline.assert_not_called()

    def test_pipeline_warnings_are_written_to_stderr(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_path = root / "shelf.jpg"
            output_path = root / "catalogue.csv"
            image_path.touch()
            stderr = io.StringIO()

            with (
                patch(
                    "uma_bct.cli.build_pipeline",
                    return_value=FakePipeline(("OpenCV fallback used",)),
                ),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(stderr),
            ):
                exit_code = main(["-i", str(image_path), "-o", str(output_path)])

            self.assertEqual(exit_code, 0)
            self.assertIn("Warning for", stderr.getvalue())
            self.assertIn("OpenCV fallback used", stderr.getvalue())

    def test_partial_batch_exports_successful_rows_and_returns_one(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "a.jpg").touch()
            (root / "b.jpg").touch()
            output_path = root / "catalogue.csv"
            stdout = io.StringIO()

            with (
                patch(
                    "uma_bct.cli.build_pipeline",
                    return_value=PartiallyFailingPipeline(),
                ),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                exit_code = main(["-i", str(root), "-o", str(output_path)])

            summary = json.loads(stdout.getvalue())
            written = pd.read_csv(output_path, keep_default_na=False)
            self.assertEqual(exit_code, 1)
            self.assertEqual(summary["status"], "partial")
            self.assertEqual(summary["catalogue_records"], 1)
            self.assertEqual(len(written), 1)
            manifest = json.loads(
                Path(summary["run_manifest"]).read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["failures"][0]["source_image"], str(root / "b.jpg")
            )

    def test_zero_row_forced_run_does_not_replace_existing_catalogue(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_path = root / "shelf.jpg"
            output_path = root / "catalogue.csv"
            image_path.touch()
            output_path.write_text("existing catalogue\n", encoding="utf-8")
            normal_manifest = root / "catalogue.run.json"
            normal_manifest.write_text("existing manifest\n", encoding="utf-8")
            stdout = io.StringIO()

            with (
                patch("uma_bct.cli.build_pipeline", return_value=EmptyPipeline()),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                exit_code = main(
                    ["-i", str(image_path), "-o", str(output_path), "--force"]
                )

            summary = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 1)
            self.assertEqual(summary["status"], "failed")
            self.assertEqual(summary["catalogue_records"], 0)
            self.assertEqual(
                output_path.read_text(encoding="utf-8"), "existing catalogue\n"
            )
            self.assertEqual(
                normal_manifest.read_text(encoding="utf-8"), "existing manifest\n"
            )
            failed_manifest = Path(summary["run_manifest"])
            self.assertNotEqual(failed_manifest, normal_manifest)
            self.assertTrue(failed_manifest.is_file())
            self.assertIn(".failed-", failed_manifest.name)

    def test_manifest_failure_rolls_back_a_forced_catalogue_replacement(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_path = root / "shelf.jpg"
            output_path = root / "catalogue.csv"
            manifest = root / "catalogue.run.json"
            image_path.touch()
            output_path.write_text("existing catalogue\n", encoding="utf-8")
            manifest.write_text("existing manifest\n", encoding="utf-8")
            stdout = io.StringIO()

            with (
                patch("uma_bct.cli.build_pipeline", return_value=FakePipeline()),
                patch(
                    "uma_bct.cli.write_json_atomic",
                    side_effect=OSError("manifest disk full"),
                ),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                exit_code = main(
                    ["-i", str(image_path), "-o", str(output_path), "--force"]
                )

            summary = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 1)
            self.assertEqual(summary["status"], "failed")
            self.assertIsNone(summary["catalogue_output"])
            self.assertEqual(summary["catalogue_records"], 0)
            self.assertIsNone(summary["run_manifest"])
            self.assertEqual(
                output_path.read_text(encoding="utf-8"), "existing catalogue\n"
            )
            self.assertEqual(
                manifest.read_text(encoding="utf-8"), "existing manifest\n"
            )

    def test_manifest_failure_writes_a_separate_recovery_manifest(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_path = root / "shelf.jpg"
            output_path = root / "catalogue.csv"
            normal_manifest = root / "catalogue.run.json"
            image_path.touch()
            output_path.write_text("existing catalogue\n", encoding="utf-8")
            normal_manifest.write_text("existing manifest\n", encoding="utf-8")
            stdout = io.StringIO()
            real_write_json_atomic = cli_module.write_json_atomic
            call_count = 0

            def fail_once(destination, value):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise OSError("manifest disk full")
                return real_write_json_atomic(destination, value)

            with (
                patch("uma_bct.cli.build_pipeline", return_value=FakePipeline()),
                patch("uma_bct.cli.write_json_atomic", side_effect=fail_once),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                exit_code = main(
                    ["-i", str(image_path), "-o", str(output_path), "--force"]
                )

            summary = json.loads(stdout.getvalue())
            recovery_manifest = Path(summary["run_manifest"])
            self.assertEqual(exit_code, 1)
            self.assertEqual(summary["status"], "failed")
            self.assertNotEqual(recovery_manifest, normal_manifest)
            self.assertTrue(recovery_manifest.is_file())
            recovery = json.loads(recovery_manifest.read_text(encoding="utf-8"))
            self.assertEqual(recovery["status"], "failed")
            self.assertIsNone(recovery["catalogue_output"])
            self.assertEqual(recovery["catalogue_records"], 0)
            self.assertEqual(recovery["run_manifest"], str(recovery_manifest))
            self.assertEqual(recovery["failures"][-1]["stage"], "manifest")
            self.assertEqual(
                output_path.read_text(encoding="utf-8"), "existing catalogue\n"
            )
            self.assertEqual(
                normal_manifest.read_text(encoding="utf-8"), "existing manifest\n"
            )

    def test_manifest_failure_removes_new_catalogue_and_writes_recovery_manifest(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_path = root / "shelf.jpg"
            output_path = root / "catalogue.csv"
            image_path.touch()
            stdout = io.StringIO()
            real_write_json_atomic = cli_module.write_json_atomic
            call_count = 0

            def fail_once(destination, value):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise OSError("manifest disk full")
                return real_write_json_atomic(destination, value)

            with (
                patch("uma_bct.cli.build_pipeline", return_value=FakePipeline()),
                patch("uma_bct.cli.write_json_atomic", side_effect=fail_once),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                exit_code = main(["-i", str(image_path), "-o", str(output_path)])

            summary = json.loads(stdout.getvalue())
            recovery_manifest = Path(summary["run_manifest"])
            recovery = json.loads(recovery_manifest.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 1)
            self.assertFalse(output_path.exists())
            self.assertEqual(recovery["status"], "failed")
            self.assertIsNone(recovery["catalogue_output"])
            self.assertEqual(recovery["catalogue_records"], 0)
            self.assertEqual(recovery["run_manifest"], str(recovery_manifest))
            self.assertEqual(recovery["failures"][-1]["stage"], "manifest")

    def test_export_failure_preserves_existing_catalogue_and_records_failure(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_path = root / "shelf.jpg"
            output_path = root / "catalogue.csv"
            image_path.touch()
            output_path.write_bytes(b"existing catalogue\n")
            stdout = io.StringIO()

            with (
                patch("uma_bct.cli.build_pipeline", return_value=FakePipeline()),
                patch(
                    "uma_bct.cli.export_catalogue",
                    side_effect=OSError("export disk full"),
                ),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                exit_code = main(
                    ["-i", str(image_path), "-o", str(output_path), "--force"]
                )

            summary = json.loads(stdout.getvalue())
            manifest = json.loads(
                Path(summary["run_manifest"]).read_text(encoding="utf-8")
            )
            self.assertEqual(exit_code, 1)
            self.assertEqual(output_path.read_bytes(), b"existing catalogue\n")
            self.assertEqual(summary["status"], "failed")
            self.assertIsNone(summary["catalogue_output"])
            self.assertEqual(manifest["failures"][-1]["stage"], "export")
            self.assertIsNone(summary["catalogue_backup"])

    def test_rollback_and_recovery_failure_expose_retained_private_backup(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_path = root / "shelf.jpg"
            output_path = root / "catalogue.csv"
            image_path.touch()
            output_path.write_text("existing catalogue\n", encoding="utf-8")
            stdout = io.StringIO()
            stderr = io.StringIO()

            with (
                patch("uma_bct.cli.build_pipeline", return_value=FakePipeline()),
                patch(
                    "uma_bct.cli.write_json_atomic",
                    side_effect=OSError("manifest disk full"),
                ),
                patch(
                    "uma_bct.cli.restore_catalogue",
                    side_effect=OSError("rollback denied"),
                ),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                exit_code = main(
                    ["-i", str(image_path), "-o", str(output_path), "--force"]
                )

            summary = json.loads(stdout.getvalue())
            retained_backup = Path(summary["catalogue_backup"])
            self.assertEqual(exit_code, 1)
            self.assertEqual(summary["status"], "partial")
            self.assertIsNone(summary["run_manifest"])
            self.assertTrue(retained_backup.is_file())
            self.assertIn(str(retained_backup), stderr.getvalue())

    def test_backup_cleanup_failure_is_partial_and_recorded_for_recovery(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_path = root / "shelf.jpg"
            output_path = root / "catalogue.csv"
            image_path.touch()
            output_path.write_text("existing catalogue\n", encoding="utf-8")
            stdout = io.StringIO()
            stderr = io.StringIO()
            real_unlink = Path.unlink

            def fail_backup_unlink(path, *args, **kwargs):
                if path.name.startswith(".uma-bct-catalogue-backup-"):
                    raise PermissionError("cleanup denied")
                return real_unlink(path, *args, **kwargs)

            with (
                patch("uma_bct.cli.build_pipeline", return_value=FakePipeline()),
                patch("pathlib.Path.unlink", new=fail_backup_unlink),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                exit_code = main(
                    ["-i", str(image_path), "-o", str(output_path), "--force"]
                )

            summary = json.loads(stdout.getvalue())
            retained_backup = Path(summary["catalogue_backup"])
            recovery_manifest = Path(summary["run_manifest"])
            manifest = json.loads(recovery_manifest.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 1)
            self.assertEqual(summary["status"], "partial")
            self.assertTrue(retained_backup.is_file())
            self.assertEqual(manifest["catalogue_backup"], str(retained_backup))
            self.assertIn(str(retained_backup), stderr.getvalue())

    def test_refuses_an_output_reserved_by_another_run(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_path = root / "shelf.jpg"
            output_path = root / "catalogue.csv"
            image_path.touch()
            lock_path = output_lock_path(output_path)
            lock_path.write_text("pid=123\n", encoding="utf-8")

            with self.assertRaises(SystemExit) as raised:
                main(["-i", str(image_path), "-o", str(output_path), "--force"])

            self.assertEqual(raised.exception.code, 2)
            self.assertTrue(lock_path.is_file())

    def test_csv_and_xlsx_with_the_same_stem_share_a_lock(self):
        self.assertEqual(
            output_lock_path(Path("catalogue.csv")),
            output_lock_path(Path("catalogue.xlsx")),
        )

    def test_output_lock_permission_failure_is_a_usage_error(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_path = root / "shelf.jpg"
            image_path.touch()

            with (
                patch("uma_bct.cli.os.open", side_effect=PermissionError("denied")),
                self.assertRaises(SystemExit) as raised,
            ):
                main(["-i", str(image_path), "-o", str(root / "catalogue.csv")])

            self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
