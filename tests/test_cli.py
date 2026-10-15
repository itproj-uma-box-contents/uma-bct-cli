import argparse
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from helpers import make_box, make_image, make_result, make_transcription

from uma_bct.atomic import rename_exclusive, rename_new_file
from uma_bct.batch_runner import export_results_into, run_batch
from uma_bct.cli import discover_images, main, parse_args, validate_output_layout, validate_output_target
from uma_bct.constants import DEFAULT_MIN_DETECTOR_CONFIDENCE, DEFAULT_MIN_TRANSCRIPTION_CONFIDENCE, IMAGE_EXTENSIONS
from uma_bct.domain import BoxResult
from uma_bct.export import export_catalogue
from uma_bct.path_safety import check_path
from uma_bct.run_artifacts import (
    backup_existing_results,
    catalogue_lock_path,
    regular_file_sha256,
    replace_backed_file,
    same_regular_file_content,
    write_json_atomic,
)


@contextlib.contextmanager
def temporary_root():
    with tempfile.TemporaryDirectory() as directory:
        yield Path(directory).resolve()


def manifest_disk_failure(manifest: Path | None = None):
    def fail_manifest_write(destination, value, *, replace_existing=True):
        if destination == manifest or destination.name.startswith(".uma-bct-json-publish-"):
            raise OSError("manifest disk failure")
        return write_json_atomic(destination, value, replace_existing=replace_existing)

    return fail_manifest_write


class FakeBatchPipeline:
    """Exercise batch publication without reading photos or starting Ollama."""

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


def arguments(root: Path, *, force=False, file_type="csv"):
    return argparse.Namespace(
        detector="qwen",
        transcriber="qwen",
        ai_dates=False,
        model="qwen-test",
        min_transcription_confidence=DEFAULT_MIN_TRANSCRIPTION_CONFIDENCE,
        min_detector_confidence=DEFAULT_MIN_DETECTOR_CONFIDENCE,
        request_timeout=10.0,
        path_to_inputs=root / "inputs",
        path_to_outputs=root / ("outputs/catalogue.xlsx" if file_type == "excel" else "outputs/catalogue.csv"),
        path_to_box_images=root / "outputs/crops",
        force=force,
        json=True,
    )


class CliTest(unittest.TestCase):
    def test_parser_keeps_defaults_and_existing_aliases(self):
        defaults = parse_args([])
        aliases = parse_args(["--path-to-output", "other.csv", "--work-dir", "review"])
        self.assertEqual(defaults.path_to_inputs, Path("inputs"))
        self.assertEqual(defaults.path_to_outputs, Path("outputs/catalogue.csv"))
        self.assertFalse(hasattr(defaults, "html_review"))
        self.assertEqual(defaults.min_detector_confidence, 0.6)
        self.assertEqual(defaults.min_transcription_confidence, 50)
        self.assertTrue(defaults.detect_missing_boxes)
        self.assertEqual(aliases.path_to_outputs, Path("other.csv"))
        self.assertEqual(aliases.path_to_box_images, Path("review"))
        self.assertFalse(defaults.json)
        self.assertTrue(parse_args(["--json"]).json)
        self.assertTrue(parse_args(["--human"]).human)

    def test_can_disable_missing_box_detection(self):
        self.assertFalse(parse_args(["--no-missing-box-detection"]).detect_missing_boxes)

    def test_ai_dates_is_a_model_selection_independent_switch(self):
        self.assertFalse(parse_args([]).ai_dates)
        self.assertTrue(parse_args(["--ai-dates"]).ai_dates)
        self.assertTrue(parse_args(["--hosted", "--ai-dates"]).ai_dates)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(["--date-detector"])

    def test_date_detector_option_is_removed(self):
        self.assertFalse(hasattr(parse_args([]), "date_detector"))
        for option in ("--date-detector", "-u"):
            with self.subTest(option=option), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args([option, "regex"])

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
        with temporary_root() as root:
            (root / "b.jpg").write_bytes(b"x")
            (root / "A.JPG").write_bytes(b"x")
            (root / "notes.txt").write_text("ignore", encoding="utf-8")
            self.assertEqual([path.name for path in discover_images(root)], ["A.JPG", "b.jpg"])

    def test_discovers_every_documented_image_extension_case_insensitively(self):
        with temporary_root() as root:
            for number, extension in enumerate(sorted(IMAGE_EXTENSIONS)):
                (root / f"{number:02d}{extension.upper()}").write_bytes(b"x")
            self.assertEqual({path.suffix.casefold() for path in discover_images(root)}, IMAGE_EXTENSIONS)
    def test_rejects_redirected_input(self):
        with temporary_root() as root:
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
        with temporary_root() as root:
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
        with temporary_root() as root:
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
        with temporary_root() as root:
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
        with temporary_root() as root:
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
        with temporary_root() as root:
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
        with temporary_root() as root:
            manifest = root / "catalogue.run.json"
            manifest.write_text('{"owner": "other process"}\n', encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "already exists"):
                write_json_atomic(manifest, {"status": "failed"}, replace_existing=False)
            self.assertEqual(manifest.read_text(encoding="utf-8"), '{"owner": "other process"}\n')

    def test_manifest_bytes_are_portable_across_platforms(self):
        with temporary_root() as root:
            manifest = root / "catalogue.run.json"
            write_json_atomic(manifest, {"status": "complete"})
            self.assertNotIn(b"\r\n", manifest.read_bytes())

    def test_manifest_writer_does_not_replace_a_file_appearing_at_the_publication_syscall(self):
        with temporary_root() as root:
            manifest = root / "catalogue.run.json"

            def late_destination(source, destination):
                destination.write_text('{"owner": "late process"}\n', encoding="utf-8")
                return rename_exclusive(source, destination)

            with (
                patch("uma_bct.atomic.rename_exclusive", side_effect=late_destination),
                self.assertRaisesRegex(FileExistsError, "already exists"),
            ):
                write_json_atomic(manifest, {"status": "failed"}, replace_existing=False)

            self.assertEqual(manifest.read_text(encoding="utf-8"), '{"owner": "late process"}\n')

    def test_parent_reparse_detection_is_not_platform_dependent(self):
        with temporary_root() as root:
            parent = root / "outputs"
            parent.mkdir()
            target = parent / "catalogue.csv"

            def fake_kind(path, metadata):
                return "reparse point" if path == parent else None

            with patch("uma_bct.path_safety.redirected_path_kind", side_effect=fake_kind):
                with self.assertRaisesRegex(ValueError, "reparse point"):
                    check_path(target, "Catalogue output")

    def test_path_check_wraps_reparse_inspection_errors(self):
        with temporary_root() as root:
            target = root / "catalogue.csv"
            with (
                patch("uma_bct.path_safety.redirected_path_kind", side_effect=OSError("inspection failed")),
                self.assertRaisesRegex(ValueError, "Could not inspect Catalogue output"),
            ):
                check_path(target, "Catalogue output")

    def test_routes_setup_without_starting_a_batch(self):
        with patch("uma_bct.cli.run_setup", return_value=0) as setup:
            self.assertEqual(main(["setup"]), 0)
        setup.assert_called_once_with()

    def test_routes_custom_managed_setup_paths(self):
        with patch("uma_bct.cli.run_setup", return_value=0) as setup:
            self.assertEqual(main(["setup", "--path-to-inputs", "Photos", "--results-directory", "Results"]), 0)
        setup.assert_called_once_with(input_directory=Path("Photos"), results_directory=Path("Results"))

    def test_managed_options_conflict_with_direct_outputs_and_force(self):
        for extra in (
            ("-o", "out.csv"), ("-oout.csv",), ("--path-to-outputs=out.csv",),
            ("--work-dir", "crops"), ("-bcrops",), ("--work-dir=crops",), ("--force",),
        ):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["--results-directory", "Results", *extra])

    def test_reprocess_requires_a_results_directory(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["--reprocess"])

    def test_cli_routes_a_managed_batch(self):
        with temporary_root() as root:
            inputs = root / "photos"
            inputs.mkdir()
            (inputs / "shelf.jpg").write_bytes(b"x")
            results = root / "results"
            with patch("uma_bct.cli.run_managed_batch", return_value=0) as batch:
                status = main(["-i", str(inputs), "--results-directory", str(results)])
            self.assertEqual(status, 0)
            batch.assert_called_once_with(batch.call_args.args[0], [inputs / "shelf.jpg"], results)

    def test_direct_cli_interrupt_emits_clean_json_without_a_traceback(self):
        with temporary_root() as root:
            image = root / "shelf.jpg"
            image.write_bytes(b"synthetic input")
            for interruption, expected in ((KeyboardInterrupt(), 130), (SystemExit(143), 143)):
                stdout, stderr = io.StringIO(), io.StringIO()
                with patch("uma_bct.cli.run_batch", side_effect=interruption), \
                     contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    status = main(["-i", str(image), "-o", str(root / "catalogue.csv"), "--json"])
                self.assertEqual(status, expected)
                self.assertEqual(json.loads(stdout.getvalue())["status"], "interrupted")
                self.assertNotIn("Traceback", stderr.getvalue())

    def test_selects_excel_and_csv_and_rejects_conflicting_options(self):
        for options, expected in (
            ([], "outputs/catalogue.csv"),
            (["--file-type", "EXCEL"], "outputs/catalogue.xlsx"),
            (["--file-type", "CSV"], "outputs/catalogue.csv"),
            (["-o", "custom.csv"], "custom.csv"),
            (["-o", "custom.XLSX"], "custom.XLSX"),
        ):
            with self.subTest(options=options):
                self.assertEqual(parse_args(options).path_to_outputs, Path(expected))
        for options in (
            ["--html-review"],
            ["-o", "catalogue.pdf"],
            ["-o", "catalogue.xlsx", "--file-type", "csv"],
            ["-o", "catalogue.csv", "--file-type", "excel"],
        ):
            with self.subTest(options=options), patch("uma_bct.cli.run_batch") as run:
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                    main(options)
                self.assertEqual(raised.exception.code, 2)
                run.assert_not_called()

    def test_default_run_shows_a_readable_summary(self):
        with temporary_root() as root:
            inputs = root / "inputs"
            inputs.mkdir()
            (inputs / "shelf.jpg").write_bytes(b"fixture")
            stdout, stderr = io.StringIO(), io.StringIO()
            with (
                contextlib.chdir(root),
                patch("uma_bct.cli.keep_awake", return_value=contextlib.nullcontext()),
                patch("uma_bct.batch_runner.build_pipeline", return_value=FakeBatchPipeline()),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                status = main([])
            self.assertEqual(status, 0)
            self.assertIn("Finished\nPhotos finished: 1 of 1", stdout.getvalue())
            self.assertIn("Catalogue: 1 rows", stdout.getvalue())
            self.assertNotIn('"schema_version"', stdout.getvalue())

            json_stdout = io.StringIO()
            with (
                contextlib.chdir(root),
                patch("uma_bct.cli.keep_awake", return_value=contextlib.nullcontext()),
                patch("uma_bct.batch_runner.build_pipeline", return_value=FakeBatchPipeline()),
                contextlib.redirect_stdout(json_stdout),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                json_status = main(["--json", "--force"])
            self.assertEqual(json_status, 0)
            self.assertEqual(json.loads(json_stdout.getvalue())["catalogue_records"], 1)

    def test_cli_routes_a_valid_batch(self):
        with temporary_root() as root:
            inputs = root / "inputs"
            inputs.mkdir()
            (inputs / "shelf.jpg").write_bytes(b"x")
            output = root / "outputs/catalogue.csv"
            with patch("uma_bct.cli.run_batch", return_value=0) as batch:
                status = main(["-i", str(inputs), "-o", str(output), "--work-dir", str(root / "outputs/crops")])
            self.assertEqual(status, 0)
            self.assertEqual(batch.call_args.args[1], [inputs / "shelf.jpg"])

    def test_cli_rejects_catalogue_inside_review_crop_directory(self):
        with temporary_root() as root:
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
        with temporary_root() as root:
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
    def run_batch_and_capture_output(self, root, pipeline, images, *, force=False, preserve=False, file_type="csv"):
        """Keep JSON stdout separate from progress and warnings on stderr."""

        args = arguments(root, force=force, file_type=file_type)
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

    def prepare_outputs(
        self, root: Path, *, catalogue_content: str | None = "old catalogue",
        manifest_content: str | None = "old manifest"
    ) -> tuple[Path, Path, Path]:
        outputs = root / "outputs"
        outputs.mkdir()
        catalogue = outputs / "catalogue.csv"
        manifest = outputs / "catalogue.run.json"
        if catalogue_content is not None:
            catalogue.write_text(catalogue_content, encoding="utf-8")
        if manifest_content is not None:
            manifest.write_text(manifest_content, encoding="utf-8")
        return outputs, catalogue, manifest

    def assert_run_interrupted(self, root: Path, catalogue: Path, manifest: Path, *patches) -> None:
        args = arguments(root, force=True)
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch("uma_bct.batch_runner.build_pipeline", return_value=FakeBatchPipeline()))
            for fault in patches:
                stack.enter_context(fault)
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            stack.enter_context(self.assertRaises(KeyboardInterrupt))
            run_batch(
                args,
                [root / "shelf.jpg"],
                catalogue,
                manifest,
                preserve_existing_manifest_on_failure=True,
            )

    def assert_existing_output_edit_is_preserved(self, output_kind: str, *, during_publication: bool) -> None:
        with temporary_root() as root:
            _outputs, catalogue, manifest = self.prepare_outputs(root)
            target = catalogue if output_kind == "Catalogue output" else manifest
            edit = f"late {output_kind.casefold()} edit"

            # Place a human edit at either side of the backup boundary. Both
            # races must leave that edit intact instead of publishing over it.
            if during_publication:
                raced = False

                def compare_and_edit_output(first, second, purpose):
                    nonlocal raced
                    matches = same_regular_file_content(first, second, purpose)
                    if purpose == f"{output_kind} publication source" and not raced:
                        target.write_text(edit, encoding="utf-8")
                        raced = True
                    return matches

                fault = patch("uma_bct.run_artifacts.same_regular_file_content", side_effect=compare_and_edit_output)
            else:
                def back_up_and_edit_output(paths):
                    backups = backup_existing_results(paths)
                    target.write_text(edit, encoding="utf-8")
                    return backups

                fault = patch("uma_bct.batch_runner.backup_existing_results", side_effect=back_up_and_edit_output)

            with fault:
                status, summary, _progress = self.run_batch_and_capture_output(
                    root, FakeBatchPipeline(), [root / "shelf.jpg"], force=True, preserve=True
                )
            self.assertEqual(status, 1)
            self.assertEqual(target.read_text(encoding="utf-8"), edit)
            if target == catalogue:
                self.assertIsNone(summary["catalogue_output"])
            else:
                self.assertEqual(catalogue.read_text(encoding="utf-8"), "old catalogue")
                self.assertIn(".failed-", summary["run_manifest"])

    def test_writes_csv_manifest_and_clean_json_summary(self):
        with temporary_root() as root:
            image = root / "inputs/shelf.jpg"
            status, summary, progress = self.run_batch_and_capture_output(root, FakeBatchPipeline(), [image])
            self.assertEqual(status, 0)
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["catalogue_records"], 1)
            self.assertTrue((root / "outputs/catalogue.csv").is_file())
            manifest = json.loads((root / "outputs/catalogue.run.json").read_text(encoding="utf-8"))
            self.assertNotIn("started_at", manifest)
            self.assertNotIn("attempts", str(manifest))
            self.assertNotIn("recovery_path", manifest)
            self.assertNotIn("recovery_path", summary)
            self.assertEqual(manifest["configuration"]["minimum_transcription_confidence"], DEFAULT_MIN_TRANSCRIPTION_CONFIDENCE)
            self.assertEqual(manifest["configuration"]["minimum_detector_confidence"], DEFAULT_MIN_DETECTOR_CONFIDENCE)
            self.assertIn("Processing", progress)

            # The opt-in report uses the same published result without changing JSON mode.
            human_args = arguments(root, force=True)
            human_args.json = False
            human_stdout = io.StringIO()
            with (
                patch("uma_bct.batch_runner.build_pipeline", return_value=FakeBatchPipeline()),
                contextlib.redirect_stdout(human_stdout),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                human_status = run_batch(
                    human_args, [image], human_args.path_to_outputs,
                    root / "outputs/catalogue.run.json", preserve_existing_manifest_on_failure=True,
                )
            self.assertEqual(human_status, 0)
            self.assertIn("Finished\nPhotos finished: 1 of 1", human_stdout.getvalue())
            self.assertIn("Catalogue: 1 rows", human_stdout.getvalue())
            self.assertNotIn('"schema_version"', human_stdout.getvalue())

    def test_continues_after_one_image_failure(self):
        with temporary_root() as root:
            images = [root / "good.jpg", root / "bad.jpg"]
            status, summary, progress = self.run_batch_and_capture_output(root, FakeBatchPipeline(failing_name="bad.jpg"), images)
            self.assertEqual(status, 1)
            self.assertEqual(summary["status"], "partial")
            self.assertEqual(summary["catalogue_records"], 1)
            self.assertEqual(summary["failure_count"], 1)

    def test_empty_forced_run_preserves_previous_results(self):
        with temporary_root() as root:
            _outputs, catalogue, manifest = self.prepare_outputs(root)
            status, summary, progress = self.run_batch_and_capture_output(
                root, FakeBatchPipeline(empty=True), [root / "empty.jpg"], force=True, preserve=True
            )
            self.assertEqual(status, 1)
            self.assertEqual(catalogue.read_text(encoding="utf-8"), "old catalogue")
            self.assertEqual(manifest.read_text(encoding="utf-8"), "old manifest")
            self.assertIn(".failed-", summary["run_manifest"])
            self.assertEqual(summary["failure_count"], 1)

    def test_forced_success_keeps_timestamped_previous_results(self):
        with temporary_root() as root:
            self.prepare_outputs(root)
            status, summary, progress = self.run_batch_and_capture_output(root, FakeBatchPipeline(), [root / "shelf.jpg"], force=True)
            self.assertEqual(status, 0)
            self.assertEqual(len(summary["previous_results"]), 2)
            for path in summary["previous_results"]:
                self.assertTrue(Path(path).is_file())

    def test_preflight_failure_is_reported_once_and_releases_the_lock(self):
        with temporary_root() as root:
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
        with temporary_root() as root:
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

    def test_manifest_failure_preserves_legacy_html_and_restores_csv(self):
        with temporary_root() as root:
            outputs, catalogue, manifest = self.prepare_outputs(root)
            review = outputs / "catalogue.review.html"
            review.write_text("previous review", encoding="utf-8")
            with patch("uma_bct.batch_runner.write_json_atomic", side_effect=manifest_disk_failure(manifest)):
                status, summary, _ = self.run_batch_and_capture_output(root, FakeBatchPipeline(), [root / "shelf.jpg"],
                                                                      force=True)
            self.assertEqual(status, 1)
            self.assertEqual(catalogue.read_text(), "old catalogue")
            self.assertEqual(review.read_text(), "previous review")
            self.assertNotIn("review_output", summary)

    def test_forced_csv_run_preserves_existing_legacy_html(self):
        with temporary_root() as root:
            outputs, catalogue, _ = self.prepare_outputs(root)
            review = outputs / "catalogue.review.html"
            review.write_text("reviewer snapshot", encoding="utf-8")
            status, summary, stderr = self.run_batch_and_capture_output(
                root, FakeBatchPipeline(), [root / "shelf.jpg"], force=True,
            )
            self.assertEqual(status, 0, stderr)
            self.assertTrue(catalogue.exists())
            self.assertNotIn("review_output", summary)
            self.assertEqual(review.read_text(encoding="utf-8"), "reviewer snapshot")


    def test_manifest_failure_preserves_edited_csv_and_legacy_html(self):
        with temporary_root() as root:
            outputs, catalogue, manifest = self.prepare_outputs(root)
            review = catalogue.with_suffix(".review.html")
            review.write_text("previous review", encoding="utf-8")

            def edit_csv_and_fail(destination, value, *, replace_existing=True):
                if destination.name.startswith(".uma-bct-json-publish-"):
                    catalogue.write_text("late CSV edit", encoding="utf-8")
                    raise OSError("manifest disk failure")
                return write_json_atomic(destination, value, replace_existing=replace_existing)

            with patch("uma_bct.batch_runner.write_json_atomic", side_effect=edit_csv_and_fail):
                status, summary, _ = self.run_batch_and_capture_output(
                    root, FakeBatchPipeline(), [root / "shelf.jpg"], force=True, preserve=True,
                )
            self.assertEqual(status, 1)
            self.assertEqual(summary["status"], "partial")
            self.assertEqual(catalogue.read_text(), "late CSV edit")
            self.assertEqual(review.read_text(), "previous review")
            self.assertEqual(manifest.read_text(), "old manifest")
            self.assertNotIn("review_output", summary)
            recovery = Path(summary["recovery_path"])
            self.assertEqual(recovery.suffix, ".csv")
            self.assertEqual(recovery.read_text(), "old catalogue")
            self.assertEqual(summary["recovery_paths"], [str(recovery)])
            saved = json.loads(Path(summary["run_manifest"]).read_text())
            self.assertNotIn("review_output", saved)
            self.assertEqual(list(outputs.glob(".uma-bct-*")), [])


    def test_manifest_failure_restores_old_csv_and_writes_a_failed_run_manifest(self):
        with temporary_root() as root:
            _outputs, catalogue, manifest = self.prepare_outputs(root)
            with patch("uma_bct.batch_runner.write_json_atomic", side_effect=manifest_disk_failure(manifest)):
                status, summary, progress = self.run_batch_and_capture_output(root, FakeBatchPipeline(), [root / "shelf.jpg"], force=True)

            self.assertEqual(status, 1)
            self.assertEqual(summary["status"], "failed")
            self.assertIsNone(summary["catalogue_output"])
            self.assertEqual(catalogue.read_text(encoding="utf-8"), "old catalogue")
            self.assertEqual(manifest.read_text(encoding="utf-8"), "old manifest")
            self.assertIn(".failed-", summary["run_manifest"])
            self.assertTrue(Path(summary["run_manifest"]).is_file())
            self.assertEqual(len(summary["previous_results"]), 2)
            self.assertIn("Manifest write failed", progress)

    def test_excel_forced_publication_and_manifest_failure_restore_the_workbook(self):
        with temporary_root() as root:
            status, first, stderr = self.run_batch_and_capture_output(
                root, FakeBatchPipeline(), [root / "shelf.jpg"], file_type="excel",
            )
            self.assertEqual(status, 0, stderr)
            catalogue = Path(first["catalogue_output"])
            manifest = Path(first["run_manifest"])
            previous = {path: path.read_bytes() for path in (catalogue, manifest)}
            with patch("uma_bct.batch_runner.write_json_atomic", side_effect=manifest_disk_failure(manifest)):
                status, failed, _ = self.run_batch_and_capture_output(
                    root, FakeBatchPipeline(), [root / "shelf.jpg"], file_type="excel", force=True, preserve=True,
                )
            self.assertEqual(status, 1)
            self.assertIsNone(failed["catalogue_output"])
            for path, content in previous.items():
                self.assertEqual(path.read_bytes(), content)
            status, replaced, stderr = self.run_batch_and_capture_output(
                root, FakeBatchPipeline(), [root / "shelf.jpg"], file_type="excel", force=True,
            )
            self.assertEqual(status, 0, stderr)
            self.assertEqual(replaced["catalogue_output"], str(catalogue))
            self.assertTrue(any(Path(path).suffix == ".xlsx" for path in replaced["previous_results"]))

    def test_manifest_interruption_restores_the_previous_csv(self):
        with temporary_root() as root:
            _outputs, catalogue, manifest = self.prepare_outputs(root)
            self.assert_run_interrupted(
                root, catalogue, manifest,
                patch("uma_bct.batch_runner.write_json_atomic", side_effect=KeyboardInterrupt),
            )

            self.assertEqual(catalogue.read_text(encoding="utf-8"), "old catalogue")
            self.assertEqual(manifest.read_text(encoding="utf-8"), "old manifest")
            self.assertFalse(catalogue_lock_path(catalogue).exists())

    def test_interruption_after_csv_staging_does_not_cross_the_manifest_boundary(self):
        with temporary_root() as root:
            outputs, catalogue, manifest = self.prepare_outputs(root)

            def interrupt_after_staging(*call_args, **call_kwargs):
                export_catalogue(*call_args, **call_kwargs)
                raise KeyboardInterrupt

            self.assert_run_interrupted(
                root, catalogue, manifest,
                patch("uma_bct.batch_runner.export_catalogue", side_effect=interrupt_after_staging),
            )

            self.assertEqual(catalogue.read_text(encoding="utf-8"), "old catalogue")
            self.assertEqual(manifest.read_text(encoding="utf-8"), "old manifest")
            self.assertEqual(tuple(outputs.glob(".uma-bct-catalogue-*")), ())

    def test_interruption_after_export_return_uses_the_callers_transaction_state(self):
        with temporary_root() as root:
            _outputs, catalogue, manifest = self.prepare_outputs(root)
            def interrupt_after_return(*call_args, **call_kwargs):
                export_results_into(*call_args, **call_kwargs)
                raise KeyboardInterrupt

            self.assert_run_interrupted(
                root, catalogue, manifest,
                patch("uma_bct.batch_runner.export_results_into", side_effect=interrupt_after_return),
            )

            self.assertEqual(catalogue.read_text(encoding="utf-8"), "old catalogue")
            self.assertEqual(manifest.read_text(encoding="utf-8"), "old manifest")

    def test_interruption_after_manifest_rename_restores_both_previous_files(self):
        with temporary_root() as root:
            _outputs, catalogue, manifest = self.prepare_outputs(root)
            def interrupt_after_manifest(staging, destination, backup, purpose):
                replace_backed_file(staging, destination, backup, purpose)
                if purpose == "Run manifest":
                    raise KeyboardInterrupt

            self.assert_run_interrupted(
                root, catalogue, manifest,
                patch("uma_bct.batch_runner.replace_backed_file", side_effect=interrupt_after_manifest),
            )

            self.assertEqual(catalogue.read_text(encoding="utf-8"), "old catalogue")
            self.assertEqual(manifest.read_text(encoding="utf-8"), "old manifest")

    def test_interruption_after_backed_file_quarantine_restores_both_previous_files(self):
        for interrupted_name in ("catalogue.csv", "catalogue.run.json"):
            with self.subTest(interrupted_name=interrupted_name), temporary_root() as root:
                outputs, catalogue, manifest = self.prepare_outputs(root)
                interrupted_path = outputs / interrupted_name
                real_replace = os.replace
                interrupted = False

                def interrupt_after_quarantine(
                    source, destination, real_replace=real_replace, interrupted_path=interrupted_path
                ):
                    nonlocal interrupted
                    result = real_replace(source, destination)
                    if Path(source) == interrupted_path and not interrupted:
                        interrupted = True
                        raise KeyboardInterrupt
                    return result

                self.assert_run_interrupted(
                    root, catalogue, manifest,
                    patch("uma_bct.run_artifacts.os.replace", side_effect=interrupt_after_quarantine),
                )

                self.assertTrue(interrupted)
                self.assertEqual(catalogue.read_text(encoding="utf-8"), "old catalogue")
                self.assertEqual(manifest.read_text(encoding="utf-8"), "old manifest")
                self.assertEqual(tuple(outputs.glob(".uma-bct-rollback-*")), ())

    def test_manifest_failure_removes_a_new_csv(self):
        with temporary_root() as root:
            manifest = root / "outputs/catalogue.run.json"
            with patch("uma_bct.batch_runner.write_json_atomic", side_effect=manifest_disk_failure(manifest)):
                status, summary, progress = self.run_batch_and_capture_output(root, FakeBatchPipeline(), [root / "shelf.jpg"])

            self.assertEqual(status, 1)
            self.assertEqual(summary["status"], "failed")
            self.assertFalse((root / "outputs/catalogue.csv").exists())
            self.assertTrue(Path(summary["run_manifest"]).is_file())

    def test_rollback_failure_reports_partial_and_retains_the_recovery_path(self):
        with temporary_root() as root:
            _outputs, catalogue, manifest = self.prepare_outputs(root)
            with (
                patch("uma_bct.batch_runner.write_json_atomic", side_effect=manifest_disk_failure(manifest)),
                patch("uma_bct.batch_runner.rollback_owned_file", side_effect=OSError("rollback disk failure")),
            ):
                status, summary, progress = self.run_batch_and_capture_output(root, FakeBatchPipeline(), [root / "shelf.jpg"], force=True)

            self.assertEqual(status, 1)
            self.assertEqual(summary["status"], "partial")
            self.assertTrue(Path(summary["recovery_path"]).is_file())
            self.assertIn(summary["recovery_path"], summary["previous_results"])
            self.assertIn("Rollback failed", progress)

    def test_new_csv_rollback_failure_reports_the_csv_as_the_recovery_path(self):
        with temporary_root() as root:
            manifest = root / "outputs/catalogue.run.json"
            with (
                patch("uma_bct.batch_runner.write_json_atomic", side_effect=manifest_disk_failure(manifest)),
                patch("uma_bct.batch_runner.rollback_owned_file", side_effect=OSError("rollback disk failure")),
            ):
                status, summary, progress = self.run_batch_and_capture_output(root, FakeBatchPipeline(), [root / "shelf.jpg"])

            self.assertEqual(status, 1)
            self.assertEqual(summary["status"], "partial")
            self.assertEqual(summary["recovery_path"], str(root / "outputs/catalogue.csv"))
            self.assertTrue(Path(summary["recovery_path"]).is_file())

    def test_recovery_manifest_failure_keeps_stdout_valid_json(self):
        with temporary_root() as root:
            _outputs, catalogue, _manifest = self.prepare_outputs(root, manifest_content=None)

            with patch("uma_bct.batch_runner.write_json_atomic", side_effect=OSError("manifest disk failure")):
                status, summary, progress = self.run_batch_and_capture_output(root, FakeBatchPipeline(), [root / "shelf.jpg"], force=True)

            self.assertEqual(status, 1)
            self.assertEqual(summary["status"], "failed")
            self.assertIsNone(summary["run_manifest"])
            self.assertGreaterEqual(summary["failure_count"], 2)
            self.assertEqual(catalogue.read_text(encoding="utf-8"), "old catalogue")

    def test_lock_cleanup_failure_keeps_json_and_returns_lock_error_status(self):
        with temporary_root() as root:
            with patch("uma_bct.batch_runner.CatalogueLock.release", side_effect=OSError("cleanup denied")):
                status, summary, progress = self.run_batch_and_capture_output(root, FakeBatchPipeline(), [root / "shelf.jpg"])

            self.assertEqual(status, 2)
            self.assertEqual(summary["status"], "partial")
            self.assertEqual(summary["failure_count"], 1)
            self.assertIn("Lock cleanup failed", progress)
            self.assertEqual(summary["run_manifest"], str(root / "outputs/catalogue.run.json"))
            manifest = json.loads(Path(summary["run_manifest"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], summary["status"])
            self.assertEqual(manifest["failures"], [{"stage": "lock_cleanup", "error": "cleanup denied"}])
            self.assertEqual(list((root / "outputs").glob("*.run.json")), [root / "outputs/catalogue.run.json"])

    def test_lock_cleanup_failure_does_not_overwrite_a_late_manifest_edit(self):
        with temporary_root() as root:
            manifest = root / "outputs/catalogue.run.json"

            def edit_manifest_then_fail(_lock):
                manifest.write_text("late unbacked manifest edit", encoding="utf-8")
                raise OSError("cleanup denied")

            with patch("uma_bct.batch_runner.CatalogueLock.release", autospec=True, side_effect=edit_manifest_then_fail):
                status, summary, progress = self.run_batch_and_capture_output(root, FakeBatchPipeline(), [root / "shelf.jpg"])

            self.assertEqual(status, 2)
            self.assertIsNone(summary["run_manifest"])
            self.assertEqual(summary["failure_count"], 2)
            self.assertEqual(manifest.read_text(encoding="utf-8"), "late unbacked manifest edit")
            self.assertIn("was not updated", progress)

    def test_preflight_and_lock_cleanup_failures_still_emit_one_json_document(self):
        with temporary_root() as root:
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
        with temporary_root() as root:
            _outputs, catalogue, manifest = self.prepare_outputs(root)

            def catalogue_disappears(paths):
                catalogue.unlink()
                return backup_existing_results(paths)

            with (
                patch("uma_bct.batch_runner.backup_existing_results", side_effect=catalogue_disappears),
                patch("uma_bct.batch_runner.write_json_atomic", side_effect=manifest_disk_failure(manifest)),
            ):
                status, summary, progress = self.run_batch_and_capture_output(root, FakeBatchPipeline(), [root / "shelf.jpg"], force=True)

            self.assertEqual(status, 1)
            self.assertFalse(catalogue.exists())
            self.assertEqual(manifest.read_text(encoding="utf-8"), "old manifest")
            self.assertTrue(all(not path.casefold().endswith(".csv") for path in summary["previous_results"]))

    def test_forced_backup_race_restores_a_catalogue_that_appears(self):
        with temporary_root() as root:
            _outputs, catalogue, manifest = self.prepare_outputs(root, catalogue_content=None)

            def catalogue_appears(paths):
                catalogue.write_text("raced catalogue", encoding="utf-8")
                return backup_existing_results(paths)

            with (
                patch("uma_bct.batch_runner.backup_existing_results", side_effect=catalogue_appears),
                patch("uma_bct.batch_runner.write_json_atomic", side_effect=manifest_disk_failure(manifest)),
            ):
                status, summary, progress = self.run_batch_and_capture_output(root, FakeBatchPipeline(), [root / "shelf.jpg"], force=True)

            self.assertEqual(status, 1)
            self.assertEqual(catalogue.read_text(encoding="utf-8"), "raced catalogue")

    def test_catalogue_appearing_after_backup_is_not_overwritten(self):
        with temporary_root() as root:
            _outputs, catalogue, _manifest = self.prepare_outputs(root, catalogue_content=None)

            def appear_after_backup(paths):
                backups = backup_existing_results(paths)
                catalogue.write_text("late catalogue", encoding="utf-8")
                return backups

            with patch("uma_bct.batch_runner.backup_existing_results", side_effect=appear_after_backup):
                status, summary, progress = self.run_batch_and_capture_output(root, FakeBatchPipeline(), [root / "shelf.jpg"], force=True)

            self.assertEqual(status, 1)
            self.assertEqual(catalogue.read_text(encoding="utf-8"), "late catalogue")
            self.assertIsNone(summary["catalogue_output"])

    def test_existing_catalogue_changed_after_backup_is_not_overwritten(self):
        self.assert_existing_output_edit_is_preserved("Catalogue output", during_publication=False)

    def test_interrupt_after_restoring_a_late_catalogue_edit_does_not_roll_it_back(self):
        with temporary_root() as root:
            _outputs, catalogue, manifest = self.prepare_outputs(root)

            def edit_after_backup(paths):
                backups = backup_existing_results(paths)
                catalogue.write_text("late unbacked edit", encoding="utf-8")
                return backups

            def interrupt_after_restore(source, destination, *, existing_message):
                result = rename_new_file(source, destination, existing_message=existing_message)
                if Path(source).name.startswith(".uma-bct-rollback-"):
                    raise KeyboardInterrupt
                return result

            self.assert_run_interrupted(
                root, catalogue, manifest,
                patch("uma_bct.batch_runner.backup_existing_results", side_effect=edit_after_backup),
                patch("uma_bct.run_artifacts.rename_new_file", side_effect=interrupt_after_restore),
            )

            self.assertEqual(catalogue.read_text(encoding="utf-8"), "late unbacked edit")
            self.assertEqual(manifest.read_text(encoding="utf-8"), "old manifest")

    def test_existing_catalogue_changed_at_publication_syscall_is_not_overwritten(self):
        self.assert_existing_output_edit_is_preserved("Catalogue output", during_publication=True)

    def test_existing_manifest_changed_after_backup_is_not_overwritten(self):
        self.assert_existing_output_edit_is_preserved("Run manifest", during_publication=False)

    def test_existing_manifest_changed_at_publication_syscall_is_not_overwritten(self):
        self.assert_existing_output_edit_is_preserved("Run manifest", during_publication=True)

    def test_catalogue_changed_at_rollback_syscall_is_not_overwritten(self):
        with temporary_root() as root:
            _outputs, catalogue, _manifest = self.prepare_outputs(root)
            raced = False

            def edit_after_displaced_digest(path, purpose):
                nonlocal raced
                digest = regular_file_sha256(path, purpose)
                if purpose == "Published catalogue" and not raced:
                    catalogue.write_text("late rollback edit", encoding="utf-8")
                    raced = True
                return digest

            with (
                patch("uma_bct.batch_runner.write_json_atomic", side_effect=manifest_disk_failure()),
                patch("uma_bct.run_artifacts.regular_file_sha256", side_effect=edit_after_displaced_digest),
            ):
                status, summary, _progress = self.run_batch_and_capture_output(
                    root, FakeBatchPipeline(), [root / "shelf.jpg"], force=True, preserve=True
                )

            self.assertTrue(raced)
            self.assertEqual(status, 1)
            self.assertEqual(catalogue.read_text(encoding="utf-8"), "late rollback edit")
            self.assertEqual(summary["status"], "partial")

    def test_manifest_appearing_after_backup_is_not_overwritten(self):
        with temporary_root() as root:
            _outputs, catalogue, manifest = self.prepare_outputs(
                root, catalogue_content=None, manifest_content=None
            )

            def appear_after_backup(paths):
                backups = backup_existing_results(paths)
                manifest.write_text("late manifest", encoding="utf-8")
                return backups

            with patch("uma_bct.batch_runner.backup_existing_results", side_effect=appear_after_backup):
                status, summary, progress = self.run_batch_and_capture_output(root, FakeBatchPipeline(), [root / "shelf.jpg"], force=True)

            self.assertEqual(status, 1)
            self.assertEqual(manifest.read_text(encoding="utf-8"), "late manifest")
            self.assertFalse(catalogue.exists())
            self.assertIn(".failed-", summary["run_manifest"])


if __name__ == "__main__":
    unittest.main()
