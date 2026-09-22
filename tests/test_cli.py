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


# Create an isolated temporary root.
@contextlib.contextmanager
def temporary_root():
    with tempfile.TemporaryDirectory() as directory:
        yield Path(directory).resolve()


# Handle the manifest disk failure step.
def manifest_disk_failure(manifest: Path | None = None):
    # Raise the simulated failure.
    def fail(destination, value, *, replace_existing=True):
        if destination == manifest or destination.name.startswith(".uma-bct-json-publish-"):
            raise OSError("manifest disk failure")
        return write_json_atomic(destination, value, replace_existing=replace_existing)

    return fail


# Provide a controllable batch pipeline test double.
class FakeBatchPipeline:
    # Store the state this object needs.
    def __init__(self, failing_name=None, empty=False):
        self.failing_name = failing_name
        self.empty = empty
        self.work_directory = Path("crops")
        self.model_metadata = {"model": "qwen-test", "model_digest": "sha256:test", "ollama_host": "http://127.0.0.1:11434"}

    # Process the current input.
    def process(self, path):
        if path.name == self.failing_name:
            raise RuntimeError("model failed")
        if self.empty:
            return make_result(status="failed")
        image = make_image(path)
        box = make_box(image=image, box_id=f"{path.stem}:001", crop_path=self.work_directory / path.stem / "box_001.png")
        return make_result(BoxResult(box, make_transcription(box)))


# Build the command arguments.
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


# Group the CLI checks.
class CliTest(unittest.TestCase):
    # Check that parser keeps defaults and existing aliases.
    def test_parser_keeps_defaults_and_existing_aliases(self):
        defaults = parse_args([])
        aliases = parse_args(["--path-to-output", "other.csv", "--work-dir", "review"])
        self.assertEqual(defaults.path_to_inputs, Path("inputs"))
        self.assertEqual(defaults.path_to_outputs, Path("outputs/catalogue.csv"))
        self.assertEqual(defaults.min_detector_confidence, 0.6)
        self.assertEqual(defaults.min_transcription_confidence, 80)
        self.assertEqual(aliases.path_to_outputs, Path("other.csv"))
        self.assertEqual(aliases.path_to_box_images, Path("review"))

    # Check that confidence options keep their distinct meanings.
    def test_confidence_options_keep_their_distinct_meanings(self):
        legacy_detector = parse_args(["--min-confidence", "0.7"])
        explicit_detector = parse_args(["--min-detector-confidence", "0.8"])
        short_transcription = parse_args(["-c", "70"])
        explicit_transcription = parse_args(["--min-transcription-confidence", "75"])

        self.assertEqual(legacy_detector.min_detector_confidence, 0.7)
        self.assertEqual(explicit_detector.min_detector_confidence, 0.8)
        self.assertEqual(short_transcription.min_transcription_confidence, 70)
        self.assertEqual(explicit_transcription.min_transcription_confidence, 75)

    # Check that the code discovers supported images in stable order.
    def test_discovers_supported_images_in_stable_order(self):
        with temporary_root() as root:
            (root / "b.jpg").write_bytes(b"x")
            (root / "A.JPG").write_bytes(b"x")
            (root / "notes.txt").write_text("ignore", encoding="utf-8")
            self.assertEqual([path.name for path in discover_images(root)], ["A.JPG", "b.jpg"])

    # Check that the code rejects redirected input.
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

    # Check that the code rejects an input beneath a redirected parent.
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

    # Check that the code rejects a redirected component before dot dot.
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

    # Check that the code rejects a redirected parent of a new output.
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

    # Check that the code rejects a redirected parent of a new crop root.
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

    # Check that manifest writer rechecks a parent swapped after validation.
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

    # Check that manifest writer does not replace a new unexpected file.
    def test_manifest_writer_does_not_replace_a_new_unexpected_file(self):
        with temporary_root() as root:
            manifest = root / "catalogue.run.json"
            manifest.write_text('{"owner": "other process"}\n', encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "already exists"):
                write_json_atomic(manifest, {"status": "failed"}, replace_existing=False)
            self.assertEqual(manifest.read_text(encoding="utf-8"), '{"owner": "other process"}\n')

    # Check that manifest bytes are portable across platforms.
    def test_manifest_bytes_are_portable_across_platforms(self):
        with temporary_root() as root:
            manifest = root / "catalogue.run.json"
            write_json_atomic(manifest, {"status": "complete"})
            self.assertNotIn(b"\r\n", manifest.read_bytes())

    # Check that manifest writer does not replace a file appearing at the publication syscall.
    def test_manifest_writer_does_not_replace_a_file_appearing_at_the_publication_syscall(self):
        with temporary_root() as root:
            manifest = root / "catalogue.run.json"

            # Create the competing destination used by this race check.
            def late_destination(source, destination):
                destination.write_text('{"owner": "late process"}\n', encoding="utf-8")
                return rename_exclusive(source, destination)

            with (
                patch("uma_bct.atomic.rename_exclusive", side_effect=late_destination),
                self.assertRaisesRegex(FileExistsError, "already exists"),
            ):
                write_json_atomic(manifest, {"status": "failed"}, replace_existing=False)

            self.assertEqual(manifest.read_text(encoding="utf-8"), '{"owner": "late process"}\n')

    # Check that parent reparse detection is not platform dependent.
    def test_parent_reparse_detection_is_not_platform_dependent(self):
        with temporary_root() as root:
            parent = root / "outputs"
            parent.mkdir()
            target = parent / "catalogue.csv"

            # Return the simulated path kind.
            def fake_kind(path, metadata):
                return "reparse point" if path == parent else None

            with patch("uma_bct.path_safety.redirected_path_kind", side_effect=fake_kind):
                with self.assertRaisesRegex(ValueError, "reparse point"):
                    check_path(target, "Catalogue output")

    # Check that path check wraps reparse inspection errors.
    def test_path_check_wraps_reparse_inspection_errors(self):
        with temporary_root() as root:
            target = root / "catalogue.csv"
            with (
                patch("uma_bct.path_safety.redirected_path_kind", side_effect=OSError("inspection failed")),
                self.assertRaisesRegex(ValueError, "Could not inspect Catalogue output"),
            ):
                check_path(target, "Catalogue output")

    # Check that the code routes setup without starting a batch.
    def test_routes_setup_without_starting_a_batch(self):
        with patch("uma_bct.cli.run_setup", return_value=0) as setup:
            self.assertEqual(main(["setup"]), 0)
        setup.assert_called_once_with()

    # Check that the code routes custom managed setup paths.
    def test_routes_custom_managed_setup_paths(self):
        with patch("uma_bct.cli.run_setup", return_value=0) as setup:
            self.assertEqual(main(["setup", "--path-to-inputs", "Photos", "--results-directory", "Results"]), 0)
        setup.assert_called_once_with(input_directory=Path("Photos"), results_directory=Path("Results"))

    # Check that managed options conflict with direct outputs and force.
    def test_managed_options_conflict_with_direct_outputs_and_force(self):
        for extra in (("-o", "out.csv"), ("--work-dir", "crops"), ("--force",)):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["--results-directory", "Results", *extra])

    # Check that reprocess requires a results directory.
    def test_reprocess_requires_a_results_directory(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["--reprocess"])

    # Check that CLI routes a managed batch.
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

    # Check that CLI rejects XLSX before processing.
    def test_cli_rejects_xlsx_before_processing(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            main(["--path-to-outputs", "catalogue.xlsx"])
        self.assertEqual(raised.exception.code, 2)

    # Check that CLI routes a valid batch.
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

    # Check that CLI rejects catalogue inside review crop directory.
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

    # Check that CLI rejects review crops inside catalogue or manifest paths.
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

    # Check that the code rejects cross drive windows output layouts.
    def test_rejects_cross_drive_windows_output_layouts(self):
        with self.assertRaisesRegex(ValueError, "same Windows drive"):
            validate_output_layout(r"C:\results\catalogue.csv", r"C:\results\catalogue.run.json", r"D:\review\crops")

    # Check that output layout compares physical macOS alias paths.
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


# Group the batch runner checks.
class BatchRunnerTest(unittest.TestCase):
    # Run the with.
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

    # Prepare the outputs.
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

    # Check that run interrupted.
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

    # Check that existing output edit is preserved.
    def assert_existing_output_edit_is_preserved(self, output_kind: str, *, during_publication: bool) -> None:
        with temporary_root() as root:
            _outputs, catalogue, manifest = self.prepare_outputs(root)
            target = catalogue if output_kind == "Catalogue output" else manifest
            edit = f"late {output_kind.casefold()} edit"

            if during_publication:
                raced = False

                # Handle the inject step.
                def inject(first, second, purpose):
                    nonlocal raced
                    matches = same_regular_file_content(first, second, purpose)
                    if purpose == f"{output_kind} publication source" and not raced:
                        target.write_text(edit, encoding="utf-8")
                        raced = True
                    return matches

                fault = patch("uma_bct.run_artifacts.same_regular_file_content", side_effect=inject)
            else:
                # Handle the inject step.
                def inject(paths):
                    backups = backup_existing_results(paths)
                    target.write_text(edit, encoding="utf-8")
                    return backups

                fault = patch("uma_bct.batch_runner.backup_existing_results", side_effect=inject)

            with fault:
                status, summary, _progress = self.run_with(
                    root, FakeBatchPipeline(), [root / "shelf.jpg"], force=True, preserve=True
                )
            self.assertEqual(status, 1)
            self.assertEqual(target.read_text(encoding="utf-8"), edit)
            if target == catalogue:
                self.assertIsNone(summary["catalogue_output"])
            else:
                self.assertEqual(catalogue.read_text(encoding="utf-8"), "old catalogue")
                self.assertIn(".failed-", summary["run_manifest"])

    # Check that the code writes CSV manifest and clean JSON summary.
    def test_writes_csv_manifest_and_clean_json_summary(self):
        with temporary_root() as root:
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

    # Check that the code continues after one image failure.
    def test_continues_after_one_image_failure(self):
        with temporary_root() as root:
            images = [root / "good.jpg", root / "bad.jpg"]
            status, summary, progress = self.run_with(root, FakeBatchPipeline(failing_name="bad.jpg"), images)
            self.assertEqual(status, 1)
            self.assertEqual(summary["status"], "partial")
            self.assertEqual(summary["catalogue_records"], 1)
            self.assertEqual(summary["failure_count"], 1)

    # Check that empty forced run preserves previous results.
    def test_empty_forced_run_preserves_previous_results(self):
        with temporary_root() as root:
            _outputs, catalogue, manifest = self.prepare_outputs(root)
            status, summary, progress = self.run_with(
                root, FakeBatchPipeline(empty=True), [root / "empty.jpg"], force=True, preserve=True
            )
            self.assertEqual(status, 1)
            self.assertEqual(catalogue.read_text(encoding="utf-8"), "old catalogue")
            self.assertEqual(manifest.read_text(encoding="utf-8"), "old manifest")
            self.assertIn(".failed-", summary["run_manifest"])
            self.assertEqual(summary["failure_count"], 1)

    # Check that forced success keeps timestamped previous results.
    def test_forced_success_keeps_timestamped_previous_results(self):
        with temporary_root() as root:
            self.prepare_outputs(root)
            status, summary, progress = self.run_with(root, FakeBatchPipeline(), [root / "shelf.jpg"], force=True)
            self.assertEqual(status, 0)
            self.assertEqual(len(summary["previous_results"]), 2)
            for path in summary["previous_results"]:
                self.assertTrue(Path(path).is_file())

    # Check that preflight failure is reported once and releases the lock.
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

    # Check that existing lock rejects the batch before model work.
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

    # Check that manifest failure restores old CSV and writes a failed run manifest.
    def test_manifest_failure_restores_old_csv_and_writes_a_failed_run_manifest(self):
        with temporary_root() as root:
            _outputs, catalogue, manifest = self.prepare_outputs(root)
            with patch("uma_bct.batch_runner.write_json_atomic", side_effect=manifest_disk_failure(manifest)):
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

    # Check that manifest interruption restores the previous CSV.
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

    # Check that interruption after CSV staging does not cross the manifest boundary.
    def test_interruption_after_csv_staging_does_not_cross_the_manifest_boundary(self):
        with temporary_root() as root:
            outputs, catalogue, manifest = self.prepare_outputs(root)

            # Handle the interrupt after staging step.
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

    # Check that interruption after export return uses the callers transaction state.
    def test_interruption_after_export_return_uses_the_callers_transaction_state(self):
        with temporary_root() as root:
            _outputs, catalogue, manifest = self.prepare_outputs(root)
            # Handle the interrupt after return step.
            def interrupt_after_return(*call_args, **call_kwargs):
                export_results_into(*call_args, **call_kwargs)
                raise KeyboardInterrupt

            self.assert_run_interrupted(
                root, catalogue, manifest,
                patch("uma_bct.batch_runner.export_results_into", side_effect=interrupt_after_return),
            )

            self.assertEqual(catalogue.read_text(encoding="utf-8"), "old catalogue")
            self.assertEqual(manifest.read_text(encoding="utf-8"), "old manifest")

    # Check that interruption after manifest rename restores both previous files.
    def test_interruption_after_manifest_rename_restores_both_previous_files(self):
        with temporary_root() as root:
            _outputs, catalogue, manifest = self.prepare_outputs(root)
            # Handle the interrupt after manifest step.
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

    # Check that interruption after backed file quarantine restores both previous files.
    def test_interruption_after_backed_file_quarantine_restores_both_previous_files(self):
        for interrupted_name in ("catalogue.csv", "catalogue.run.json"):
            with self.subTest(interrupted_name=interrupted_name), temporary_root() as root:
                outputs, catalogue, manifest = self.prepare_outputs(root)
                interrupted_path = outputs / interrupted_name
                real_replace = os.replace
                interrupted = False

                # Handle the interrupt after quarantine step.
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

    # Check that manifest failure removes a new CSV.
    def test_manifest_failure_removes_a_new_csv(self):
        with temporary_root() as root:
            manifest = root / "outputs/catalogue.run.json"
            with patch("uma_bct.batch_runner.write_json_atomic", side_effect=manifest_disk_failure(manifest)):
                status, summary, progress = self.run_with(root, FakeBatchPipeline(), [root / "shelf.jpg"])

            self.assertEqual(status, 1)
            self.assertEqual(summary["status"], "failed")
            self.assertFalse((root / "outputs/catalogue.csv").exists())
            self.assertTrue(Path(summary["run_manifest"]).is_file())

    # Check that rollback failure reports partial and retains the recovery path.
    def test_rollback_failure_reports_partial_and_retains_the_recovery_path(self):
        with temporary_root() as root:
            _outputs, catalogue, manifest = self.prepare_outputs(root)
            with (
                patch("uma_bct.batch_runner.write_json_atomic", side_effect=manifest_disk_failure(manifest)),
                patch("uma_bct.batch_runner.rollback_owned_file", side_effect=OSError("rollback disk failure")),
            ):
                status, summary, progress = self.run_with(root, FakeBatchPipeline(), [root / "shelf.jpg"], force=True)

            self.assertEqual(status, 1)
            self.assertEqual(summary["status"], "partial")
            self.assertTrue(Path(summary["recovery_path"]).is_file())
            self.assertIn(summary["recovery_path"], summary["previous_results"])
            self.assertIn("Rollback failed", progress)

    # Check that new CSV rollback failure reports the CSV as the recovery path.
    def test_new_csv_rollback_failure_reports_the_csv_as_the_recovery_path(self):
        with temporary_root() as root:
            manifest = root / "outputs/catalogue.run.json"
            with (
                patch("uma_bct.batch_runner.write_json_atomic", side_effect=manifest_disk_failure(manifest)),
                patch("uma_bct.batch_runner.rollback_owned_file", side_effect=OSError("rollback disk failure")),
            ):
                status, summary, progress = self.run_with(root, FakeBatchPipeline(), [root / "shelf.jpg"])

            self.assertEqual(status, 1)
            self.assertEqual(summary["status"], "partial")
            self.assertEqual(summary["recovery_path"], str(root / "outputs/catalogue.csv"))
            self.assertTrue(Path(summary["recovery_path"]).is_file())

    # Check that recovery manifest failure keeps stdout valid JSON.
    def test_recovery_manifest_failure_keeps_stdout_valid_json(self):
        with temporary_root() as root:
            _outputs, catalogue, _manifest = self.prepare_outputs(root, manifest_content=None)

            with patch("uma_bct.batch_runner.write_json_atomic", side_effect=OSError("manifest disk failure")):
                status, summary, progress = self.run_with(root, FakeBatchPipeline(), [root / "shelf.jpg"], force=True)

            self.assertEqual(status, 1)
            self.assertEqual(summary["status"], "failed")
            self.assertIsNone(summary["run_manifest"])
            self.assertGreaterEqual(summary["failure_count"], 2)
            self.assertEqual(catalogue.read_text(encoding="utf-8"), "old catalogue")

    # Check that lock cleanup failure keeps JSON and returns lock error status.
    def test_lock_cleanup_failure_keeps_json_and_returns_lock_error_status(self):
        with temporary_root() as root:
            with patch("uma_bct.batch_runner.CatalogueLock.release", side_effect=OSError("cleanup denied")):
                status, summary, progress = self.run_with(root, FakeBatchPipeline(), [root / "shelf.jpg"])

            self.assertEqual(status, 2)
            self.assertEqual(summary["status"], "partial")
            self.assertEqual(summary["failure_count"], 1)
            self.assertIn("Lock cleanup failed", progress)
            self.assertEqual(summary["run_manifest"], str(root / "outputs/catalogue.run.json"))
            manifest = json.loads(Path(summary["run_manifest"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], summary["status"])
            self.assertEqual(manifest["failures"], [{"stage": "lock_cleanup", "error": "cleanup denied"}])
            self.assertEqual(list((root / "outputs").glob("*.run.json")), [root / "outputs/catalogue.run.json"])

    # Check that lock cleanup failure does not overwrite a late manifest edit.
    def test_lock_cleanup_failure_does_not_overwrite_a_late_manifest_edit(self):
        with temporary_root() as root:
            manifest = root / "outputs/catalogue.run.json"

            # Handle the edit manifest then fail step.
            def edit_manifest_then_fail(_lock):
                manifest.write_text("late unbacked manifest edit", encoding="utf-8")
                raise OSError("cleanup denied")

            with patch("uma_bct.batch_runner.CatalogueLock.release", autospec=True, side_effect=edit_manifest_then_fail):
                status, summary, progress = self.run_with(root, FakeBatchPipeline(), [root / "shelf.jpg"])

            self.assertEqual(status, 2)
            self.assertIsNone(summary["run_manifest"])
            self.assertEqual(summary["failure_count"], 2)
            self.assertEqual(manifest.read_text(encoding="utf-8"), "late unbacked manifest edit")
            self.assertIn("was not updated", progress)

    # Check that preflight and lock cleanup failures still emit one JSON document.
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

    # Check that forced backup race does not restore a manifest as CSV.
    def test_forced_backup_race_does_not_restore_a_manifest_as_csv(self):
        with temporary_root() as root:
            _outputs, catalogue, manifest = self.prepare_outputs(root)

            # Handle the catalogue disappears step.
            def catalogue_disappears(paths):
                catalogue.unlink()
                return backup_existing_results(paths)

            with (
                patch("uma_bct.batch_runner.backup_existing_results", side_effect=catalogue_disappears),
                patch("uma_bct.batch_runner.write_json_atomic", side_effect=manifest_disk_failure(manifest)),
            ):
                status, summary, progress = self.run_with(root, FakeBatchPipeline(), [root / "shelf.jpg"], force=True)

            self.assertEqual(status, 1)
            self.assertFalse(catalogue.exists())
            self.assertEqual(manifest.read_text(encoding="utf-8"), "old manifest")
            self.assertTrue(all(not path.casefold().endswith(".csv") for path in summary["previous_results"]))

    # Check that forced backup race restores a catalogue that appears.
    def test_forced_backup_race_restores_a_catalogue_that_appears(self):
        with temporary_root() as root:
            _outputs, catalogue, manifest = self.prepare_outputs(root, catalogue_content=None)

            # Handle the catalogue appears step.
            def catalogue_appears(paths):
                catalogue.write_text("raced catalogue", encoding="utf-8")
                return backup_existing_results(paths)

            with (
                patch("uma_bct.batch_runner.backup_existing_results", side_effect=catalogue_appears),
                patch("uma_bct.batch_runner.write_json_atomic", side_effect=manifest_disk_failure(manifest)),
            ):
                status, summary, progress = self.run_with(root, FakeBatchPipeline(), [root / "shelf.jpg"], force=True)

            self.assertEqual(status, 1)
            self.assertEqual(catalogue.read_text(encoding="utf-8"), "raced catalogue")

    # Check that catalogue appearing after backup is not overwritten.
    def test_catalogue_appearing_after_backup_is_not_overwritten(self):
        with temporary_root() as root:
            _outputs, catalogue, _manifest = self.prepare_outputs(root, catalogue_content=None)

            # Handle the appear after backup step.
            def appear_after_backup(paths):
                backups = backup_existing_results(paths)
                catalogue.write_text("late catalogue", encoding="utf-8")
                return backups

            with patch("uma_bct.batch_runner.backup_existing_results", side_effect=appear_after_backup):
                status, summary, progress = self.run_with(root, FakeBatchPipeline(), [root / "shelf.jpg"], force=True)

            self.assertEqual(status, 1)
            self.assertEqual(catalogue.read_text(encoding="utf-8"), "late catalogue")
            self.assertIsNone(summary["catalogue_output"])

    # Check that existing catalogue changed after backup is not overwritten.
    def test_existing_catalogue_changed_after_backup_is_not_overwritten(self):
        self.assert_existing_output_edit_is_preserved("Catalogue output", during_publication=False)

    # Check that interrupt after restoring a late catalogue edit does not roll it back.
    def test_interrupt_after_restoring_a_late_catalogue_edit_does_not_roll_it_back(self):
        with temporary_root() as root:
            _outputs, catalogue, manifest = self.prepare_outputs(root)

            # Handle the edit after backup step.
            def edit_after_backup(paths):
                backups = backup_existing_results(paths)
                catalogue.write_text("late unbacked edit", encoding="utf-8")
                return backups

            # Handle the interrupt after restore step.
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

    # Check that existing catalogue changed at publication syscall is not overwritten.
    def test_existing_catalogue_changed_at_publication_syscall_is_not_overwritten(self):
        self.assert_existing_output_edit_is_preserved("Catalogue output", during_publication=True)

    # Check that existing manifest changed after backup is not overwritten.
    def test_existing_manifest_changed_after_backup_is_not_overwritten(self):
        self.assert_existing_output_edit_is_preserved("Run manifest", during_publication=False)

    # Check that existing manifest changed at publication syscall is not overwritten.
    def test_existing_manifest_changed_at_publication_syscall_is_not_overwritten(self):
        self.assert_existing_output_edit_is_preserved("Run manifest", during_publication=True)

    # Check that catalogue changed at rollback syscall is not overwritten.
    def test_catalogue_changed_at_rollback_syscall_is_not_overwritten(self):
        with temporary_root() as root:
            _outputs, catalogue, _manifest = self.prepare_outputs(root)
            raced = False

            # Calculate the edit after displaced digest.
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
                status, summary, _progress = self.run_with(
                    root, FakeBatchPipeline(), [root / "shelf.jpg"], force=True, preserve=True
                )

            self.assertTrue(raced)
            self.assertEqual(status, 1)
            self.assertEqual(catalogue.read_text(encoding="utf-8"), "late rollback edit")
            self.assertEqual(summary["status"], "partial")

    # Check that manifest appearing after backup is not overwritten.
    def test_manifest_appearing_after_backup_is_not_overwritten(self):
        with temporary_root() as root:
            _outputs, catalogue, manifest = self.prepare_outputs(
                root, catalogue_content=None, manifest_content=None
            )

            # Handle the appear after backup step.
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
