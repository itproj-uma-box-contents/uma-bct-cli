import argparse
import contextlib
import csv
import io
import json
import os
import stat
import sys
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

from helpers import make_box, make_image, make_transcription, read_excel

from uma_bct.atomic import open_stable_file, rename_new_file
from uma_bct.constants import (
    DATE_OUTPUT_COLUMNS,
    DEFAULT_MIN_DETECTOR_CONFIDENCE,
    DEFAULT_MIN_TRANSCRIPTION_CONFIDENCE,
    LEGACY_OUTPUT_COLUMNS,
    OUTPUT_COLUMNS,
    REVIEW_DATE_OUTPUT_COLUMNS,
)
from uma_bct.domain import BoxResult
from uma_bct.export import export_catalogue
from uma_bct.managed_batch import (
    BOX_IMAGES_DIRECTORY_NAME,
    CATALOGUE_FILE_NAME,
    DETAILS_FILE_NAME,
    PROGRESS_FILE_NAME,
    ManagedInterruption,
    ResultsRootLock,
    allocate_source_names,
    create_batch_directory,
    deserialize_result,
    fingerprint_file,
    publish_reserved_crop,
    read_json_file,
    restore_unlinked_crop_content,
    run_managed_batch,
    sanitize_source_name,
    serialize_result,
)
from uma_bct.pipeline import ImageResult
from uma_bct.util.date_util import Date, DateRange


def managed_arguments(root: Path, *, reprocess: bool = False, detect_missing_boxes: bool = True,
                      file_type: str | None = "csv") -> argparse.Namespace:
    return argparse.Namespace(
        detector="qwen",
        transcriber="qwen",
        ai_dates=False,
        model="qwen-test",
        min_transcription_confidence=DEFAULT_MIN_TRANSCRIPTION_CONFIDENCE,
        min_detector_confidence=DEFAULT_MIN_DETECTOR_CONFIDENCE,
        request_timeout=10.0,
        path_to_inputs=root / "Add Photos Here",
        path_to_outputs=root / "unused.csv",
        path_to_box_images=None,
        force=False,
        results_directory=root / "Results",
        reprocess=reprocess,
        detect_missing_boxes=detect_missing_boxes,
        file_type=file_type,
        json=True,
    )


class ManagedPipeline:
    """Write small crop files while scripting failures, restarts, and model identity."""

    def __init__(
        self, *, interrupt_name: str | None = None, failed_names: tuple[str, ...] = (), port: int = 12000,
        full_image: bool = False, digest: str = "sha256:test"
    ):
        self.interrupt_name = interrupt_name
        self.failed_names = failed_names
        self.work_directory = Path(".")
        self.calls: list[str] = []
        self.full_image = full_image
        self.model_metadata = {
            "model": "qwen-test",
            "model_digest": digest,
            "ollama_host": f"http://127.0.0.1:{port}",
        }

    def process(self, path: Path) -> ImageResult:
        self.calls.append(path.name)
        if path.name == self.interrupt_name:
            raise KeyboardInterrupt
        image = make_image(path)
        if path.name in self.failed_names:
            return ImageResult(
                image, "qwen", "qwen", ("qwen",), "qwen", "failed", (), (), self.model_metadata, date_detector="none"
            )
        crop = self.work_directory / path.stem / ("full_image.png" if self.full_image else "box_001.png")
        crop.parent.mkdir(parents=True)
        crop.write_bytes(f"crop for {path.name}".encode())
        box_id = f"{path.stem}:full-image" if self.full_image else f"{path.stem}:001"
        box = make_box(image=image, box_id=box_id, crop_path=crop)
        return ImageResult(
            image,
            "qwen",
            "qwen",
            ("qwen",),
            "qwen",
            "complete",
            (BoxResult(box, make_transcription(box)),),
            (),
            self.model_metadata,
            date_detector="none",
        )


class ManagedBatchTest(unittest.TestCase):
    def test_managed_interruption_preserves_its_exit_code(self):
        self.assertEqual(ManagedInterruption(143).exit_code, 143)

    def run_managed_batch_and_capture_output(
        self, root: Path, pipeline: ManagedPipeline, *, reprocess: bool = False,
        detect_missing_boxes: bool = True, verification: str | None = None, patches=(),
        min_transcription_confidence: int = DEFAULT_MIN_TRANSCRIPTION_CONFIDENCE,
        file_type: str | None = "csv",
        date_detector: str = "regex",
    ):
        """Run the real checkpoint flow with a scripted local pipeline."""

        arguments = managed_arguments(root, reprocess=reprocess, detect_missing_boxes=detect_missing_boxes,
                                      file_type=file_type)
        arguments.verification = verification
        arguments.ai_dates = date_detector == "qwen"
        arguments.min_transcription_confidence = min_transcription_confidence
        images = sorted(arguments.path_to_inputs.glob("*.jpg"), key=lambda path: path.name)
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch("uma_bct.managed_batch.build_pipeline", return_value=pipeline))
            for fault in patches:
                stack.enter_context(fault)
            stack.enter_context(contextlib.redirect_stdout(stdout))
            stack.enter_context(contextlib.redirect_stderr(stderr))
            status = run_managed_batch(arguments, images, arguments.results_directory)
        return status, json.loads(stdout.getvalue()), stderr.getvalue()

    def prepare_inputs(self, root: Path, *names: str) -> None:
        inputs = root / "Add Photos Here"
        inputs.mkdir()
        for name in names:
            (inputs / name).write_bytes(f"photo {name}".encode())

    @contextlib.contextmanager
    def prepared_root(self, *names: str):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            self.prepare_inputs(root, *names)
            yield root

    def test_opt_in_excel_batch_and_completed_reuse_preserve_reviewer_edits(self):
        with self.prepared_root("Shelf Photo.jpg") as root:
            status, summary, stderr = self.run_managed_batch_and_capture_output(root, ManagedPipeline(), file_type="excel")
            self.assertEqual(status, 0, stderr)
            catalogue = Path(summary["catalogue_output"])
            self.assertEqual(catalogue.name, "Catalogue.xlsx")
            self.assertFalse((catalogue.parent / "Catalogue.csv").exists())
            rows, links = read_excel(catalogue)
            self.assertEqual(rows[0], OUTPUT_COLUMNS)
            self.assertEqual(rows[1][1], "0004")
            self.assertIn("Shelf%20Photo", links["I2"])
            edited = catalogue.read_bytes() + b"reviewer edit"
            catalogue.write_bytes(edited)
            pipeline = ManagedPipeline()
            status, reused, stderr = self.run_managed_batch_and_capture_output(root, pipeline, file_type=None)
            self.assertEqual(status, 0, stderr)
            self.assertEqual(pipeline.calls, [])
            self.assertEqual(reused["catalogue_output"], str(catalogue))
            self.assertTrue(reused["catalogue_modified"])
            self.assertEqual(catalogue.read_bytes(), edited)

    def test_staged_excel_resumes_with_matching_hash_without_model_work(self):
        with self.prepared_root("01.jpg") as root:
            status, interrupted, _ = self.run_managed_batch_and_capture_output(
                root, ManagedPipeline(), file_type="excel",
                patches=(patch("uma_bct.managed_batch.file_digest", side_effect=KeyboardInterrupt),),
            )
            self.assertEqual(status, 130)
            batch = Path(interrupted["batch_directory"])
            pending = batch / ".uma-bct-catalogue-pending.xlsx"
            expected = pending.read_bytes()
            pipeline = ManagedPipeline()
            status, summary, stderr = self.run_managed_batch_and_capture_output(root, pipeline, file_type=None)
            self.assertEqual(status, 0, stderr)
            self.assertEqual(pipeline.calls, [])
            self.assertEqual(Path(summary["catalogue_output"]).read_bytes(), expected)
            self.assertFalse(pending.exists())

    def test_completed_csv_requires_explicit_reprocessing_for_a_new_excel_batch(self):
        with self.prepared_root("01.jpg") as root:
            status, first, stderr = self.run_managed_batch_and_capture_output(root, ManagedPipeline())
            self.assertEqual(status, 0, stderr)
            csv_path = Path(first["catalogue_output"])
            edited = csv_path.read_bytes() + b"\nreviewer note"
            csv_path.write_bytes(edited)

            # A format option must not overwrite a completed or edited batch.
            pipeline = ManagedPipeline()
            status, reused, stderr = self.run_managed_batch_and_capture_output(root, pipeline, file_type="excel")
            self.assertEqual(status, 0, stderr)
            self.assertEqual(reused["catalogue_output"], str(csv_path))
            self.assertEqual(pipeline.calls, [])
            self.assertEqual(csv_path.read_bytes(), edited)
            self.assertIn("--format excel --reprocess", stderr)

            # Explicit reprocessing creates a separate workbook and preserves
            # the original CSV, including changes made during human review.
            pipeline = ManagedPipeline()
            status, fresh, stderr = self.run_managed_batch_and_capture_output(
                root, pipeline, file_type="excel", reprocess=True,
            )
            self.assertEqual(status, 0, stderr)
            self.assertEqual(pipeline.calls, ["01.jpg"])
            self.assertEqual(Path(fresh["catalogue_output"]).name, "Catalogue.xlsx")
            self.assertNotEqual(fresh["batch_directory"], first["batch_directory"])
            self.assertEqual(csv_path.read_bytes(), edited)

    def test_new_batch_defaults_to_csv_without_html_or_verification(self):
        with self.prepared_root("01.jpg") as root:
            pipeline = ManagedPipeline()
            pipeline.transcriber = Mock()
            status, summary, stderr = self.run_managed_batch_and_capture_output(
                root, pipeline, file_type=None,
            )
            self.assertEqual(status, 0, stderr)
            self.assertEqual(Path(summary["catalogue_output"]).name, "Catalogue.csv")
            self.assertNotIn("review_output", summary)
            self.assertFalse((Path(summary["batch_directory"]) / "Review.html").exists())
            pipeline.transcriber.verify.assert_not_called()


    def test_explicit_format_change_blocks_resume_before_model_work(self):
        with self.prepared_root("01.jpg", "02.jpg") as root:
            _, interrupted, _ = self.run_managed_batch_and_capture_output(
                root, ManagedPipeline(interrupt_name="02.jpg"), file_type="excel",
            )
            progress = Path(interrupted["batch_directory"]) / PROGRESS_FILE_NAME
            saved = progress.read_bytes()
            pipeline = ManagedPipeline()
            status, _, stderr = self.run_managed_batch_and_capture_output(root, pipeline, file_type="csv")
            self.assertEqual(status, 2)
            self.assertIn("Cannot change catalogue format", stderr)
            self.assertEqual(pipeline.calls, [])
            self.assertEqual(progress.read_bytes(), saved)

    def test_another_format_pending_file_blocks_resume_and_is_preserved(self):
        for file_type, alternate in (("excel", "csv"), ("csv", "xlsx")):
            with self.subTest(file_type=file_type), self.prepared_root("01.jpg") as root:
                _, interrupted, _ = self.run_managed_batch_and_capture_output(
                    root, ManagedPipeline(), file_type=file_type,
                    patches=(patch("uma_bct.managed_batch.file_digest", side_effect=KeyboardInterrupt),),
                )
                batch = Path(interrupted["batch_directory"])
                progress = batch / PROGRESS_FILE_NAME
                saved = progress.read_bytes()
                unexpected = batch / f".uma-bct-catalogue-pending.{alternate}"
                unexpected.write_bytes(b"unknown file")
                pipeline = ManagedPipeline()
                status, _, stderr = self.run_managed_batch_and_capture_output(root, pipeline, file_type=None)
                self.assertEqual(status, 2)
                self.assertIn("Unexpected pending catalogue format", stderr)
                self.assertEqual(pipeline.calls, [])
                self.assertEqual(unexpected.read_bytes(), b"unknown file")
                self.assertEqual(progress.read_bytes(), saved)
                self.assertFalse((batch / DETAILS_FILE_NAME).exists())

    def test_legacy_csv_publication_checkpoint_is_still_readable(self):
        with self.prepared_root("01.jpg") as root:
            def stop_publication(source, destination, *, existing_message):
                if destination.name == CATALOGUE_FILE_NAME:
                    raise KeyboardInterrupt
                return rename_new_file(source, destination, existing_message=existing_message)

            _, interrupted, _ = self.run_managed_batch_and_capture_output(
                root, ManagedPipeline(),
                patches=(patch("uma_bct.managed_batch.rename_new_file", side_effect=stop_publication),),
            )
            batch = Path(interrupted["batch_directory"])
            progress_path = batch / PROGRESS_FILE_NAME
            progress = read_json_file(progress_path, "Run progress")
            progress.pop("file_type")
            progress.pop("html_review", None)
            progress["configuration"].pop("date_detector")
            for item in progress["inputs"]:
                item["result"].pop("date_detector")
                for record in item["result"]["records"]:
                    record["transcription"].pop("dates")
            publication = progress["publication"]
            publication["pending_csv"] = publication.pop("pending_catalogue")
            progress_path.write_text(json.dumps(progress), encoding="utf-8")
            pipeline = ManagedPipeline()
            status, summary, stderr = self.run_managed_batch_and_capture_output(root, pipeline, file_type=None)
            self.assertEqual(status, 0, stderr)
            self.assertEqual(pipeline.calls, [])
            self.assertEqual(Path(summary["catalogue_output"]).name, CATALOGUE_FILE_NAME)

    def test_legacy_staging_preserves_historical_catalogue_layouts(self):
        for columns, date_detector, file_type in (
            (OUTPUT_COLUMNS, "regex", "csv"),
            (LEGACY_OUTPUT_COLUMNS, "none", "csv"), (LEGACY_OUTPUT_COLUMNS, "none", "excel"),
            (DATE_OUTPUT_COLUMNS, "regex", "csv"),
            (REVIEW_DATE_OUTPUT_COLUMNS, "regex", "csv"),
            (REVIEW_DATE_OUTPUT_COLUMNS, "regex", "excel"),
        ):
            with self.subTest(columns=columns, file_type=file_type), self.prepared_root("01.jpg") as root:
                def stop_review(*args, **kwargs):
                    raise KeyboardInterrupt

                _, interrupted, _ = self.run_managed_batch_and_capture_output(
                    root, ManagedPipeline(), file_type=file_type,
                    patches=(patch("uma_bct.managed_batch.file_digest", side_effect=stop_review),),
                )
                batch = Path(interrupted["batch_directory"])
                progress_path = batch / PROGRESS_FILE_NAME
                progress = read_json_file(progress_path, "Run progress")
                progress["configuration"].pop("date_detector")
                progress["publication"].pop("output_columns", None)
                result = progress["inputs"][0]["result"]
                if date_detector == "none":
                    result.pop("date_detector")
                    for record in result["records"]:
                        record["transcription"].pop("dates")
                else:
                    result["date_detector"] = date_detector
                    result["records"][0]["transcription"]["dates"] = [{"year": 1950, "month": None, "day": None}]
                progress_path.write_text(json.dumps(progress), encoding="utf-8")
                results = [deserialize_result(result, batch)]
                pending = batch / progress["publication"]["pending_catalogue"]
                export_catalogue(results, pending, output_columns=columns, trusted_crop_links=True)
                catalogue_bytes = pending.read_bytes()
                # An omitted CLI option keeps the original extraction method on resume.
                arguments = managed_arguments(root, file_type=file_type)
                arguments.ai_dates = False
                stdout, stderr = io.StringIO(), io.StringIO()
                with (contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr),
                      patch("uma_bct.managed_batch.build_pipeline", side_effect=AssertionError("must not load"))):
                    status = run_managed_batch(arguments, list(arguments.path_to_inputs.glob("*.jpg")), arguments.results_directory)
                self.assertEqual(status, 0, stderr.getvalue())
                summary = json.loads(stdout.getvalue())
                self.assertEqual(Path(summary["catalogue_output"]).read_bytes(), catalogue_bytes)
                self.assertEqual(getattr(arguments, "_resume_date_detector", "regex"), date_detector)

    def test_changed_date_detector_cannot_mix_an_unfinished_batch(self):
        with self.prepared_root("01.jpg", "02.jpg") as root:
            _, interrupted, _ = self.run_managed_batch_and_capture_output(
                root, ManagedPipeline(interrupt_name="02.jpg")
            )
            batch = Path(interrupted["batch_directory"])
            original_progress = (batch / PROGRESS_FILE_NAME).read_bytes()
            pipeline = ManagedPipeline()
            status, _, stderr = self.run_managed_batch_and_capture_output(root, pipeline, date_detector="qwen")
            self.assertEqual(status, 2)
            self.assertIn("processing configuration changed", stderr)
            self.assertEqual(pipeline.calls, [])
            self.assertEqual((batch / PROGRESS_FILE_NAME).read_bytes(), original_progress)

    def test_ai_date_batch_requires_the_ai_switch_again_on_resume(self):
        with self.prepared_root("01.jpg", "02.jpg") as root:
            arguments = managed_arguments(root)
            arguments.ai_dates = True
            images = sorted(arguments.path_to_inputs.glob("*.jpg"))
            first = ManagedPipeline(interrupt_name="02.jpg")
            with (patch("uma_bct.managed_batch.build_pipeline", return_value=first) as builder,
                  contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO())):
                self.assertEqual(run_managed_batch(arguments, images, arguments.results_directory), 130)
                self.assertEqual(builder.call_args.kwargs["date_detector_name"], "qwen")
            status, _, stderr = self.run_managed_batch_and_capture_output(root, ManagedPipeline(), date_detector="regex")
            self.assertEqual(status, 2, stderr)
            self.assertIn("processing configuration changed", stderr)
            resumed = managed_arguments(root)
            resumed.ai_dates = True
            pipeline = ManagedPipeline()
            with (patch("uma_bct.managed_batch.build_pipeline", return_value=pipeline) as builder,
                  contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO())):
                self.assertEqual(run_managed_batch(resumed, images, resumed.results_directory), 0)
                self.assertEqual(builder.call_args.kwargs["date_detector_name"], "qwen")
            self.assertEqual(pipeline.calls, ["02.jpg"])


    def test_added_photo_during_run_is_deferred_and_only_it_runs_next_time(self):
        class AddingPipeline(ManagedPipeline):
            def process(self, path):
                result = super().process(path)
                path.with_name("02.jpg").write_bytes(b"new photo")
                return result

        with self.prepared_root("01.jpg") as root:
            status, first, stderr = self.run_managed_batch_and_capture_output(root, AddingPipeline())
            self.assertEqual(status, 0)
            self.assertEqual(first["images_total"], 1)
            self.assertIn("next launch", stderr)
            pipeline = ManagedPipeline()
            status, second, _ = self.run_managed_batch_and_capture_output(root, pipeline)
            self.assertEqual(status, 0)
            self.assertEqual(pipeline.calls, ["02.jpg"])
            self.assertEqual(second["images_total"], 1)
            pipeline = ManagedPipeline()
            status, third, _ = self.run_managed_batch_and_capture_output(root, pipeline)
            self.assertEqual(status, 0)
            self.assertEqual(pipeline.calls, [])
            self.assertTrue(third["already_complete"])

    def test_added_photo_does_not_block_resume(self):
        with self.prepared_root("01.jpg", "02.jpg") as root:
            self.run_managed_batch_and_capture_output(root, ManagedPipeline(interrupt_name="02.jpg"))
            (root / "Add Photos Here/00.jpg").write_bytes(b"new photo sorted first")
            pipeline = ManagedPipeline()
            status, summary, _ = self.run_managed_batch_and_capture_output(root, pipeline)
            self.assertEqual(status, 0)
            self.assertEqual(pipeline.calls, ["02.jpg"])
            self.assertEqual(summary["images_total"], 2)

    def test_history_reuse_preserves_failures_across_batches(self):
        with self.prepared_root("01.jpg") as root:
            self.run_managed_batch_and_capture_output(root, ManagedPipeline(failed_names=("01.jpg",)))
            (root / "Add Photos Here/02.jpg").write_bytes(b"new photo")
            _, second, _ = self.run_managed_batch_and_capture_output(root, ManagedPipeline())
            pipeline = ManagedPipeline()
            status, summary, _ = self.run_managed_batch_and_capture_output(root, pipeline)

            self.assertEqual(status, 1)
            self.assertEqual(summary["status"], "partial")
            self.assertEqual(summary["images_total"], 2)
            self.assertEqual(summary["images_reused"], 2)
            self.assertEqual(summary["images_failed"], 1)
            self.assertEqual(summary["failure_count"], 1)
            self.assertEqual(summary["catalogue_records"], 1)
            self.assertEqual(summary["catalogue_outputs"], [second["catalogue_output"]])
            self.assertEqual(len(summary["run_manifests"]), 2)
            self.assertIsNone(summary["batch_directory"])
            self.assertIsNone(summary["catalogue_output"])
            self.assertEqual(pipeline.calls, [])

    def test_history_reuse_aggregates_successful_batches_and_current_subsets(self):
        with self.prepared_root("01.jpg", "02.jpg") as root:
            _, first, _ = self.run_managed_batch_and_capture_output(root, ManagedPipeline())
            (root / "Add Photos Here/03.jpg").write_bytes(b"new photo")
            _, second, _ = self.run_managed_batch_and_capture_output(root, ManagedPipeline())
            catalogue = Path(first["catalogue_output"])
            catalogue.write_bytes(b"human corrections")
            for removed, count in ((None, 3), ("02.jpg", 2), ("03.jpg", 1)):
                with self.subTest(removed=removed):
                    if removed:
                        (root / "Add Photos Here" / removed).unlink()
                    pipeline = ManagedPipeline()
                    status, summary, _ = self.run_managed_batch_and_capture_output(root, pipeline)
                    self.assertEqual(status, 0)
                    self.assertEqual(summary["images_total"], count)
                    self.assertEqual(summary["images_complete"], count)
                    self.assertEqual(summary["images_processed"], count)
                    self.assertEqual(summary["images_reused"], count)
                    self.assertEqual(summary["current_input_records"], count)
                    self.assertEqual(summary["catalogue_records"], 3 if count > 1 else 2)
                    self.assertTrue(summary["catalogue_modified"])
                    self.assertEqual(catalogue.read_bytes(), b"human corrections")
                    expected = [first["catalogue_output"]]
                    if count > 1:
                        expected.append(second["catalogue_output"])
                        self.assertIsNone(summary["catalogue_output"])
                        self.assertIsNone(summary["run_manifest"])
                        self.assertNotIn("review_outputs", summary)
                    else:
                        self.assertEqual(summary["catalogue_output"], first["catalogue_output"])
                        self.assertEqual(summary["batch_directory"], first["batch_directory"])
                    self.assertCountEqual(summary["catalogue_outputs"], expected)
                    self.assertEqual(pipeline.calls, [])

    def test_history_reuse_excludes_removed_failures_and_prefers_latest_reprocessing(self):
        with self.prepared_root("01.jpg", "02.jpg") as root:
            self.run_managed_batch_and_capture_output(root, ManagedPipeline(failed_names=("01.jpg",)))
            (root / "Add Photos Here/01.jpg").unlink()
            status, summary, _ = self.run_managed_batch_and_capture_output(root, ManagedPipeline())
            self.assertEqual(status, 0)
            self.assertEqual(summary["images_total"], 1)
            self.assertEqual(summary["failure_count"], 0)
            self.assertEqual(summary["images_failed"], 0)

            _, latest, _ = self.run_managed_batch_and_capture_output(
                root, ManagedPipeline(failed_names=("02.jpg",)), reprocess=True
            )
            status, summary, _ = self.run_managed_batch_and_capture_output(root, ManagedPipeline())
            self.assertEqual(status, 1)
            self.assertEqual(summary["status"], "failed")
            self.assertEqual(summary["images_total"], 1)
            self.assertEqual(summary["images_failed"], 1)
            self.assertEqual(summary["batch_directory"], latest["batch_directory"])
            self.assertEqual(summary["catalogue_outputs"], [])

    def test_history_reuse_preserves_partial_photo_outcomes(self):
        class PartialPipeline(ManagedPipeline):
            def process(self, path):
                return replace(super().process(path), status="partial")

        with self.prepared_root("01.jpg") as root:
            self.run_managed_batch_and_capture_output(root, PartialPipeline())
            (root / "Add Photos Here/02.jpg").write_bytes(b"new photo")
            self.run_managed_batch_and_capture_output(root, ManagedPipeline())
            status, summary, _ = self.run_managed_batch_and_capture_output(root, ManagedPipeline())
            self.assertEqual(status, 1)
            self.assertEqual(summary["status"], "partial")
            self.assertEqual(summary["images_partial"], 1)
            self.assertEqual(summary["images_complete"], 1)
            self.assertEqual(summary["catalogue_records"], 2)

    def test_reviewer_files_metadata_and_csv_edits_are_preserved(self):
        with self.prepared_root("01.jpg") as root:
            _, summary, _ = self.run_managed_batch_and_capture_output(root, ManagedPipeline())
            batch = Path(summary["batch_directory"])
            files = [batch / ".DS_Store", batch / "Box Images/.DS_Store", batch / "Catalogue corrected.csv"]
            for path in files:
                path.write_bytes(b"reviewer content")
            catalogue = batch / CATALOGUE_FILE_NAME
            catalogue.write_bytes(b"reviewed catalogue")
            status, summary, stderr = self.run_managed_batch_and_capture_output(root, ManagedPipeline())
            self.assertEqual(status, 0)
            self.assertTrue(summary["catalogue_modified"])
            self.assertIn("preserved", stderr)
            self.assertEqual(catalogue.read_bytes(), b"reviewed catalogue")
            self.assertTrue(all(path.read_bytes() == b"reviewer content" for path in files))

    def test_publication_resume_does_not_require_ollama(self):
        with self.prepared_root("01.jpg") as root:
            with patch("uma_bct.managed_batch.promote_details", side_effect=KeyboardInterrupt):
                self.run_managed_batch_and_capture_output(root, ManagedPipeline())
            args = managed_arguments(root)
            with patch("uma_bct.managed_batch.build_pipeline", side_effect=AssertionError("no inference needed")), \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                status = run_managed_batch(args, sorted(args.path_to_inputs.glob("*.jpg")), args.results_directory)
            self.assertEqual(status, 0)

    def test_verification_interruption_resumes_without_first_pass_inference(self):
        from uma_bct.verification import reading

        with self.prepared_root("01.jpg") as root:
            args = managed_arguments(root)
            args.verification = "always"
            initial = ManagedPipeline()
            initial.transcriber = Mock()
            initial.transcriber.verify.side_effect = KeyboardInterrupt
            with patch("uma_bct.managed_batch.build_pipeline", return_value=initial), \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                status = run_managed_batch(args, sorted(args.path_to_inputs.glob("*.jpg")), args.results_directory)
            self.assertEqual(status, 130)
            progress_path = next(args.results_directory.glob("Batch */Run Progress.json"))
            progress = json.loads(progress_path.read_text())
            original = reading(progress["inputs"][0]["result"]["records"][0]["verification"]["original"][0])
            resumed = ManagedPipeline()
            resumed.transcriber = Mock()
            resumed.transcriber.verify.return_value = [original]
            with patch("uma_bct.managed_batch.build_pipeline", return_value=resumed), \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                status = run_managed_batch(args, sorted(args.path_to_inputs.glob("*.jpg")), args.results_directory)
            self.assertEqual(status, 0)
            self.assertEqual(resumed.calls, [])
            self.assertEqual(resumed.transcriber.verify.call_count, 1)
            self.assertFalse(progress_path.with_name("Review.html").exists())

    def test_cached_verification_readings_finalize_without_model_preflight(self):
        from uma_bct.managed_batch import write_progress

        for cached_reading in ("second", "third"):
            with self.subTest(cached_reading=cached_reading), self.prepared_root("01.jpg") as root:
                pipeline = ManagedPipeline()
                pipeline.transcriber = Mock()
                handwriting = ("corrected content",) if cached_reading == "third" else ()
                pipeline.transcriber.verify.side_effect = lambda box, handwriting=handwriting, **kw: [
                    make_transcription(box, handwritten=handwriting)
                ]
                pipeline.transcriber.resolve_discrepancy.side_effect = lambda box, *a, handwriting=handwriting, **kw: [
                    make_transcription(box, handwritten=handwriting)
                ]

                def interrupt_after_reading(batch, progress, cached_reading=cached_reading):
                    write_progress(batch, progress)
                    records = progress["inputs"][0].get("result", {}).get("records", [])
                    audit = records[0].get("verification", {}) if records else {}
                    if (progress["phase"] == "verifying" and cached_reading in audit
                            and audit["status"] == "pending"):
                        raise KeyboardInterrupt

                status, first, _ = self.run_managed_batch_and_capture_output(
                    root, pipeline, verification="always",
                    patches=(patch("uma_bct.managed_batch.write_progress", side_effect=interrupt_after_reading),),
                )
                self.assertEqual(status, 130)
                self.assertEqual(pipeline.transcriber.verify.call_count, 1)
                self.assertEqual(pipeline.transcriber.resolve_discrepancy.call_count, cached_reading == "third")
                resumed = ManagedPipeline()
                status, summary, _ = self.run_managed_batch_and_capture_output(
                    root, resumed, verification="always",
                    patches=(patch("uma_bct.managed_batch.build_pipeline", side_effect=AssertionError("offline")),),
                )
                self.assertEqual(status, 0)
                self.assertEqual(summary["failure_count"], 0)
                self.assertEqual(summary["images_reused"], 1)
                self.assertEqual(resumed.calls, [])
                batch = Path(first["batch_directory"])
                details = json.loads((batch / DETAILS_FILE_NAME).read_text())
                self.assertTrue(details["verification"]["complete"])
                self.assertEqual(len(details["endpoint_history"]), 1)
                audit = details["inputs"][0]["result"]["records"][0]["verification"]
                self.assertEqual(audit["status"], "complete")
                self.assertFalse((batch / "Review.html").exists())
                with (batch / CATALOGUE_FILE_NAME).open(newline="", encoding="utf-8") as source:
                    row = next(csv.DictReader(source))
                self.assertEqual(row["Description"], "; ".join(handwriting))

    def test_pending_verification_blocks_changed_model_digest(self):
        with self.prepared_root("01.jpg") as root:
            pipeline = ManagedPipeline()
            pipeline.transcriber = Mock()
            pipeline.transcriber.verify.side_effect = KeyboardInterrupt
            status, first, _ = self.run_managed_batch_and_capture_output(root, pipeline, verification="always")
            self.assertEqual(status, 130)
            resumed = ManagedPipeline(digest="sha256:changed")
            resumed.transcriber = Mock()
            status, summary, stderr = self.run_managed_batch_and_capture_output(
                root, resumed, verification="always"
            )
            self.assertEqual(status, 2)
            self.assertIn("model digest changed", stderr)
            self.assertEqual(resumed.calls, [])
            resumed.transcriber.verify.assert_not_called()
            self.assertIsNone(summary["catalogue_output"])
            batch = Path(first["batch_directory"])
            progress = json.loads((batch / PROGRESS_FILE_NAME).read_text())
            self.assertEqual(progress["model"]["model_digest"], "sha256:test")
            self.assertEqual(len(progress["endpoint_history"]), 1)
            audit = progress["inputs"][0]["result"]["records"][0]["verification"]
            self.assertEqual(audit["status"], "pending")
            self.assertFalse((batch / DETAILS_FILE_NAME).exists())


    def test_added_photo_is_not_read_at_publication(self):
        class AddingPipeline(ManagedPipeline):
            def process(self, path):
                result = super().process(path)
                path.with_name("copying.jpg").write_bytes(b"not yet a finished image")
                return result

        from uma_bct.managed_batch import fingerprint_file

        def reject_extra(path, purpose):
            self.assertNotEqual(path.name, "copying.jpg")
            return fingerprint_file(path, purpose)

        with self.prepared_root("01.jpg") as root, \
             patch("uma_bct.managed_batch.fingerprint_file", side_effect=reject_extra):
            status, _, _ = self.run_managed_batch_and_capture_output(root, AddingPipeline())
            self.assertEqual(status, 0)

    def test_zero_row_batch_preserves_unrelated_html(self):
        class FailedPipeline(ManagedPipeline):
            def process(self, path):
                (self.work_directory.parents[1] / "Review.html").write_text("unexpected review")
                return super().process(path)

        with self.prepared_root("01.jpg") as root:
            status, summary, _ = self.run_managed_batch_and_capture_output(root, FailedPipeline(failed_names=("01.jpg",)))
            self.assertEqual(status, 1)
            batch = Path(summary["batch_directory"])
            self.assertEqual((batch / "Review.html").read_text(), "unexpected review")
            self.assertTrue((batch / "Run Details.json").exists())

    def test_managed_catalogue_honours_missing_box_option(self):
        class GapPipeline(ManagedPipeline):
            def process(self, path: Path) -> ImageResult:
                result = super().process(path)
                record = result.records[0]
                unit = "1" if path.name == "01.jpg" else "3"
                transcription = make_transcription(record.box, unit=unit)
                return replace(result, records=(BoxResult(record.box, transcription),))

        for detect_missing_boxes, expected_numbers in (
            (True, ["0001", "0002", "0003"]),
            (False, ["0001", "0003"]),
        ):
            with self.subTest(detect_missing_boxes=detect_missing_boxes), self.prepared_root("01.jpg", "03.jpg") as root:
                status, summary, _stderr = self.run_managed_batch_and_capture_output(
                    root, GapPipeline(), detect_missing_boxes=detect_missing_boxes
                )
                with (Path(summary["batch_directory"]) / CATALOGUE_FILE_NAME).open(newline="", encoding="utf-8") as source:
                    rows = list(csv.DictReader(source))
                self.assertEqual(status, 0)
                self.assertEqual(summary["catalogue_records"], len(expected_numbers))
                self.assertEqual([row["Unit Number"] for row in rows], expected_numbers)
                if detect_missing_boxes:
                    self.assertEqual(rows[1]["Review Flags"], "Missing Box")

                # Full reuse retains the actual generated row count, including suggestions.
                status, reused, _ = self.run_managed_batch_and_capture_output(
                    root, ManagedPipeline(), detect_missing_boxes=detect_missing_boxes
                )
                self.assertEqual(status, 0)
                self.assertEqual(reused["catalogue_records"], len(rows))
                self.assertEqual(reused["current_input_records"], 2)
                self.assertEqual(reused["catalogue_record_counts"], {summary["catalogue_output"]: len(rows)})

                # A subset changes the selected photo count, never the saved file's count.
                (root / "Add Photos Here/03.jpg").unlink()
                status, subset, _ = self.run_managed_batch_and_capture_output(root, ManagedPipeline())
                self.assertEqual(status, 0)
                self.assertEqual(subset["images_total"], 1)
                self.assertEqual(subset["current_input_records"], 1)
                self.assertEqual(subset["catalogue_records"], len(rows))
                self.assertEqual(subset["catalogue_output"], summary["catalogue_output"])

                (root / "Add Photos Here/05.jpg").write_bytes(b"new photo")
                _, second, _ = self.run_managed_batch_and_capture_output(root, GapPipeline())
                status, aggregate, _ = self.run_managed_batch_and_capture_output(root, ManagedPipeline())
                self.assertEqual(status, 0)
                self.assertEqual(aggregate["images_total"], 2)
                self.assertEqual(aggregate["current_input_records"], 2)
                self.assertEqual(aggregate["catalogue_records"], len(rows) + 1)
                self.assertEqual(aggregate["catalogue_record_counts"], {
                    summary["catalogue_output"]: len(rows), second["catalogue_output"]: 1,
                })

    def test_reused_failed_subset_does_not_inherit_historical_catalogue_success(self):
        with self.prepared_root("01.jpg", "02.jpg") as root:
            _, original, _ = self.run_managed_batch_and_capture_output(
                root, ManagedPipeline(failed_names=("01.jpg",))
            )
            (root / "Add Photos Here/02.jpg").unlink()
            status, summary, _ = self.run_managed_batch_and_capture_output(root, ManagedPipeline())
            self.assertEqual(status, 1)
            self.assertEqual(summary["status"], "failed")
            self.assertEqual(summary["images_total"], 1)
            self.assertEqual(summary["images_failed"], 1)
            self.assertEqual(summary["failure_count"], 1)
            self.assertEqual(summary["current_input_records"], 0)
            self.assertEqual(summary["catalogue_records"], 1)
            self.assertEqual(summary["catalogue_record_counts"], {original["catalogue_output"]: 1})

    def test_missing_box_option_change_blocks_resume(self):
        with self.prepared_root("01.jpg", "02.jpg") as root:
            self.run_managed_batch_and_capture_output(root, ManagedPipeline(interrupt_name="02.jpg"))
            status, _summary, stderr = self.run_managed_batch_and_capture_output(
                root, ManagedPipeline(), detect_missing_boxes=False
            )
            self.assertEqual(status, 2)
            self.assertIn("processing configuration changed", stderr)

    def test_legacy_checkpoint_resumes_without_verification(self):
        with self.prepared_root("01.jpg", "02.jpg") as root:
            status, interrupted, _ = self.run_managed_batch_and_capture_output(
                root, ManagedPipeline(interrupt_name="02.jpg")
            )
            self.assertEqual(status, 130)
            batch = Path(interrupted["batch_directory"])
            progress_path = batch / PROGRESS_FILE_NAME
            progress = read_json_file(progress_path, "Run progress")
            progress["configuration"].pop("detect_missing_boxes")
            progress["configuration"].pop("date_detector")
            for item in progress["inputs"]:
                if result := item.get("result"):
                    result.pop("date_detector")
                    for record in result["records"]:
                        record["transcription"].pop("dates")
            progress.pop("verification")
            progress.pop("file_type")
            progress_path.write_text(json.dumps(progress), encoding="utf-8")
            crops = {path: path.read_bytes() for path in (batch / BOX_IMAGES_DIRECTORY_NAME).iterdir()}

            resumed = ManagedPipeline()
            resumed.transcriber = Mock()
            with patch.dict(os.environ, {"UMA_BCT_VERIFICATION": "always"}):
                status, summary, _ = self.run_managed_batch_and_capture_output(root, resumed, file_type=None)

            self.assertEqual(status, 0)
            self.assertEqual(resumed.calls, ["02.jpg"])
            self.assertEqual(summary["images_reused"], 1)
            resumed.transcriber.verify.assert_not_called()
            resumed.transcriber.resolve_discrepancy.assert_not_called()
            details = read_json_file(Path(summary["run_manifest"]), "Run details")
            self.assertEqual(Path(summary["catalogue_output"]).suffix, ".csv")
            self.assertEqual(details["verification"], {"mode": "never", "decision": False, "complete": True})
            for path, content in crops.items():
                self.assertEqual(path.read_bytes(), content)

    def test_older_confidence_threshold_requires_original_option_to_resume(self):
        with self.prepared_root("01.jpg", "02.jpg") as root:
            status, interrupted, _ = self.run_managed_batch_and_capture_output(
                root, ManagedPipeline(interrupt_name="02.jpg"), min_transcription_confidence=80
            )
            self.assertEqual(status, 130)
            progress_path = Path(interrupted["batch_directory"]) / PROGRESS_FILE_NAME
            progress_before = progress_path.read_bytes()
            rejected = ManagedPipeline()
            status, _, stderr = self.run_managed_batch_and_capture_output(root, rejected)
            self.assertEqual(status, 2)
            self.assertEqual(rejected.calls, [])
            self.assertIn("processing configuration changed", stderr)
            self.assertEqual(progress_path.read_bytes(), progress_before)

            resumed = ManagedPipeline()
            status, summary, _ = self.run_managed_batch_and_capture_output(
                root, resumed, min_transcription_confidence=80
            )
            self.assertEqual(status, 0)
            self.assertTrue(summary["resumed"])
            self.assertEqual(summary["images_reused"], 1)
            self.assertEqual(resumed.calls, ["02.jpg"])
            details = read_json_file(Path(summary["run_manifest"]), "Run details")
            self.assertEqual(details["configuration"]["minimum_transcription_confidence"], 80)

    def test_legacy_checkpoint_still_rejects_changed_processing_options(self):
        for change in ("missing_boxes", "confidence"):
            with self.subTest(change=change), self.prepared_root("01.jpg", "02.jpg") as root:
                _, interrupted, _ = self.run_managed_batch_and_capture_output(
                    root, ManagedPipeline(interrupt_name="02.jpg")
                )
                progress_path = Path(interrupted["batch_directory"]) / PROGRESS_FILE_NAME
                progress = read_json_file(progress_path, "Run progress")
                progress["configuration"].pop("detect_missing_boxes")
                progress.pop("verification")
                if change == "confidence":
                    progress["configuration"]["minimum_transcription_confidence"] = 90
                progress_path.write_text(json.dumps(progress), encoding="utf-8")
                pipeline = ManagedPipeline()

                status, _, stderr = self.run_managed_batch_and_capture_output(
                    root, pipeline, detect_missing_boxes=change != "missing_boxes"
                )

                self.assertEqual(status, 2)
                self.assertEqual(pipeline.calls, [])
                self.assertIn("processing configuration changed", stderr)

    def test_stable_read_rejects_a_short_read_hidden_by_restored_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.bin"
            path.write_bytes(b"a" * (2 * 1024 * 1024))
            metadata = path.stat()
            with self.assertRaisesRegex(ValueError, "changed while it was being read"):
                with open_stable_file(path, "Source") as (source, _before):
                    source.read(1024 * 1024)
                    path.write_bytes(b"b" * metadata.st_size)
                    os.utime(path, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))

    def test_json_read_stays_bounded_if_a_small_file_grows(self):
        metadata = Mock(st_size=1)

        @contextlib.contextmanager
        def growing_file(*_args, **_kwargs):
            yield io.BytesIO(b"x" * (32 * 1024 * 1024 + 1)), metadata

        with (
            patch("uma_bct.managed_batch.open_stable_file", side_effect=growing_file),
            self.assertRaisesRegex(RuntimeError, "unexpectedly large"),
        ):
            read_json_file(Path("progress.json"), "Run progress")

    @unittest.skipIf(os.name == "nt", "POSIX private-file mode")
    def test_retained_crop_recovery_is_private(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            quarantine = root / "recovery.tmp"
            target = root / "crop.png"
            target.write_bytes(b"unknown target")
            previous_umask = os.umask(0o022)
            try:
                with self.assertRaisesRegex(RuntimeError, "retained for recovery"):
                    restore_unlinked_crop_content(quarantine, target, b"private pixels", "Crop")
            finally:
                os.umask(previous_umask)
            self.assertEqual(stat.S_IMODE(quarantine.stat().st_mode), 0o600)

    def test_publishes_readable_batch_with_plain_crop_paths_and_exact_columns(self):
        with self.prepared_root("PXL_230150462.jpg") as root:

            status, summary, _stderr = self.run_managed_batch_and_capture_output(root, ManagedPipeline())

            self.assertEqual(status, 0)
            batch = Path(summary["batch_directory"])
            self.assertFalse((batch / PROGRESS_FILE_NAME).exists())
            self.assertTrue((batch / DETAILS_FILE_NAME).is_file())
            crop = batch / BOX_IMAGES_DIRECTORY_NAME / "PXL_230150462 - Box 01.png"
            self.assertTrue(crop.is_file())
            with (batch / CATALOGUE_FILE_NAME).open(newline="", encoding="utf-8") as source:
                reader = csv.DictReader(source)
                rows = list(reader)
            self.assertEqual(reader.fieldnames, OUTPUT_COLUMNS)
            self.assertEqual(
                rows[0]["Cropped Box Image Link"],
                "Box Images/PXL_230150462 - Box 01.png",
            )
            details = json.loads((batch / DETAILS_FILE_NAME).read_text(encoding="utf-8"))
            self.assertEqual(details["inputs"][0]["original_filename"], "PXL_230150462.jpg")

    def test_full_image_fallback_uses_the_readable_full_photo_name(self):
        with self.prepared_root("Shelf.jpg") as root:
            status, summary, _stderr = self.run_managed_batch_and_capture_output(root, ManagedPipeline(full_image=True))
            self.assertEqual(status, 0)
            evidence = Path(summary["batch_directory"]) / BOX_IMAGES_DIRECTORY_NAME / "Shelf - Full Photo.png"
            self.assertTrue(evidence.is_file())

    def test_interrupt_resumes_committed_photos_and_reprocesses_only_the_active_photo(self):
        with self.prepared_root("01.jpg", "02.jpg", "03.jpg") as root:
            interrupted = ManagedPipeline(interrupt_name="02.jpg", port=12000)

            status, summary, _stderr = self.run_managed_batch_and_capture_output(root, interrupted)

            self.assertEqual(status, 130)
            self.assertEqual(interrupted.calls, ["01.jpg", "02.jpg"])
            self.assertEqual(summary["images_processed"], 1)
            batch = Path(summary["batch_directory"])
            self.assertTrue((batch / PROGRESS_FILE_NAME).is_file())
            self.assertFalse((batch / CATALOGUE_FILE_NAME).exists())

            resumed = ManagedPipeline(port=13000)
            status, summary, _stderr = self.run_managed_batch_and_capture_output(root, resumed)

            self.assertEqual(status, 0)
            self.assertEqual(resumed.calls, ["02.jpg", "03.jpg"])
            self.assertEqual(summary["images_reused"], 1)
            self.assertEqual(summary["images_processed_this_start"], 2)
            details = json.loads(Path(summary["run_manifest"]).read_text(encoding="utf-8"))
            self.assertEqual(
                details["endpoint_history"],
                ["http://127.0.0.1:12000", "http://127.0.0.1:13000"],
            )

    def test_resumed_human_report_uses_final_counts(self):
        with self.prepared_root("01.jpg", "02.jpg") as root:
            self.run_managed_batch_and_capture_output(root, ManagedPipeline(interrupt_name="02.jpg"))
            arguments = managed_arguments(root)
            arguments.json = False
            images = sorted(arguments.path_to_inputs.glob("*.jpg"))
            stdout, stderr = io.StringIO(), io.StringIO()
            with (
                patch("uma_bct.managed_batch.build_pipeline", return_value=ManagedPipeline()),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                status = run_managed_batch(arguments, images, arguments.results_directory)

            self.assertEqual(status, 0)
            self.assertIn("Resuming: 1 photo already finished", stderr.getvalue())
            self.assertIn("Processing Photo 2 of 2", stderr.getvalue())
            self.assertIn("Finished\nPhotos finished: 2 of 2", stdout.getvalue())
            self.assertIn("Catalogue: 2 rows", stdout.getvalue())
            self.assertNotIn('"schema_version"', stdout.getvalue())

    def test_changed_input_blocks_resume_before_pipeline_preflight(self):
        with self.prepared_root("01.jpg", "02.jpg") as root:
            self.run_managed_batch_and_capture_output(root, ManagedPipeline(interrupt_name="01.jpg"))
            (root / "Add Photos Here/02.jpg").write_bytes(b"changed")
            builder = Mock()
            arguments = managed_arguments(root)
            images = sorted(arguments.path_to_inputs.glob("*.jpg"))
            stdout, stderr = io.StringIO(), io.StringIO()

            with (
                patch("uma_bct.managed_batch.build_pipeline", builder),
                contextlib.redirect_stdout(stdout),
                contextlib.redirect_stderr(stderr),
            ):
                status = run_managed_batch(arguments, images, arguments.results_directory)

            self.assertEqual(status, 2)
            self.assertIn("byte-changed", stderr.getvalue())
            builder.assert_not_called()

    def test_source_changed_after_its_turn_blocks_final_publication(self):
        class LateChangingPipeline(ManagedPipeline):
            def process(self, path: Path) -> ImageResult:
                result = super().process(path)
                if path.name == "02.jpg":
                    path.with_name("01.jpg").write_bytes(b"changed after first photo committed")
                return result

        with self.prepared_root("01.jpg", "02.jpg") as root:

            status, summary, stderr = self.run_managed_batch_and_capture_output(root, LateChangingPipeline())

            self.assertEqual(status, 2)
            self.assertIn("Input photos changed before publication", stderr)
            self.assertFalse((Path(summary["batch_directory"]) / CATALOGUE_FILE_NAME).exists())

    def test_changed_model_digest_blocks_resume_but_a_new_loopback_port_does_not(self):
        with self.prepared_root("01.jpg") as root:
            self.run_managed_batch_and_capture_output(root, ManagedPipeline(interrupt_name="01.jpg", port=12000))

            status, _summary, stderr = self.run_managed_batch_and_capture_output(
                root, ManagedPipeline(port=13000, digest="sha256:different")
            )

            self.assertEqual(status, 2)
            self.assertIn("model digest changed", stderr)

    def test_relative_batch_path_resumes_through_the_equivalent_absolute_path(self):
        with self.prepared_root("01.jpg") as root:
            arguments = managed_arguments(root)
            arguments.path_to_inputs = Path("Add Photos Here")
            arguments.results_directory = Path("Results")
            with (
                contextlib.chdir(root),
                patch(
                    "uma_bct.managed_batch.build_pipeline",
                    return_value=ManagedPipeline(interrupt_name="01.jpg"),
                ),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                first_status = run_managed_batch(
                    arguments, [Path("Add Photos Here/01.jpg")], arguments.results_directory
                )
            self.assertEqual(first_status, 130)

            resumed = ManagedPipeline(port=13000)
            status, summary, _stderr = self.run_managed_batch_and_capture_output(root, resumed)

            self.assertEqual(status, 0)
            self.assertTrue(summary["resumed"])
            self.assertEqual(resumed.calls, ["01.jpg"])

    def test_completed_snapshot_is_not_repeated_without_reprocess(self):
        with self.prepared_root("01.jpg") as root:
            first = ManagedPipeline()
            self.run_managed_batch_and_capture_output(root, first)
            second = ManagedPipeline()

            status, summary, _stderr = self.run_managed_batch_and_capture_output(root, second)

            self.assertEqual(status, 0)
            self.assertTrue(summary["already_complete"])
            self.assertEqual(second.calls, [])
            self.assertEqual(len(list((root / "Results").glob("Batch *"))), 1)

            status, summary, _stderr = self.run_managed_batch_and_capture_output(root, ManagedPipeline(), reprocess=True)
            self.assertEqual(status, 0)
            self.assertEqual(len(list((root / "Results").glob("Batch *"))), 2)

    def test_completed_reuse_blocks_a_missing_or_linked_catalogue(self):
        for damage in ("missing", "linked"):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                self.prepare_inputs(root, "01.jpg")
                _status, summary, _stderr = self.run_managed_batch_and_capture_output(root, ManagedPipeline())
                catalogue = Path(summary["batch_directory"]) / CATALOGUE_FILE_NAME
                if damage == "missing":
                    catalogue.unlink()
                elif damage == "changed":
                    catalogue.write_bytes(b"changed catalogue")
                else:
                    target = root / "outside.csv"
                    target.write_bytes(catalogue.read_bytes())
                    catalogue.unlink()
                    try:
                        catalogue.symlink_to(target)
                    except OSError as error:
                        self.skipTest(str(error))
                pipeline = ManagedPipeline(port=13000)

                status, _summary, stderr = self.run_managed_batch_and_capture_output(root, pipeline)

                self.assertEqual(status, 2)
                self.assertEqual(pipeline.calls, [])
                self.assertTrue("Published catalogue" in stderr or "symbolic link" in stderr)

    def test_legacy_top_level_results_are_left_byte_for_byte_unchanged(self):
        with self.prepared_root("01.jpg") as root:
            results = root / "Results"
            legacy_crop = results / "crops/run-old/box.png"
            legacy_crop.parent.mkdir(parents=True)
            legacy_crop.write_bytes(b"legacy crop")
            legacy_catalogue = results / "catalogue.csv"
            legacy_catalogue.write_bytes(b"legacy catalogue\r\n")
            legacy_manifest = results / "catalogue.run.json"
            legacy_manifest.write_bytes(b'{"legacy": true}\n')

            status, _summary, _stderr = self.run_managed_batch_and_capture_output(root, ManagedPipeline())

            self.assertEqual(status, 0)
            self.assertEqual(legacy_crop.read_bytes(), b"legacy crop")
            self.assertEqual(legacy_catalogue.read_bytes(), b"legacy catalogue\r\n")
            self.assertEqual(legacy_manifest.read_bytes(), b'{"legacy": true}\n')

    def test_failed_photo_counts_once_and_zero_rows_publish_no_catalogue(self):
        with self.prepared_root("failed.jpg") as root:

            status, summary, _stderr = self.run_managed_batch_and_capture_output(root, ManagedPipeline(failed_names=("failed.jpg",)))

            self.assertEqual(status, 1)
            self.assertEqual(summary["failure_count"], 1)
            self.assertEqual(summary["catalogue_records"], 0)
            self.assertFalse((Path(summary["batch_directory"]) / CATALOGUE_FILE_NAME).exists())

    def test_terminal_partial_batch_publishes_usable_rows_and_counts_only_failed_sources(self):
        with self.prepared_root("good.jpg", "failed.jpg") as root:

            status, summary, _stderr = self.run_managed_batch_and_capture_output(root, ManagedPipeline(failed_names=("failed.jpg",)))

            self.assertEqual(status, 1)
            self.assertEqual(summary["status"], "partial")
            self.assertEqual(summary["catalogue_records"], 1)
            self.assertEqual(summary["failure_count"], 1)
            self.assertTrue((Path(summary["batch_directory"]) / CATALOGUE_FILE_NAME).is_file())

    def test_changed_crop_blocks_reuse(self):
        with self.prepared_root("01.jpg", "02.jpg") as root:
            status, summary, _stderr = self.run_managed_batch_and_capture_output(root, ManagedPipeline(interrupt_name="02.jpg"))
            self.assertEqual(status, 130)
            crop = next((Path(summary["batch_directory"]) / BOX_IMAGES_DIRECTORY_NAME).iterdir())
            crop.write_bytes(b"tampered")

            status, summary, stderr = self.run_managed_batch_and_capture_output(root, ManagedPipeline())

            self.assertEqual(status, 2)
            self.assertIn("changed after it was committed", stderr)

    def test_restart_finishes_a_checkpointed_catalogue_publication_without_model_reuse(self):
        with self.prepared_root("01.jpg") as root:
            pipeline = ManagedPipeline()

            def stop_details_promotion(source: Path, destination: Path, *, existing_message: str):
                if destination.name == DETAILS_FILE_NAME:
                    raise OSError("details promotion stopped")
                return rename_new_file(source, destination, existing_message=existing_message)

            first_status, first_summary, _stderr = self.run_managed_batch_and_capture_output(
                root, pipeline,
                patches=(patch("uma_bct.managed_batch.rename_new_file", side_effect=stop_details_promotion),),
            )

            self.assertEqual(first_status, 2)
            batch = Path(first_summary["batch_directory"])
            self.assertTrue((batch / CATALOGUE_FILE_NAME).is_file())
            self.assertTrue((batch / PROGRESS_FILE_NAME).is_file())

            resumed = ManagedPipeline(port=13000)
            status, summary, _stderr = self.run_managed_batch_and_capture_output(root, resumed)

            self.assertEqual(status, 1)
            self.assertEqual(resumed.calls, [])
            self.assertEqual(summary["images_reused"], 1)
            self.assertTrue((batch / DETAILS_FILE_NAME).is_file())
            self.assertFalse((batch / PROGRESS_FILE_NAME).exists())

    def test_interrupt_after_details_rename_does_not_recreate_progress(self):
        with self.prepared_root("01.jpg") as root:

            def interrupt_after_rename(source: Path, destination: Path, *, existing_message: str):
                rename_new_file(source, destination, existing_message=existing_message)
                if destination.name == DETAILS_FILE_NAME:
                    raise KeyboardInterrupt

            status, first_summary, _stderr = self.run_managed_batch_and_capture_output(
                root, ManagedPipeline(),
                patches=(patch("uma_bct.managed_batch.rename_new_file", side_effect=interrupt_after_rename),),
            )

            self.assertEqual(status, 130)
            batch = Path(first_summary["batch_directory"])
            self.assertTrue((batch / DETAILS_FILE_NAME).is_file())
            self.assertFalse((batch / PROGRESS_FILE_NAME).exists())

            resumed = ManagedPipeline(port=13000)
            status, summary, _stderr = self.run_managed_batch_and_capture_output(root, resumed)
            self.assertEqual(status, 0)
            self.assertTrue(summary["already_complete"])
            self.assertEqual(resumed.calls, [])

    def test_crop_publication_stop_removes_only_the_checkpointed_orphan_and_reprocesses_photo(self):
        with self.prepared_root("01.jpg") as root:
            pipeline = ManagedPipeline()
            stopped = False

            def stop_after_crop_move(source: Path, destination: Path, evidence: dict[str, object]):
                nonlocal stopped
                publish_reserved_crop(source, destination, evidence)
                if not stopped:
                    stopped = True
                    raise OSError("crop publication stopped")

            status, first_summary, _stderr = self.run_managed_batch_and_capture_output(
                root, pipeline,
                patches=(
                    patch("uma_bct.managed_batch.publish_reserved_crop", side_effect=stop_after_crop_move),
                ),
            )
            self.assertEqual(status, 2)
            batch = Path(first_summary["batch_directory"])
            self.assertEqual(len(list((batch / BOX_IMAGES_DIRECTORY_NAME).iterdir())), 1)

            resumed = ManagedPipeline(port=13000)
            status, summary, _stderr = self.run_managed_batch_and_capture_output(root, resumed)

            self.assertEqual(status, 1)
            self.assertEqual(resumed.calls, ["01.jpg"])
            self.assertEqual(len(list((batch / BOX_IMAGES_DIRECTORY_NAME).iterdir())), 1)
            self.assertTrue((batch / DETAILS_FILE_NAME).is_file())

    def test_crop_changed_at_resume_cleanup_syscall_is_restored_and_blocked(self):
        with self.prepared_root("01.jpg") as root:

            def stop_after_crop(source: Path, destination: Path, evidence: dict[str, object]):
                publish_reserved_crop(source, destination, evidence)
                raise OSError("stop after crop")

            first_status, _summary, _stderr = self.run_managed_batch_and_capture_output(
                root, ManagedPipeline(),
                patches=(patch("uma_bct.managed_batch.publish_reserved_crop", side_effect=stop_after_crop),),
            )
            self.assertEqual(first_status, 2)

            changed = False

            def edit_after_cleanup_check(path: Path, purpose: str):
                nonlocal changed
                result = fingerprint_file(path, purpose)
                if purpose == "Interrupted crop publication" and not changed:
                    path.write_bytes(b"late unknown cleanup edit")
                    changed = True
                return result

            with patch("uma_bct.managed_batch.fingerprint_file", side_effect=edit_after_cleanup_check):
                status, summary, stderr = self.run_managed_batch_and_capture_output(root, ManagedPipeline(port=13000))
            target = Path(summary["batch_directory"]) / BOX_IMAGES_DIRECTORY_NAME / "01 - Box 01.png"

            self.assertTrue(changed)
            self.assertEqual(status, 2)
            self.assertIn("Unknown interrupted crop publication content", stderr)
            self.assertEqual(target.read_bytes(), b"late unknown cleanup edit")

    def test_same_byte_preexisting_crop_collision_is_never_deleted_as_an_orphan(self):
        class CollidingPipeline(ManagedPipeline):
            def process(self, path: Path) -> ImageResult:
                result = super().process(path)
                batch = self.work_directory.parent.parent
                collision = batch / BOX_IMAGES_DIRECTORY_NAME / "01 - Box 01.png"
                collision.write_bytes(b"crop for 01.jpg")
                return result

        with self.prepared_root("01.jpg") as root:
            status, summary, stderr = self.run_managed_batch_and_capture_output(root, CollidingPipeline())
            batch = Path(summary["batch_directory"])
            collision = batch / BOX_IMAGES_DIRECTORY_NAME / "01 - Box 01.png"

            self.assertEqual(status, 2)
            self.assertIn("was not overwritten", stderr)
            self.assertEqual(summary["catalogue_records"], 0)
            self.assertIsNone(summary["catalogue_output"])
            self.assertEqual(collision.read_bytes(), b"crop for 01.jpg")

            status, failed_summary, stderr = self.run_managed_batch_and_capture_output(root, ManagedPipeline(port=13000))
            self.assertEqual(status, 2)
            self.assertIn("Unexpected file", stderr)
            self.assertIsNone(failed_summary["catalogue_output"])
            self.assertEqual(collision.read_bytes(), b"crop for 01.jpg")

    def test_crop_changed_at_publication_syscall_is_not_overwritten(self):
        with self.prepared_root("01.jpg") as root:
            reservation_checks = 0

            def edit_after_reservation_displacement(path: Path, purpose: str):
                nonlocal reservation_checks
                result = fingerprint_file(path, purpose)
                if purpose == "Crop-name reservation":
                    reservation_checks += 1
                    if reservation_checks == 2:
                        batch = path.parents[3]
                        target = batch / BOX_IMAGES_DIRECTORY_NAME / "01 - Box 01.png"
                        target.write_bytes(b"late unknown crop edit")
                return result

            with patch(
                "uma_bct.managed_batch.fingerprint_file", side_effect=edit_after_reservation_displacement
            ):
                status, summary, stderr = self.run_managed_batch_and_capture_output(root, ManagedPipeline())
            target = Path(summary["batch_directory"]) / BOX_IMAGES_DIRECTORY_NAME / "01 - Box 01.png"

            self.assertEqual(status, 2)
            self.assertIn("appeared during publication", stderr)
            self.assertEqual(target.read_bytes(), b"late unknown crop edit")

            status, _summary, stderr = self.run_managed_batch_and_capture_output(root, ManagedPipeline(port=13000))
            self.assertEqual(status, 2)
            self.assertIn("Unknown interrupted crop publication content", stderr)
            self.assertEqual(target.read_bytes(), b"late unknown crop edit")

    def test_mismatched_published_catalogue_is_blocked_without_overwrite(self):
        with self.prepared_root("01.jpg") as root:

            def stop_details(source: Path, destination: Path, *, existing_message: str):
                if destination.name == DETAILS_FILE_NAME:
                    raise OSError("stop before details")
                return rename_new_file(source, destination, existing_message=existing_message)

            _status, first_summary, _stderr = self.run_managed_batch_and_capture_output(
                root, ManagedPipeline(),
                patches=(patch("uma_bct.managed_batch.rename_new_file", side_effect=stop_details),),
            )
            batch = Path(first_summary["batch_directory"])
            catalogue = batch / CATALOGUE_FILE_NAME
            catalogue.write_bytes(b"unknown replacement")

            status, failed_summary, stderr = self.run_managed_batch_and_capture_output(root, ManagedPipeline(port=13000))

            self.assertEqual(status, 2)
            self.assertIn("does not match its checkpoint", stderr)
            self.assertIsNone(failed_summary["catalogue_output"])
            self.assertEqual(catalogue.read_bytes(), b"unknown replacement")

    def test_lock_is_os_backed_and_malformed_stale_lock_is_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory).resolve() / "Results"
            first = ResultsRootLock(results)
            first.acquire()
            try:
                with self.assertRaisesRegex(RuntimeError, "already using"):
                    ResultsRootLock(results).acquire()
            finally:
                first.release()

            recovered = ResultsRootLock(results)
            recovered.acquire()
            recovered.release()

            lock_path = results / ".uma-bct.lock"
            lock_path.write_text("malformed", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "malformed"):
                ResultsRootLock(results).acquire()

    def test_distinct_sibling_results_roots_can_be_locked_concurrently(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory).resolve()
            first = ResultsRootLock(parent / "Results One")
            second = ResultsRootLock(parent / "Results Two")
            first.acquire()
            try:
                second.acquire()
                second.release()
            finally:
                first.release()

    @unittest.skipUnless(sys.platform == "darwin", "macOS filesystem alias fixture")
    def test_case_aliases_of_one_macos_results_root_share_the_same_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory).resolve()
            canonical = parent / "ResultsCase"
            first = ResultsRootLock(canonical)
            first.acquire()
            alias = parent / "resultscase"
            try:
                if not alias.exists() or not os.path.samefile(canonical, alias):
                    self.skipTest("temporary filesystem is case-sensitive")
                with self.assertRaisesRegex(RuntimeError, "already using"):
                    ResultsRootLock(alias).acquire()
            finally:
                first.release()

    @unittest.skipIf(os.name == "nt", "POSIX lock replacement fixture")
    def test_replacing_a_held_lock_or_results_root_cannot_create_a_second_holder(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory).resolve()
            results = parent / "Results"
            first = ResultsRootLock(results)
            first.acquire()
            lock_path = results / ".uma-bct.lock"
            original_lock = results / "original.lock"
            lock_path.rename(original_lock)
            lock_path.write_text('{"schema_version": 1, "pid": 999}\n', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "already using"):
                ResultsRootLock(results).acquire()
            with self.assertRaisesRegex(RuntimeError, "changed"):
                first.release()

        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory).resolve() / "Results"
            first = ResultsRootLock(results)
            first.acquire()
            guard = first._guard_identity[0]
            guard.rename(guard.with_name("original.guard"))
            guard.write_bytes(b"\0")
            with self.assertRaisesRegex(RuntimeError, "already using"):
                ResultsRootLock(results).acquire()
            with self.assertRaisesRegex(RuntimeError, "changed"):
                first.release()

        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory).resolve()
            results = parent / "Results"
            moved = parent / "Results moved"
            first = ResultsRootLock(results)
            first.acquire()
            results.rename(moved)
            results.mkdir()
            with self.assertRaisesRegex(RuntimeError, "already using"):
                ResultsRootLock(results).acquire()
            with self.assertRaisesRegex(RuntimeError, "changed"):
                first.release()

    def test_linked_results_lock_is_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory).resolve() / "Results"
            results.mkdir()
            target = results / "target.lock"
            target.write_text('{"schema_version": 1, "pid": 1}\n', encoding="utf-8")
            lock_path = results / ".uma-bct.lock"
            try:
                lock_path.symlink_to(target)
            except OSError as error:
                self.skipTest(str(error))
            with self.assertRaisesRegex((RuntimeError, ValueError), "symbolic link"):
                ResultsRootLock(results).acquire()

    def test_dangling_progress_link_blocks_batch_selection_before_model_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            self.prepare_inputs(root, "01.jpg")
            batch = root / "Results/Batch 2026-09-11 14-30-05"
            (batch / BOX_IMAGES_DIRECTORY_NAME).mkdir(parents=True)
            try:
                (batch / PROGRESS_FILE_NAME).symlink_to(batch / "missing.json")
            except OSError as error:
                self.skipTest(str(error))
            pipeline = ManagedPipeline()

            status, _summary, stderr = self.run_managed_batch_and_capture_output(root, pipeline)

            self.assertEqual(status, 2)
            self.assertIn("symbolic link", stderr)
            self.assertEqual(pipeline.calls, [])
            self.assertEqual(len(list((root / "Results").glob("Batch *"))), 1)

    def test_zero_row_batch_does_not_finalize_over_a_dangling_catalogue_link(self):
        class DanglingCataloguePipeline(ManagedPipeline):
            def process(self, path: Path) -> ImageResult:
                batch = self.work_directory.parent.parent
                (batch / CATALOGUE_FILE_NAME).symlink_to(batch / "missing-catalogue.csv")
                return super().process(path)

        with self.prepared_root("failed.jpg") as root:
            pipeline = DanglingCataloguePipeline(failed_names=("failed.jpg",))

            status, summary, stderr = self.run_managed_batch_and_capture_output(root, pipeline)

            batch = Path(summary["batch_directory"])
            self.assertEqual(status, 2)
            self.assertIn("unexpected catalogue", stderr)
            self.assertTrue((batch / CATALOGUE_FILE_NAME).is_symlink())
            self.assertTrue((batch / PROGRESS_FILE_NAME).is_file())
            self.assertFalse((batch / DETAILS_FILE_NAME).exists())

    def test_zero_row_batch_preserves_and_blocks_an_unknown_pending_catalogue(self):
        class PendingCataloguePipeline(ManagedPipeline):
            def process(self, path: Path) -> ImageResult:
                batch = self.work_directory.parent.parent
                (batch / ".uma-bct-catalogue-pending.csv").write_bytes(b"unknown pending output")
                return super().process(path)

        with self.prepared_root("failed.jpg") as root:
            status, summary, stderr = self.run_managed_batch_and_capture_output(
                root, PendingCataloguePipeline(failed_names=("failed.jpg",))
            )
            batch = Path(summary["batch_directory"])

            self.assertEqual(status, 2)
            self.assertIn("unexpected pending catalogue", stderr)
            self.assertEqual(
                (batch / ".uma-bct-catalogue-pending.csv").read_bytes(), b"unknown pending output"
            )
            self.assertFalse((batch / DETAILS_FILE_NAME).exists())

    def test_naming_is_portable_readable_and_batch_collisions_are_numbered(self):
        self.assertEqual(sanitize_source_name("CON.jpg"), "Photo CON")
        self.assertEqual(sanitize_source_name('  a<>:"/\\|?* b.jpg'), "a- b")
        names = [
            {"original_filename": "Shelf.jpg"},
            {"original_filename": "shelf.PNG"},
            {"original_filename": "100% #1.jpg"},
        ]
        allocate_source_names(names)
        self.assertEqual(
            [item["readable_source_name"] for item in names],
            ["Shelf", "shelf (2)", "100- -1"],
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            moment = datetime(2026, 9, 11, 14, 30, 5)
            first = create_batch_directory(root, moment)
            second = create_batch_directory(root, moment)
            self.assertEqual(second.name, f"{first.name} (2)")

    def test_completed_batch_links_survive_moving_the_whole_folder(self):
        with self.prepared_root("Shelf Photo.jpg") as root:
            status, summary, _stderr = self.run_managed_batch_and_capture_output(root, ManagedPipeline())
            self.assertEqual(status, 0)
            original = Path(summary["batch_directory"])
            moved = root / "Moved Completed Batch"
            original.rename(moved)

            with (moved / CATALOGUE_FILE_NAME).open(newline="", encoding="utf-8") as source:
                row = next(csv.DictReader(source))

            self.assertEqual(row["Cropped Box Image Link"], "Box Images/Shelf Photo - Box 01.png")
            self.assertTrue((moved / row["Cropped Box Image Link"]).is_file())

    def test_serialize_results(self):
        batch_dir=Path("/test")

        model_metadata = {
            "model": "qwen-test",
            "model_digest": "test",
            "ollama_host": "test",
        }

        image = make_image(Path("/test/test.jpg"))
        box = make_box(image=image, box_id="1", crop_path=Path("/test/test.jpg"))
        date = Date(day=10,month=1,year=2005)
        date_range = DateRange(Date(day=10,month=12,year=1901),Date(day=10,month=5,year=1920))

        results = [
            ImageResult(
                image, "qwen", "qwen", ("qwen",), "qwen", "failed", (), (), model_metadata, date_detector="none"
            ),
            ImageResult(
                image, "qwen", "qwen", ("qwen",), "qwen", "complete", (BoxResult(box, make_transcription(box)),), (), model_metadata, date_detector="none"
            ),
            ImageResult(
                image, "qwen", "qwen", ("qwen",), "qwen", "complete", (BoxResult(box, make_transcription(box, dates=(date, date_range))),), (), model_metadata, date_detector="none"
            ),
        ]

        for result in results:
            self.assertEqual(result, deserialize_result(serialize_result(result, batch_dir), batch_dir))


    def test_legacy_html_metadata_does_not_block_offline_publication(self):
        for phase in ("staging", "publishing", "finalizing"):
            with self.subTest(phase=phase), self.prepared_root("01.jpg") as root:
                def interrupt(path, progress, phase=phase):
                    from uma_bct.managed_batch import write_progress
                    write_progress(path, progress)
                    if progress.get("phase") == phase:
                        raise KeyboardInterrupt

                _, summary, _ = self.run_managed_batch_and_capture_output(
                    root, ManagedPipeline(), patches=(patch("uma_bct.managed_batch.write_progress", side_effect=interrupt),),
                )
                batch = Path(summary["batch_directory"])
                progress_path = batch / PROGRESS_FILE_NAME
                progress = json.loads(progress_path.read_text())
                progress["html_review"] = True
                progress["publication"]["review"] = {"sha256": "former-html-digest"}
                progress["review_output"] = str(batch / "Review.html")
                progress_path.write_text(json.dumps(progress))
                with patch("uma_bct.managed_batch.build_pipeline", side_effect=AssertionError("must remain offline")):
                    status, resumed, stderr = self.run_managed_batch_and_capture_output(root, ManagedPipeline())
                self.assertEqual(status, 0, stderr)
                self.assertNotIn("review_output", resumed)
                self.assertFalse((batch / "Review.html").exists())
                details = json.loads((batch / DETAILS_FILE_NAME).read_text())
                self.assertNotIn("review", details["publication"])
                self.assertNotIn("html_review", details)

    def test_completed_legacy_html_batch_preserves_metadata_and_reviewer_edits(self):
        with self.prepared_root("01.jpg") as root:
            _, summary, _ = self.run_managed_batch_and_capture_output(root, ManagedPipeline())
            batch = Path(summary["batch_directory"])
            details_path = batch / DETAILS_FILE_NAME
            details = json.loads(details_path.read_text())
            details["html_review"] = True
            details["publication"]["review"] = {"sha256": "former-html-digest"}
            details["review_output"] = str(batch / "Review.html")
            details_path.write_text(json.dumps(details))
            original = details_path.read_bytes()
            catalogue = Path(summary["catalogue_output"])
            catalogue.write_text("reviewer corrected catalogue")
            with patch("uma_bct.managed_batch.build_pipeline", side_effect=AssertionError("must remain offline")):
                status, saved, stderr = self.run_managed_batch_and_capture_output(root, ManagedPipeline())
            self.assertEqual(status, 0, stderr)
            self.assertTrue(saved["already_complete"])
            self.assertTrue(saved["catalogue_modified"])
            self.assertNotIn("review_output", saved)
            self.assertNotIn("review_outputs", saved)
            self.assertEqual(details_path.read_bytes(), original)
            self.assertEqual(catalogue.read_text(), "reviewer corrected catalogue")


if __name__ == "__main__":
    unittest.main()
