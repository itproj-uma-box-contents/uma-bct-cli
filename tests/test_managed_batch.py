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
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

from helpers import make_box, make_image, make_transcription

from uma_bct.atomic import open_stable_file, rename_new_file
from uma_bct.constants import OUTPUT_COLUMNS
from uma_bct.domain import BoxResult
from uma_bct.managed_batch import (
    BOX_IMAGES_DIRECTORY_NAME,
    CATALOGUE_FILE_NAME,
    DETAILS_FILE_NAME,
    PROGRESS_FILE_NAME,
    ManagedInterruption,
    ResultsRootLock,
    allocate_source_names,
    create_batch_directory,
    fingerprint_file,
    publish_reserved_crop,
    read_json_file,
    restore_unlinked_crop_content,
    run_managed_batch,
    sanitize_source_name,
)
from uma_bct.pipeline import ImageResult


# Handle the managed arguments step.
def managed_arguments(root: Path, *, reprocess: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        detector="qwen",
        transcriber="qwen",
        model="qwen-test",
        min_transcription_confidence=80,
        min_detector_confidence=0.6,
        request_timeout=10.0,
        path_to_inputs=root / "Add Photos Here",
        path_to_outputs=root / "unused.csv",
        path_to_box_images=None,
        force=False,
        results_directory=root / "Results",
        reprocess=reprocess,
    )


# Keep the managed pipeline behavior together.
class ManagedPipeline:
    # Store the state this object needs.
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

    # Process the current input.
    def process(self, path: Path) -> ImageResult:
        self.calls.append(path.name)
        if path.name == self.interrupt_name:
            raise KeyboardInterrupt
        image = make_image(path)
        if path.name in self.failed_names:
            return ImageResult(
                image, "qwen", "qwen", ("qwen",), "qwen", "none", "failed", (), (), self.model_metadata
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
            "none",
            "complete",
            (BoxResult(box, make_transcription(box)),),
            (),
            self.model_metadata,
        )


# Group the managed batch checks.
class ManagedBatchTest(unittest.TestCase):
    # Check that managed interruption preserves its exit code.
    def test_managed_interruption_preserves_its_exit_code(self):
        self.assertEqual(ManagedInterruption(143).exit_code, 143)

    # Run the with.
    def run_with(
        self, root: Path, pipeline: ManagedPipeline, *, reprocess: bool = False, patches=()
    ):
        arguments = managed_arguments(root, reprocess=reprocess)
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

    # Prepare the inputs.
    def prepare_inputs(self, root: Path, *names: str) -> None:
        inputs = root / "Add Photos Here"
        inputs.mkdir()
        for name in names:
            (inputs / name).write_bytes(f"photo {name}".encode())

    # Handle the prepared root step.
    @contextlib.contextmanager
    def prepared_root(self, *names: str):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            self.prepare_inputs(root, *names)
            yield root

    # Check that stable read rejects a short read hidden by restored metadata.
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

    # Check that JSON read stays bounded if a small file grows.
    def test_json_read_stays_bounded_if_a_small_file_grows(self):
        metadata = Mock(st_size=1)

        # Handle the growing file step.
        @contextlib.contextmanager
        def growing_file(*_args, **_kwargs):
            yield io.BytesIO(b"x" * (32 * 1024 * 1024 + 1)), metadata

        with (
            patch("uma_bct.managed_batch.open_stable_file", side_effect=growing_file),
            self.assertRaisesRegex(RuntimeError, "unexpectedly large"),
        ):
            read_json_file(Path("progress.json"), "Run progress")

    # Check that retained crop recovery is private.
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

    # Check that the code publishes readable batch with formula links and exact columns.
    def test_publishes_readable_batch_with_formula_links_and_exact_columns(self):
        with self.prepared_root("PXL_230150462.jpg") as root:

            status, summary, _stderr = self.run_with(root, ManagedPipeline())

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
                '=HYPERLINK("Box Images/PXL_230150462 - Box 01.png","PXL_230150462 - Box 01.png")',
            )
            details = json.loads((batch / DETAILS_FILE_NAME).read_text(encoding="utf-8"))
            self.assertEqual(details["inputs"][0]["original_filename"], "PXL_230150462.jpg")

    # Check that full image fallback uses the readable full photo name.
    def test_full_image_fallback_uses_the_readable_full_photo_name(self):
        with self.prepared_root("Shelf.jpg") as root:
            status, summary, _stderr = self.run_with(root, ManagedPipeline(full_image=True))
            self.assertEqual(status, 0)
            evidence = Path(summary["batch_directory"]) / BOX_IMAGES_DIRECTORY_NAME / "Shelf - Full Photo.png"
            self.assertTrue(evidence.is_file())

    # Check that interrupt resumes committed photos and reprocesses only the active photo.
    def test_interrupt_resumes_committed_photos_and_reprocesses_only_the_active_photo(self):
        with self.prepared_root("01.jpg", "02.jpg", "03.jpg") as root:
            interrupted = ManagedPipeline(interrupt_name="02.jpg", port=12000)

            status, summary, _stderr = self.run_with(root, interrupted)

            self.assertEqual(status, 130)
            self.assertEqual(interrupted.calls, ["01.jpg", "02.jpg"])
            self.assertEqual(summary["images_processed"], 1)
            batch = Path(summary["batch_directory"])
            self.assertTrue((batch / PROGRESS_FILE_NAME).is_file())
            self.assertFalse((batch / CATALOGUE_FILE_NAME).exists())

            resumed = ManagedPipeline(port=13000)
            status, summary, _stderr = self.run_with(root, resumed)

            self.assertEqual(status, 0)
            self.assertEqual(resumed.calls, ["02.jpg", "03.jpg"])
            self.assertEqual(summary["images_reused"], 1)
            self.assertEqual(summary["images_processed_this_start"], 2)
            details = json.loads(Path(summary["run_manifest"]).read_text(encoding="utf-8"))
            self.assertEqual(
                details["endpoint_history"],
                ["http://127.0.0.1:12000", "http://127.0.0.1:13000"],
            )

    # Check that changed input blocks resume before pipeline preflight.
    def test_changed_input_blocks_resume_before_pipeline_preflight(self):
        with self.prepared_root("01.jpg", "02.jpg") as root:
            self.run_with(root, ManagedPipeline(interrupt_name="01.jpg"))
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

    # Check that source changed after its turn blocks final publication.
    def test_source_changed_after_its_turn_blocks_final_publication(self):
        # Keep the late changing pipeline behavior together.
        class LateChangingPipeline(ManagedPipeline):
            # Process the current input.
            def process(self, path: Path) -> ImageResult:
                result = super().process(path)
                if path.name == "02.jpg":
                    path.with_name("01.jpg").write_bytes(b"changed after first photo committed")
                return result

        with self.prepared_root("01.jpg", "02.jpg") as root:

            status, summary, stderr = self.run_with(root, LateChangingPipeline())

            self.assertEqual(status, 2)
            self.assertIn("Input photos changed before publication", stderr)
            self.assertFalse((Path(summary["batch_directory"]) / CATALOGUE_FILE_NAME).exists())

    # Check that changed model digest blocks resume but a new loopback port does not.
    def test_changed_model_digest_blocks_resume_but_a_new_loopback_port_does_not(self):
        with self.prepared_root("01.jpg") as root:
            self.run_with(root, ManagedPipeline(interrupt_name="01.jpg", port=12000))

            status, _summary, stderr = self.run_with(
                root, ManagedPipeline(port=13000, digest="sha256:different")
            )

            self.assertEqual(status, 2)
            self.assertIn("model digest changed", stderr)

    # Check that relative batch path resumes through the equivalent absolute path.
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
            status, summary, _stderr = self.run_with(root, resumed)

            self.assertEqual(status, 0)
            self.assertTrue(summary["resumed"])
            self.assertEqual(resumed.calls, ["01.jpg"])

    # Check that completed snapshot is not repeated without reprocess.
    def test_completed_snapshot_is_not_repeated_without_reprocess(self):
        with self.prepared_root("01.jpg") as root:
            first = ManagedPipeline()
            self.run_with(root, first)
            second = ManagedPipeline()

            status, summary, _stderr = self.run_with(root, second)

            self.assertEqual(status, 0)
            self.assertTrue(summary["already_complete"])
            self.assertEqual(second.calls, [])
            self.assertEqual(len(list((root / "Results").glob("Batch *"))), 1)

            status, summary, _stderr = self.run_with(root, ManagedPipeline(), reprocess=True)
            self.assertEqual(status, 0)
            self.assertEqual(len(list((root / "Results").glob("Batch *"))), 2)

    # Check that completed reuse blocks a missing changed or linked catalogue.
    def test_completed_reuse_blocks_a_missing_changed_or_linked_catalogue(self):
        for damage in ("missing", "changed", "linked"):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                self.prepare_inputs(root, "01.jpg")
                _status, summary, _stderr = self.run_with(root, ManagedPipeline())
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

                status, _summary, stderr = self.run_with(root, pipeline)

                self.assertEqual(status, 2)
                self.assertEqual(pipeline.calls, [])
                self.assertTrue("Published catalogue" in stderr or "symbolic link" in stderr)

    # Check that legacy top level results are left byte for byte unchanged.
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

            status, _summary, _stderr = self.run_with(root, ManagedPipeline())

            self.assertEqual(status, 0)
            self.assertEqual(legacy_crop.read_bytes(), b"legacy crop")
            self.assertEqual(legacy_catalogue.read_bytes(), b"legacy catalogue\r\n")
            self.assertEqual(legacy_manifest.read_bytes(), b'{"legacy": true}\n')

    # Check that failed photo counts once and zero rows publish no catalogue.
    def test_failed_photo_counts_once_and_zero_rows_publish_no_catalogue(self):
        with self.prepared_root("failed.jpg") as root:

            status, summary, _stderr = self.run_with(root, ManagedPipeline(failed_names=("failed.jpg",)))

            self.assertEqual(status, 1)
            self.assertEqual(summary["failure_count"], 1)
            self.assertEqual(summary["catalogue_records"], 0)
            self.assertFalse((Path(summary["batch_directory"]) / CATALOGUE_FILE_NAME).exists())

    # Check that terminal partial batch publishes usable rows and counts only failed sources.
    def test_terminal_partial_batch_publishes_usable_rows_and_counts_only_failed_sources(self):
        with self.prepared_root("good.jpg", "failed.jpg") as root:

            status, summary, _stderr = self.run_with(root, ManagedPipeline(failed_names=("failed.jpg",)))

            self.assertEqual(status, 1)
            self.assertEqual(summary["status"], "partial")
            self.assertEqual(summary["catalogue_records"], 1)
            self.assertEqual(summary["failure_count"], 1)
            self.assertTrue((Path(summary["batch_directory"]) / CATALOGUE_FILE_NAME).is_file())

    # Check that changed crop blocks reuse.
    def test_changed_crop_blocks_reuse(self):
        with self.prepared_root("01.jpg", "02.jpg") as root:
            status, summary, _stderr = self.run_with(root, ManagedPipeline(interrupt_name="02.jpg"))
            self.assertEqual(status, 130)
            crop = next((Path(summary["batch_directory"]) / BOX_IMAGES_DIRECTORY_NAME).iterdir())
            crop.write_bytes(b"tampered")

            status, summary, stderr = self.run_with(root, ManagedPipeline())

            self.assertEqual(status, 2)
            self.assertIn("changed after it was committed", stderr)

    # Check that restart finishes a checkpointed catalogue publication without model reuse.
    def test_restart_finishes_a_checkpointed_catalogue_publication_without_model_reuse(self):
        with self.prepared_root("01.jpg") as root:
            pipeline = ManagedPipeline()

            # Handle the stop details promotion step.
            def stop_details_promotion(source: Path, destination: Path, *, existing_message: str):
                if destination.name == DETAILS_FILE_NAME:
                    raise OSError("details promotion stopped")
                return rename_new_file(source, destination, existing_message=existing_message)

            first_status, first_summary, _stderr = self.run_with(
                root, pipeline,
                patches=(patch("uma_bct.managed_batch.rename_new_file", side_effect=stop_details_promotion),),
            )

            self.assertEqual(first_status, 2)
            batch = Path(first_summary["batch_directory"])
            self.assertTrue((batch / CATALOGUE_FILE_NAME).is_file())
            self.assertTrue((batch / PROGRESS_FILE_NAME).is_file())

            resumed = ManagedPipeline(port=13000)
            status, summary, _stderr = self.run_with(root, resumed)

            self.assertEqual(status, 1)
            self.assertEqual(resumed.calls, [])
            self.assertEqual(summary["images_reused"], 1)
            self.assertTrue((batch / DETAILS_FILE_NAME).is_file())
            self.assertFalse((batch / PROGRESS_FILE_NAME).exists())

    # Check that interrupt after details rename does not recreate progress.
    def test_interrupt_after_details_rename_does_not_recreate_progress(self):
        with self.prepared_root("01.jpg") as root:

            # Handle the interrupt after rename step.
            def interrupt_after_rename(source: Path, destination: Path, *, existing_message: str):
                rename_new_file(source, destination, existing_message=existing_message)
                if destination.name == DETAILS_FILE_NAME:
                    raise KeyboardInterrupt

            status, first_summary, _stderr = self.run_with(
                root, ManagedPipeline(),
                patches=(patch("uma_bct.managed_batch.rename_new_file", side_effect=interrupt_after_rename),),
            )

            self.assertEqual(status, 130)
            batch = Path(first_summary["batch_directory"])
            self.assertTrue((batch / DETAILS_FILE_NAME).is_file())
            self.assertFalse((batch / PROGRESS_FILE_NAME).exists())

            resumed = ManagedPipeline(port=13000)
            status, summary, _stderr = self.run_with(root, resumed)
            self.assertEqual(status, 0)
            self.assertTrue(summary["already_complete"])
            self.assertEqual(resumed.calls, [])

    # Check that crop publication stop removes only the checkpointed orphan and reprocesses photo.
    def test_crop_publication_stop_removes_only_the_checkpointed_orphan_and_reprocesses_photo(self):
        with self.prepared_root("01.jpg") as root:
            pipeline = ManagedPipeline()
            stopped = False

            # Handle the stop after crop move step.
            def stop_after_crop_move(source: Path, destination: Path, evidence: dict[str, object]):
                nonlocal stopped
                publish_reserved_crop(source, destination, evidence)
                if not stopped:
                    stopped = True
                    raise OSError("crop publication stopped")

            status, first_summary, _stderr = self.run_with(
                root, pipeline,
                patches=(
                    patch("uma_bct.managed_batch.publish_reserved_crop", side_effect=stop_after_crop_move),
                ),
            )
            self.assertEqual(status, 2)
            batch = Path(first_summary["batch_directory"])
            self.assertEqual(len(list((batch / BOX_IMAGES_DIRECTORY_NAME).iterdir())), 1)

            resumed = ManagedPipeline(port=13000)
            status, summary, _stderr = self.run_with(root, resumed)

            self.assertEqual(status, 1)
            self.assertEqual(resumed.calls, ["01.jpg"])
            self.assertEqual(len(list((batch / BOX_IMAGES_DIRECTORY_NAME).iterdir())), 1)
            self.assertTrue((batch / DETAILS_FILE_NAME).is_file())

    # Check that crop changed at resume cleanup syscall is restored and blocked.
    def test_crop_changed_at_resume_cleanup_syscall_is_restored_and_blocked(self):
        with self.prepared_root("01.jpg") as root:

            # Handle the stop after crop step.
            def stop_after_crop(source: Path, destination: Path, evidence: dict[str, object]):
                publish_reserved_crop(source, destination, evidence)
                raise OSError("stop after crop")

            first_status, _summary, _stderr = self.run_with(
                root, ManagedPipeline(),
                patches=(patch("uma_bct.managed_batch.publish_reserved_crop", side_effect=stop_after_crop),),
            )
            self.assertEqual(first_status, 2)

            changed = False

            # Handle the edit after cleanup check step.
            def edit_after_cleanup_check(path: Path, purpose: str):
                nonlocal changed
                result = fingerprint_file(path, purpose)
                if purpose == "Interrupted crop publication" and not changed:
                    path.write_bytes(b"late unknown cleanup edit")
                    changed = True
                return result

            with patch("uma_bct.managed_batch.fingerprint_file", side_effect=edit_after_cleanup_check):
                status, summary, stderr = self.run_with(root, ManagedPipeline(port=13000))
            target = Path(summary["batch_directory"]) / BOX_IMAGES_DIRECTORY_NAME / "01 - Box 01.png"

            self.assertTrue(changed)
            self.assertEqual(status, 2)
            self.assertIn("Unknown interrupted crop publication content", stderr)
            self.assertEqual(target.read_bytes(), b"late unknown cleanup edit")

    # Check that same byte preexisting crop collision is never deleted as an orphan.
    def test_same_byte_preexisting_crop_collision_is_never_deleted_as_an_orphan(self):
        # Keep the colliding pipeline behavior together.
        class CollidingPipeline(ManagedPipeline):
            # Process the current input.
            def process(self, path: Path) -> ImageResult:
                result = super().process(path)
                batch = self.work_directory.parent.parent
                collision = batch / BOX_IMAGES_DIRECTORY_NAME / "01 - Box 01.png"
                collision.write_bytes(b"crop for 01.jpg")
                return result

        with self.prepared_root("01.jpg") as root:
            status, summary, stderr = self.run_with(root, CollidingPipeline())
            batch = Path(summary["batch_directory"])
            collision = batch / BOX_IMAGES_DIRECTORY_NAME / "01 - Box 01.png"

            self.assertEqual(status, 2)
            self.assertIn("was not overwritten", stderr)
            self.assertEqual(summary["catalogue_records"], 0)
            self.assertIsNone(summary["catalogue_output"])
            self.assertEqual(collision.read_bytes(), b"crop for 01.jpg")

            status, failed_summary, stderr = self.run_with(root, ManagedPipeline(port=13000))
            self.assertEqual(status, 2)
            self.assertIn("Unexpected file", stderr)
            self.assertIsNone(failed_summary["catalogue_output"])
            self.assertEqual(collision.read_bytes(), b"crop for 01.jpg")

    # Check that crop changed at publication syscall is not overwritten.
    def test_crop_changed_at_publication_syscall_is_not_overwritten(self):
        with self.prepared_root("01.jpg") as root:
            reservation_checks = 0

            # Handle the edit after reservation displacement step.
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
                status, summary, stderr = self.run_with(root, ManagedPipeline())
            target = Path(summary["batch_directory"]) / BOX_IMAGES_DIRECTORY_NAME / "01 - Box 01.png"

            self.assertEqual(status, 2)
            self.assertIn("appeared during publication", stderr)
            self.assertEqual(target.read_bytes(), b"late unknown crop edit")

            status, _summary, stderr = self.run_with(root, ManagedPipeline(port=13000))
            self.assertEqual(status, 2)
            self.assertIn("Unknown interrupted crop publication content", stderr)
            self.assertEqual(target.read_bytes(), b"late unknown crop edit")

    # Check that mismatched published catalogue is blocked without overwrite.
    def test_mismatched_published_catalogue_is_blocked_without_overwrite(self):
        with self.prepared_root("01.jpg") as root:

            # Handle the stop details step.
            def stop_details(source: Path, destination: Path, *, existing_message: str):
                if destination.name == DETAILS_FILE_NAME:
                    raise OSError("stop before details")
                return rename_new_file(source, destination, existing_message=existing_message)

            _status, first_summary, _stderr = self.run_with(
                root, ManagedPipeline(),
                patches=(patch("uma_bct.managed_batch.rename_new_file", side_effect=stop_details),),
            )
            batch = Path(first_summary["batch_directory"])
            catalogue = batch / CATALOGUE_FILE_NAME
            catalogue.write_bytes(b"unknown replacement")

            status, failed_summary, stderr = self.run_with(root, ManagedPipeline(port=13000))

            self.assertEqual(status, 2)
            self.assertIn("does not match its checkpoint", stderr)
            self.assertIsNone(failed_summary["catalogue_output"])
            self.assertEqual(catalogue.read_bytes(), b"unknown replacement")

    # Check that lock is os backed and malformed stale lock is blocked.
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

    # Check that distinct sibling results roots can be locked concurrently.
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

    # Check that case aliases of one macOS results root share the same guard.
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

    # Check that replacing a held lock or results root cannot create a second holder.
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

    # Check that linked results lock is blocked.
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

    # Check that dangling progress link blocks batch selection before model work.
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

            status, _summary, stderr = self.run_with(root, pipeline)

            self.assertEqual(status, 2)
            self.assertIn("symbolic link", stderr)
            self.assertEqual(pipeline.calls, [])
            self.assertEqual(len(list((root / "Results").glob("Batch *"))), 1)

    # Check that zero row batch does not finalize over a dangling catalogue link.
    def test_zero_row_batch_does_not_finalize_over_a_dangling_catalogue_link(self):
        # Keep the dangling catalogue pipeline behavior together.
        class DanglingCataloguePipeline(ManagedPipeline):
            # Process the current input.
            def process(self, path: Path) -> ImageResult:
                batch = self.work_directory.parent.parent
                (batch / CATALOGUE_FILE_NAME).symlink_to(batch / "missing-catalogue.csv")
                return super().process(path)

        with self.prepared_root("failed.jpg") as root:
            pipeline = DanglingCataloguePipeline(failed_names=("failed.jpg",))

            status, summary, stderr = self.run_with(root, pipeline)

            batch = Path(summary["batch_directory"])
            self.assertEqual(status, 2)
            self.assertIn("unexpected catalogue", stderr)
            self.assertTrue((batch / CATALOGUE_FILE_NAME).is_symlink())
            self.assertTrue((batch / PROGRESS_FILE_NAME).is_file())
            self.assertFalse((batch / DETAILS_FILE_NAME).exists())

    # Check that zero row batch preserves and blocks an unknown pending catalogue.
    def test_zero_row_batch_preserves_and_blocks_an_unknown_pending_catalogue(self):
        # Keep the pending catalogue pipeline behavior together.
        class PendingCataloguePipeline(ManagedPipeline):
            # Process the current input.
            def process(self, path: Path) -> ImageResult:
                batch = self.work_directory.parent.parent
                (batch / ".uma-bct-catalogue-pending.csv").write_bytes(b"unknown pending output")
                return super().process(path)

        with self.prepared_root("failed.jpg") as root:
            status, summary, stderr = self.run_with(
                root, PendingCataloguePipeline(failed_names=("failed.jpg",))
            )
            batch = Path(summary["batch_directory"])

            self.assertEqual(status, 2)
            self.assertIn("unexpected pending catalogue", stderr)
            self.assertEqual(
                (batch / ".uma-bct-catalogue-pending.csv").read_bytes(), b"unknown pending output"
            )
            self.assertFalse((batch / DETAILS_FILE_NAME).exists())

    # Check that naming is portable readable and batch collisions are numbered.
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

    # Check that completed batch links survive moving the whole folder.
    def test_completed_batch_links_survive_moving_the_whole_folder(self):
        with self.prepared_root("Shelf Photo.jpg") as root:
            status, summary, _stderr = self.run_with(root, ManagedPipeline())
            self.assertEqual(status, 0)
            original = Path(summary["batch_directory"])
            moved = root / "Moved Completed Batch"
            original.rename(moved)

            with (moved / CATALOGUE_FILE_NAME).open(newline="", encoding="utf-8") as source:
                row = next(csv.DictReader(source))

            self.assertIn('"Box Images/Shelf Photo - Box 01.png"', row["Cropped Box Image Link"])
            self.assertTrue((moved / BOX_IMAGES_DIRECTORY_NAME / "Shelf Photo - Box 01.png").is_file())


if __name__ == "__main__":
    unittest.main()
