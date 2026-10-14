"""Model-free checks for independently served, opt-in crop transcription."""

import contextlib
import io
import os
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from helpers import make_box, make_image, make_transcription

from uma_bct.adapters.ollama_client import normalize_ollama_host
from uma_bct.bootstrap import build_pipeline, parallel_preflight
from uma_bct.domain import BoundingBox, DetectedRegion
from uma_bct.pipeline import BoxTranscriptionPipeline


class ParallelPipelineTest(unittest.TestCase):
    def run_pipeline(self, first, second, count=2, on_stage=None, extras=()):
        image = make_image()
        boxes = [make_box(image=image, box_id=f"box-{index}",
                          bbox=BoundingBox((index * 60, 0), (index * 60 + 50, 100))) for index in range(count)]
        detector = SimpleNamespace(name="qwen", detect=lambda _: [
            DetectedRegion(box.bounding_box, 0.9) for box in boxes])
        pipeline = BoxTranscriptionPipeline(detector=detector, cropper=SimpleNamespace(create_boxes=lambda *_: boxes),
                                            transcriber=first, parallel_transcribers=(second, *extras), work_directory=Path("crops"),
                                            on_stage=on_stage)
        with patch("uma_bct.pipeline.prepare_input_image", return_value=contextlib.nullcontext(image)), \
                contextlib.redirect_stderr(io.StringIO()):
            return pipeline.process(image.file_path)

    def test_independent_requests_overlap_and_preserve_detection_order(self):
        gate = threading.Barrier(2)
        second_finished = threading.Event()
        calls = []

        def first_read(box):
            calls.append(("first", box.box_id))
            gate.wait(timeout=2)
            self.assertTrue(second_finished.wait(timeout=2))
            return [make_transcription(box)]

        def second_read(box):
            calls.append(("second", box.box_id))
            gate.wait(timeout=2)
            second_finished.set()
            return [make_transcription(box)]

        result = self.run_pipeline(SimpleNamespace(name="qwen", transcribe=first_read),
                                   SimpleNamespace(name="qwen", transcribe=second_read))
        self.assertEqual(result.status, "complete")
        self.assertEqual([record.box.box_id for record in result.records], ["box-0", "box-1"])
        self.assertEqual(sorted(calls), [("first", "box-0"), ("second", "box-1")])

    def test_four_readers_process_every_box_once_and_preserve_physical_duplicates(self):
        gate = threading.Barrier(4)
        calls = []
        def read(box):
            gate.wait(timeout=2)
            calls.append(box.box_id)
            return [make_transcription(box)]
        readers = [SimpleNamespace(name="qwen", transcribe=read) for _ in range(4)]
        result = self.run_pipeline(*readers[:2], count=8, extras=readers[2:])
        self.assertEqual(result.status, "complete")
        self.assertEqual(sorted(calls), [f"box-{index}" for index in range(8)])
        self.assertEqual([record.box.box_id for record in result.records], [f"box-{index}" for index in range(8)])

    def test_failed_crop_keeps_successful_neighbour_and_partial_status(self):
        gate = threading.Barrier(2)

        def failed(_):
            gate.wait(timeout=2)
            raise RuntimeError("unreadable crop")

        def good(box):
            gate.wait(timeout=2)
            return [make_transcription(box)]

        result = self.run_pipeline(SimpleNamespace(name="qwen", transcribe=failed),
                                   SimpleNamespace(name="qwen", transcribe=good))
        self.assertEqual(result.status, "partial")
        self.assertEqual([record.box.box_id for record in result.records], ["box-1"])
        self.assertIn("unreadable crop", result.warnings[0])

    def test_worker_interrupt_propagates_without_becoming_a_crop_warning(self):
        first = SimpleNamespace(name="qwen", transcribe=Mock(side_effect=KeyboardInterrupt))
        second = SimpleNamespace(name="qwen", transcribe=lambda box: [make_transcription(box)])
        with self.assertRaises(KeyboardInterrupt):
            self.run_pipeline(first, second)

    def test_interrupted_worker_never_starts_its_next_crop(self):
        calls = []

        def interrupted(box):
            calls.append(box.box_id)
            if box.box_id == "box-0":
                raise KeyboardInterrupt
            return [make_transcription(box)]

        original_start = threading.Thread.start

        def start_before_owner_resumes(thread):
            original_start(thread)
            thread.join(timeout=1)

        with patch("uma_bct.pipeline.threading.Thread.start", start_before_owner_resumes):
            with self.assertRaises(KeyboardInterrupt):
                self.run_pipeline(SimpleNamespace(name="qwen", transcribe=interrupted),
                                  SimpleNamespace(name="qwen", transcribe=lambda box: [make_transcription(box)]), count=3)
        self.assertEqual(calls, ["box-0"])

    def test_all_failed_parallel_crops_keep_existing_full_image_recovery(self):
        first = SimpleNamespace(name="qwen", transcribe=Mock(return_value=[]),
                                transcribe_full_image=Mock(side_effect=lambda box: [make_transcription(box)]))
        second = SimpleNamespace(name="qwen", transcribe=Mock(side_effect=RuntimeError("bad crop")))
        with patch("uma_bct.pipeline.create_full_image_box", return_value=make_box(box_id="full")):
            result = self.run_pipeline(first, second)
        self.assertEqual(result.actual_detector, "full-image")
        self.assertEqual(result.status, "partial")
        first.transcribe_full_image.assert_called_once()

    def test_progress_error_drains_active_requests_before_returning(self):
        second_started, release_second, owner_done = threading.Event(), threading.Event(), threading.Event()
        errors = []

        def first_read(box):
            self.assertTrue(second_started.wait(timeout=2))
            return [make_transcription(box)]

        def second_read(box):
            second_started.set()
            release_second.wait(timeout=2)
            return [make_transcription(box)]

        def progress(message):
            if message.startswith("Finished crop"):
                raise RuntimeError("progress failed")

        def owner():
            try:
                self.run_pipeline(SimpleNamespace(name="qwen", transcribe=first_read),
                                  SimpleNamespace(name="qwen", transcribe=second_read), on_stage=progress)
            except Exception as error:
                errors.append(str(error))
            finally:
                owner_done.set()

        thread = threading.Thread(target=owner)
        thread.start()
        try:
            self.assertTrue(second_started.wait(timeout=2))
            self.assertFalse(owner_done.wait(timeout=0.25))
        finally:
            release_second.set()
            thread.join(timeout=2)
        self.assertTrue(owner_done.is_set())
        self.assertEqual(errors, ["progress failed"])


class ParallelBootstrapTest(unittest.TestCase):
    environment = {"UMA_BCT_DEMO_TRANSCRIPTION_WORKERS": "2", "UMA_BCT_ALLOW_CONTAINER_OLLAMA": "1",
                   "OLLAMA_HOST": "http://ollama:11434"}

    def build(self):
        return build_pipeline(detector_name="qwen", transcriber_name="qwen", model_name="qwen-test",
                              work_directory=Path("crops"), minimum_detector_confidence=0.5)

    def test_private_extra_hostname_requires_both_explicit_demo_and_container_opt_in(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                normalize_ollama_host("http://ollama-parallel:11434")
        with patch.dict(os.environ, self.environment, clear=True):
            self.assertEqual(normalize_ollama_host("http://ollama-parallel:11434"), "http://ollama-parallel:11434")
            with self.assertRaises(ValueError):
                normalize_ollama_host("http://another-server:11434")

    def test_second_runner_identity_is_checked_and_recorded(self):
        metadata = {"model": "qwen-test", "model_digest": "same"}
        with patch.dict(os.environ, self.environment, clear=True), \
                patch("uma_bct.bootstrap.OllamaVisionModel.preflight", return_value=metadata):
            pipeline = self.build()
        self.assertEqual(len(pipeline.parallel_transcribers), 1)
        self.assertEqual(pipeline.model_metadata["demo_transcription_workers"], "2")
        with patch.dict(os.environ, self.environment, clear=True), \
                patch("uma_bct.bootstrap.OllamaVisionModel.preflight", side_effect=[metadata, {**metadata, "model_digest": "other"}]):
            with self.assertRaisesRegex(RuntimeError, "identity"):
                self.build()

    def test_invalid_or_local_parallel_setting_fails_before_any_model_request(self):
        for environment in ({"UMA_BCT_DEMO_TRANSCRIPTION_WORKERS": "4"},
                            {"UMA_BCT_DEMO_TRANSCRIPTION_WORKERS": "2"},
                            {"UMA_BCT_DEMO_TRANSCRIPTION_WORKERS": "no"}):
            with self.subTest(environment=environment), patch.dict(os.environ, environment, clear=True), \
                    patch("uma_bct.bootstrap.OllamaVisionModel.preflight") as preflight:
                with self.assertRaises(ValueError):
                    self.build()
                preflight.assert_not_called()

    def test_four_and_six_readers_use_only_their_declared_private_hosts(self):
        for count in (4, 6, 8):
            environment = {**self.environment, "UMA_BCT_DEMO_TRANSCRIPTION_WORKERS": str(count)}
            with patch.dict(os.environ, environment, clear=True), \
                    patch("uma_bct.bootstrap.OllamaVisionModel.preflight", return_value={"model": "same", "model_digest": "same"}):
                pipeline = self.build()
                self.assertEqual(len(pipeline.parallel_transcribers), count - 1)
                self.assertEqual(pipeline.model_metadata["demo_transcription_workers"], str(count))
                self.assertEqual(normalize_ollama_host(f"http://ollama-parallel-{count}:11434"),
                                 f"http://ollama-parallel-{count}:11434")
                with self.assertRaises(ValueError):
                    normalize_ollama_host(f"http://ollama-parallel-{count + 1}:11434")

    def test_parallel_preflight_interrupt_does_not_wait_for_blocked_http_reader(self):
        blocked, release = threading.Event(), threading.Event()
        def preflight():
            blocked.set()
            release.wait(timeout=2)
            return {"model": "same"}
        def interrupted_get(_queue):
            self.assertTrue(blocked.wait(timeout=1))
            raise KeyboardInterrupt
        try:
            with patch("uma_bct.bootstrap.Queue.get", interrupted_get):
                with self.assertRaises(KeyboardInterrupt):
                    parallel_preflight((SimpleNamespace(preflight=preflight),))
        finally:
            release.set()


if __name__ == "__main__":
    unittest.main()
