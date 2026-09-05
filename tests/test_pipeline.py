import contextlib
import io
import unittest
from pathlib import Path
from unittest.mock import patch

from helpers import make_box, make_image, make_transcription

from uma_bct.domain import BoundingBox, Box, DetectedRegion
from uma_bct.pipeline import BoxTranscriptionPipeline


class FakeDetector:
    def __init__(self, name, outcome, warnings=()):
        self.name = name
        self.outcome = outcome
        self.dropped_proposal_warnings = warnings

    def detect(self, _image):
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


class FakeCropper:
    def create_boxes(self, image, regions, _directory):
        return [
            Box(
                f"shelf:{index:03d}",
                image,
                region.bounding_box,
                Path(f"crop-{index}.png"),
                "middle-left",
                region.confidence,
            )
            for index, region in enumerate(regions, start=1)
        ]


class ScriptedTranscriber:
    name = "qwen"

    def __init__(self, outcomes, full=()):
        self.outcomes = list(outcomes)
        self.full = full
        self.full_calls = 0

    def transcribe(self, box):
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        if outcome == "record":
            return [make_transcription(box)]
        return outcome

    def transcribe_full_image(self, box):
        self.full_calls += 1
        if isinstance(self.full, Exception):
            raise self.full
        if self.full == "record":
            return [make_transcription(box)]
        return self.full


class CropOnlyTranscriber:
    name = "qwen"

    def transcribe(self, _box):
        raise RuntimeError("unreadable")


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.image = make_image()
        self.region = DetectedRegion(BoundingBox((10, 20), (90, 180)), 0.9)
        self.full_box = make_box(
            image=self.image,
            box_id="shelf:full",
            bbox=BoundingBox((0, 0), (200, 300)),
            crop_path=Path("full.png"),
        )

    def pipeline(self, detector, transcriber, fallback=None):
        return BoxTranscriptionPipeline(
            detector=detector,
            fallback_detector=fallback,
            cropper=FakeCropper(),
            transcriber=transcriber,
            work_directory=Path("crops"),
            model_metadata={"model": "qwen-test"},
        )

    def process(self, pipeline):
        with (
            patch("uma_bct.pipeline.load_input_image", return_value=self.image),
            patch(
                "uma_bct.pipeline.create_full_image_box",
                return_value=self.full_box,
            ),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            return pipeline.process(self.image.file_path)

    def test_processes_a_normal_qwen_crop(self):
        result = self.process(
            self.pipeline(
                FakeDetector("qwen", [self.region]),
                ScriptedTranscriber(["record"]),
            )
        )
        self.assertEqual(result.status, "complete")
        self.assertEqual(len(result.records), 1)
        self.assertEqual(result.fallback_path, ("qwen",))

    def test_falls_back_from_qwen_to_opencv(self):
        result = self.process(
            self.pipeline(
                FakeDetector("qwen", RuntimeError("no boxes")),
                ScriptedTranscriber(["record"]),
                FakeDetector("opencv", [self.region]),
            )
        )
        self.assertEqual(result.actual_detector, "opencv")
        self.assertEqual(result.fallback_path, ("qwen", "opencv"))
        self.assertEqual(result.status, "complete")

    def test_uses_full_image_when_both_detectors_are_empty(self):
        transcriber = ScriptedTranscriber([], full="record")
        result = self.process(
            self.pipeline(
                FakeDetector("qwen", []),
                transcriber,
                FakeDetector("opencv", []),
            )
        )
        self.assertEqual(result.actual_detector, "full-image")
        self.assertEqual(result.fallback_path, ("qwen", "opencv", "full-image"))
        self.assertEqual(result.status, "complete")
        self.assertEqual(transcriber.full_calls, 1)

    def test_retries_full_image_when_every_crop_fails(self):
        transcriber = ScriptedTranscriber([RuntimeError("bad crop")], full="record")
        result = self.process(
            self.pipeline(FakeDetector("qwen", [self.region]), transcriber)
        )
        self.assertEqual(result.actual_detector, "full-image")
        self.assertEqual(result.status, "complete")
        self.assertEqual(transcriber.full_calls, 1)

    def test_partial_crop_failure_keeps_successful_rows(self):
        second = DetectedRegion(BoundingBox((100, 20), (180, 180)), 0.9)
        result = self.process(
            self.pipeline(
                FakeDetector("qwen", [self.region, second]),
                ScriptedTranscriber([RuntimeError("bad crop"), "record"]),
            )
        )
        self.assertEqual(result.status, "partial")
        self.assertEqual(len(result.records), 1)

    def test_preserves_multiple_physical_boxes_from_one_response(self):
        box = make_box(image=self.image)
        response = [
            make_transcription(box, unit="1", typed=("1960.0009", "Unit 1")),
            make_transcription(box, unit="2", typed=("1960.0009", "Unit 2")),
        ]
        result = self.process(
            self.pipeline(
                FakeDetector("qwen", [self.region]),
                ScriptedTranscriber([response]),
            )
        )
        self.assertEqual(len(result.records), 2)
        self.assertNotEqual(
            result.records[0].box.box_id,
            result.records[1].box.box_id,
        )

    def test_crop_only_transcriber_fails_without_inventing_a_result(self):
        result = self.process(
            self.pipeline(
                FakeDetector("opencv", [self.region]),
                CropOnlyTranscriber(),
            )
        )
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.records, ())
        self.assertIn("unavailable", result.warnings[-1])

    def test_manifest_shape_is_compact(self):
        result = self.process(
            self.pipeline(
                FakeDetector("qwen", [self.region]),
                ScriptedTranscriber(["record"]),
            )
        )
        value = result.to_dict()
        self.assertIn("boxes", value)
        self.assertIn("fallback_path", value)
        self.assertNotIn("attempts", value)
        self.assertNotIn("duration_seconds", str(value))
        self.assertNotIn("proposals", str(value))


if __name__ == "__main__":
    unittest.main()
