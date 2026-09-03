import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

from uma_bct.domain import (
    BoundingBox,
    Box,
    BoxText,
    BoxTranscription,
    DetectedRegion,
    InputImage,
    Label,
)
from uma_bct.imaging import OpenCVBoxCropper
from uma_bct.pipeline import BoxTranscriptionPipeline


class FakeDetector:
    def __init__(
        self,
        confidence: float = 0.9,
        *,
        name: str = "fake-detector",
        regions: list[DetectedRegion] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.confidence = confidence
        self.name = name
        self.regions = regions
        self.error = error
        self.calls = 0

    def detect(self, image: InputImage) -> list[DetectedRegion]:
        self.calls += 1
        if self.error is not None:
            raise self.error
        if self.regions is not None:
            return self.regions
        return [
            DetectedRegion(
                BoundingBox((40, 20), (120, 180)),
                self.confidence,
            )
        ]


class FakeTranscriber:
    name = "fake-transcriber"

    def __init__(self) -> None:
        self.seen_boxes: list[Box] = []

    def transcribe(self, box: Box) -> list[BoxTranscription]:
        self.seen_boxes.append(box)
        confidence = 91
        return [
            BoxTranscription(
                box_id=box.box_id,
                position=box.position,
                label=Label(box.box_id, "0001", ("Unit 0001",)),
                text_lines=(
                    BoxText(1, "Unit 0001", "typed", confidence),
                    BoxText(1, "Correspondence", "handwritten", confidence),
                ),
                confidence=confidence,
            )
        ]


class MultiRecordTranscriber(FakeTranscriber):
    def transcribe(self, box: Box) -> list[BoxTranscription]:
        first = super().transcribe(box)[0]
        second = BoxTranscription(
            box_id=box.box_id,
            position="middle-right",
            label=Label(box.box_id, "0002", ("Unit 0002",)),
            text_lines=(BoxText(1, "Unit 0002", "typed", 89),),
            confidence=89,
        )
        return [first, second]


class FallbackTranscriber(FakeTranscriber):
    def __init__(self, *, full_image_empty: bool = False) -> None:
        super().__init__()
        self.full_image_empty = full_image_empty
        self.full_image_calls = 0

    def transcribe(self, box: Box) -> list[BoxTranscription]:
        self.seen_boxes.append(box)
        raise RuntimeError("crop inference failed")

    def transcribe_full_image(self, box: Box) -> list[BoxTranscription]:
        self.full_image_calls += 1
        if self.full_image_empty:
            return []
        return FakeTranscriber.transcribe(self, box)


class EmptyCropTranscriber(FakeTranscriber):
    def __init__(self) -> None:
        super().__init__()
        self.full_image_calls = 0

    def transcribe(self, box: Box) -> list[BoxTranscription]:
        self.seen_boxes.append(box)
        return []

    def transcribe_full_image(self, box: Box) -> list[BoxTranscription]:
        self.full_image_calls += 1
        return FakeTranscriber.transcribe(self, box)


class DuplicateFullImageTranscriber(FallbackTranscriber):
    def transcribe_full_image(self, box: Box) -> list[BoxTranscription]:
        self.full_image_calls += 1
        record = FakeTranscriber.transcribe(self, box)[0]
        return [record, record]


class FailingFullImageTranscriber(FallbackTranscriber):
    def transcribe_full_image(self, box: Box) -> list[BoxTranscription]:
        self.full_image_calls += 1
        raise RuntimeError("full-image inference failed")


class PartiallyFailingTranscriber(FakeTranscriber):
    def transcribe(self, box: Box) -> list[BoxTranscription]:
        self.seen_boxes.append(box)
        if box.box_id.endswith(":001"):
            raise RuntimeError("first crop failed")
        return FakeTranscriber.transcribe(self, box)


class BoxTranscriptionPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.image_path = self.root / "shelf.jpg"
        image = np.full((200, 200, 3), 220, dtype=np.uint8)
        self.assertTrue(cv2.imwrite(str(self.image_path), image))

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_links_independent_detector_and_transcriber_with_a_generic_box(self):
        transcriber = FakeTranscriber()
        pipeline = BoxTranscriptionPipeline(
            detector=FakeDetector(),
            cropper=OpenCVBoxCropper(),
            transcriber=transcriber,
            work_directory=self.root / "crops",
        )

        result = pipeline.process(self.image_path)

        self.assertEqual(result.detector, "fake-detector")
        self.assertEqual(result.transcriber, "fake-transcriber")
        self.assertEqual(len(result.boxes), 1)
        self.assertEqual(len(transcriber.seen_boxes), 1)
        self.assertTrue(result.boxes[0].image_file_path.is_file())
        self.assertEqual(
            result.boxes[0].bounding_box,
            BoundingBox((36, 12), (124, 188)),
        )
        self.assertEqual(result.transcriptions[0].label.unit_number, "0001")
        self.assertEqual(
            result.to_dict()["boxes"][0]["handwritten"],
            ["Correspondence"],
        )

    def test_filters_detector_regions_before_crop_and_transcription(self):
        transcriber = FakeTranscriber()
        pipeline = BoxTranscriptionPipeline(
            detector=FakeDetector(confidence=0.5, name="opencv"),
            cropper=OpenCVBoxCropper(),
            transcriber=transcriber,
            work_directory=self.root / "crops",
            minimum_detector_confidence=0.6,
        )

        result = pipeline.process(self.image_path)

        self.assertEqual(result.boxes, ())
        self.assertEqual(result.transcriptions, ())
        self.assertEqual(transcriber.seen_boxes, [])

    def test_warns_when_opencv_drops_only_low_confidence_proposals(self):
        regions = [
            DetectedRegion(BoundingBox((10, 10), (80, 190)), 0.59),
            DetectedRegion(BoundingBox((110, 10), (190, 190)), 0.60),
        ]
        pipeline = BoxTranscriptionPipeline(
            detector=FakeDetector(name="opencv", regions=regions),
            cropper=OpenCVBoxCropper(),
            transcriber=FakeTranscriber(),
            work_directory=self.root / "crops",
            minimum_detector_confidence=0.6,
        )

        result = pipeline.process(self.image_path)

        self.assertEqual(result.status, "complete")
        self.assertEqual(len(result.boxes), 1)
        self.assertTrue(
            any(
                "OpenCV dropped 1 of 2 proposed regions" in warning
                for warning in result.warnings
            )
        )
        detection_attempt = result.attempts[0]
        self.assertEqual(detection_attempt["proposal_count"], 2)
        self.assertEqual(detection_attempt["retained_proposal_count"], 1)

    def test_records_adapter_dropped_proposal_warnings(self):
        detector = FakeDetector(name="qwen")
        detector.dropped_proposal_warnings = (
            "Qwen dropped a malformed proposal: invalid corners",
        )
        pipeline = BoxTranscriptionPipeline(
            detector=detector,
            cropper=OpenCVBoxCropper(),
            transcriber=FakeTranscriber(),
            work_directory=self.root / "crops",
        )

        result = pipeline.process(self.image_path)

        self.assertIn(detector.dropped_proposal_warnings[0], result.warnings)
        self.assertEqual(result.attempts[0]["dropped_proposal_count"], 1)
        self.assertEqual(result.status, "partial")

    def test_qwen_confidence_is_not_filtered_by_opencv_threshold(self):
        transcriber = FakeTranscriber()
        pipeline = BoxTranscriptionPipeline(
            detector=FakeDetector(confidence=0.1, name="qwen"),
            cropper=OpenCVBoxCropper(),
            transcriber=transcriber,
            work_directory=self.root / "crops",
            minimum_detector_confidence=0.6,
        )

        result = pipeline.process(self.image_path)

        self.assertEqual(result.status, "complete")
        self.assertEqual(len(result.transcriptions), 1)

    def test_preserves_every_model_record_returned_for_one_wide_crop(self):
        pipeline = BoxTranscriptionPipeline(
            detector=FakeDetector(),
            cropper=OpenCVBoxCropper(),
            transcriber=MultiRecordTranscriber(),
            work_directory=self.root / "crops",
        )

        result = pipeline.process(self.image_path)

        self.assertEqual(len(result.boxes), 2)
        self.assertEqual(
            [item.label.unit_number for item in result.transcriptions],
            ["0001", "0002"],
        )
        self.assertNotEqual(result.boxes[0].box_id, result.boxes[1].box_id)
        self.assertEqual(
            result.boxes[0].image_file_path,
            result.boxes[1].image_file_path,
        )

    def test_retries_full_image_when_every_crop_request_fails(self):
        transcriber = FallbackTranscriber()
        pipeline = BoxTranscriptionPipeline(
            detector=FakeDetector(),
            cropper=OpenCVBoxCropper(),
            transcriber=transcriber,
            work_directory=self.root / "crops",
        )

        result = pipeline.process(self.image_path)

        self.assertEqual(len(result.transcriptions), 1)
        self.assertNotEqual(result.boxes[0].image_file_path, self.image_path)
        self.assertTrue(result.boxes[0].image_file_path.is_file())
        self.assertEqual(transcriber.full_image_calls, 1)
        self.assertEqual(result.actual_detector, "full-image")
        self.assertEqual(result.fallback_path, ("fake-detector", "full-image"))
        self.assertEqual(result.status, "complete")
        self.assertTrue(
            any("retried the original full image" in item for item in result.warnings)
        )

    def test_full_image_retry_stays_partial_when_not_every_crop_is_recovered(self):
        regions = [
            DetectedRegion(BoundingBox((10, 10), (80, 190)), 0.9),
            DetectedRegion(BoundingBox((110, 10), (190, 190)), 0.9),
        ]
        pipeline = BoxTranscriptionPipeline(
            detector=FakeDetector(regions=regions),
            cropper=OpenCVBoxCropper(),
            transcriber=FallbackTranscriber(),
            work_directory=self.root / "crops",
        )

        result = pipeline.process(self.image_path)

        self.assertEqual(len(result.transcriptions), 1)
        self.assertEqual(result.status, "partial")
        self.assertTrue(any("remain unresolved" in item for item in result.warnings))

    def test_record_count_alone_cannot_resolve_multiple_failed_crops(self):
        regions = [
            DetectedRegion(BoundingBox((10, 10), (80, 190)), 0.9),
            DetectedRegion(BoundingBox((110, 10), (190, 190)), 0.9),
        ]
        pipeline = BoxTranscriptionPipeline(
            detector=FakeDetector(regions=regions),
            cropper=OpenCVBoxCropper(),
            transcriber=DuplicateFullImageTranscriber(),
            work_directory=self.root / "crops",
        )

        result = pipeline.process(self.image_path)

        self.assertEqual(len(result.transcriptions), 2)
        self.assertEqual(result.status, "partial")
        self.assertTrue(
            any("associated one-to-one" in item for item in result.warnings)
        )

    def test_empty_crop_responses_retry_the_full_image_once(self):
        transcriber = EmptyCropTranscriber()
        pipeline = BoxTranscriptionPipeline(
            detector=FakeDetector(),
            cropper=OpenCVBoxCropper(),
            transcriber=transcriber,
            work_directory=self.root / "crops",
        )

        result = pipeline.process(self.image_path)

        self.assertEqual(len(result.transcriptions), 1)
        self.assertEqual(transcriber.full_image_calls, 1)
        self.assertEqual(result.status, "complete")
        self.assertTrue(any("returned no boxes" in item for item in result.warnings))

    def test_qwen_detection_failure_falls_back_to_opencv(self):
        opencv = FakeDetector(name="opencv")
        pipeline = BoxTranscriptionPipeline(
            detector=FakeDetector(name="qwen", error=RuntimeError("model offline")),
            fallback_detector=opencv,
            cropper=OpenCVBoxCropper(),
            transcriber=FakeTranscriber(),
            work_directory=self.root / "crops",
        )

        result = pipeline.process(self.image_path)

        self.assertEqual(opencv.calls, 1)
        self.assertEqual(result.requested_detector, "qwen")
        self.assertEqual(result.actual_detector, "opencv")
        self.assertEqual(result.fallback_path, ("qwen", "opencv"))
        self.assertEqual(result.status, "complete")
        self.assertTrue(any("OpenCV fallback" in item for item in result.warnings))

    def test_empty_qwen_detection_falls_back_to_opencv(self):
        opencv = FakeDetector(name="opencv")
        pipeline = BoxTranscriptionPipeline(
            detector=FakeDetector(name="qwen", regions=[]),
            fallback_detector=opencv,
            cropper=OpenCVBoxCropper(),
            transcriber=FakeTranscriber(),
            work_directory=self.root / "crops",
        )

        result = pipeline.process(self.image_path)

        self.assertEqual(result.actual_detector, "opencv")
        self.assertEqual(result.fallback_path, ("qwen", "opencv"))
        self.assertEqual(result.status, "complete")

    def test_empty_opencv_detection_falls_back_to_full_image(self):
        transcriber = EmptyCropTranscriber()
        pipeline = BoxTranscriptionPipeline(
            detector=FakeDetector(name="opencv", regions=[]),
            cropper=OpenCVBoxCropper(),
            transcriber=transcriber,
            work_directory=self.root / "crops",
        )

        result = pipeline.process(self.image_path)

        self.assertEqual(transcriber.full_image_calls, 1)
        self.assertEqual(result.actual_detector, "full-image")
        self.assertEqual(result.fallback_path, ("opencv", "full-image"))
        self.assertEqual(result.status, "complete")

    def test_opencv_detection_error_falls_back_to_full_image(self):
        transcriber = EmptyCropTranscriber()
        pipeline = BoxTranscriptionPipeline(
            detector=FakeDetector(name="opencv", error=RuntimeError("bad pixels")),
            cropper=OpenCVBoxCropper(),
            transcriber=transcriber,
            work_directory=self.root / "crops",
        )

        result = pipeline.process(self.image_path)

        self.assertEqual(result.fallback_path, ("opencv", "full-image"))
        self.assertEqual(result.status, "complete")
        self.assertTrue(any("bad pixels" in item for item in result.warnings))

    def test_crop_creation_error_is_recorded_before_full_image_recovery(self):
        transcriber = EmptyCropTranscriber()
        cropper = Mock()
        cropper.create_boxes.side_effect = OSError("crop disk full")
        pipeline = BoxTranscriptionPipeline(
            detector=FakeDetector(name="opencv"),
            cropper=cropper,
            transcriber=transcriber,
            work_directory=self.root / "crops",
        )

        result = pipeline.process(self.image_path)

        self.assertEqual(result.status, "complete")
        self.assertEqual(result.fallback_path, ("opencv", "full-image"))
        self.assertEqual(
            [attempt["status"] for attempt in result.attempts],
            ["failed", "complete"],
        )
        self.assertIn("crop creation failed", result.attempts[0]["error"])
        self.assertTrue(any("crop disk full" in item for item in result.warnings))

    def test_detector_failure_without_full_image_support_stays_failed(self):
        pipeline = BoxTranscriptionPipeline(
            detector=FakeDetector(name="opencv", regions=[]),
            cropper=OpenCVBoxCropper(),
            transcriber=FakeTranscriber(),
            work_directory=self.root / "crops",
        )

        result = pipeline.process(self.image_path)

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.fallback_path, ("opencv",))
        self.assertEqual(len(result.attempts), 1)
        self.assertTrue(
            any(
                "does not support full-image recovery" in item
                for item in result.warnings
            )
        )

    def test_full_image_copy_failure_preserves_the_recovery_route(self):
        pipeline = BoxTranscriptionPipeline(
            detector=FakeDetector(name="opencv", regions=[]),
            cropper=OpenCVBoxCropper(),
            transcriber=EmptyCropTranscriber(),
            work_directory=self.root / "crops",
        )

        with patch(
            "uma_bct.pipeline.create_full_image_box",
            side_effect=OSError("review copy denied"),
        ):
            result = pipeline.process(self.image_path)

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.actual_detector, "full-image")
        self.assertEqual(result.fallback_path, ("opencv", "full-image"))
        self.assertTrue(any("review copy denied" in item for item in result.warnings))

    def test_low_confidence_opencv_regions_fall_back_to_full_image(self):
        transcriber = EmptyCropTranscriber()
        pipeline = BoxTranscriptionPipeline(
            detector=FakeDetector(name="opencv", confidence=0.4),
            cropper=OpenCVBoxCropper(),
            transcriber=transcriber,
            work_directory=self.root / "crops",
            minimum_detector_confidence=0.6,
        )

        result = pipeline.process(self.image_path)

        self.assertEqual(result.fallback_path, ("opencv", "full-image"))
        self.assertEqual(result.status, "complete")
        self.assertTrue(
            any("minimum detector confidence" in item for item in result.warnings)
        )

    def test_qwen_then_invalid_opencv_regions_fall_back_to_full_image(self):
        outside_image = DetectedRegion(BoundingBox((300, 300), (400, 400)), 0.9)
        transcriber = EmptyCropTranscriber()
        pipeline = BoxTranscriptionPipeline(
            detector=FakeDetector(name="qwen", regions=[]),
            fallback_detector=FakeDetector(name="opencv", regions=[outside_image]),
            cropper=OpenCVBoxCropper(),
            transcriber=transcriber,
            work_directory=self.root / "crops",
        )

        result = pipeline.process(self.image_path)

        self.assertEqual(result.fallback_path, ("qwen", "opencv", "full-image"))
        self.assertEqual(result.actual_detector, "full-image")
        self.assertEqual(result.status, "complete")

    def test_some_unresolved_crops_make_the_image_partial(self):
        regions = [
            DetectedRegion(BoundingBox((10, 10), (80, 190)), 0.9),
            DetectedRegion(BoundingBox((110, 10), (190, 190)), 0.9),
        ]
        pipeline = BoxTranscriptionPipeline(
            detector=FakeDetector(regions=regions),
            cropper=OpenCVBoxCropper(),
            transcriber=PartiallyFailingTranscriber(),
            work_directory=self.root / "crops",
        )

        result = pipeline.process(self.image_path)

        self.assertEqual(len(result.transcriptions), 1)
        self.assertEqual(result.status, "partial")

    def test_failed_full_image_recovery_has_failed_status(self):
        pipeline = BoxTranscriptionPipeline(
            detector=FakeDetector(name="opencv", regions=[]),
            cropper=OpenCVBoxCropper(),
            transcriber=FallbackTranscriber(full_image_empty=True),
            work_directory=self.root / "crops",
        )

        result = pipeline.process(self.image_path)

        self.assertEqual(result.boxes, ())
        self.assertEqual(result.transcriptions, ())
        self.assertEqual(result.status, "failed")
        self.assertTrue(any("returned no boxes" in item for item in result.warnings))

    def test_full_image_transcription_error_is_a_failed_attempt(self):
        pipeline = BoxTranscriptionPipeline(
            detector=FakeDetector(name="opencv", regions=[]),
            cropper=OpenCVBoxCropper(),
            transcriber=FailingFullImageTranscriber(),
            work_directory=self.root / "crops",
        )

        result = pipeline.process(self.image_path)

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.actual_detector, "full-image")
        self.assertEqual(result.fallback_path, ("opencv", "full-image"))
        self.assertEqual(result.transcriptions, ())
        transcription_attempt = result.attempts[-1]
        self.assertEqual(transcription_attempt["mode"], "full-image")
        self.assertEqual(transcription_attempt["status"], "failed")
        self.assertIn("full-image inference failed", transcription_attempt["error"])

    def test_result_dictionary_includes_processing_provenance(self):
        pipeline = BoxTranscriptionPipeline(
            detector=FakeDetector(name="qwen", regions=[]),
            fallback_detector=FakeDetector(name="opencv"),
            cropper=OpenCVBoxCropper(),
            transcriber=FakeTranscriber(),
            work_directory=self.root / "crops",
        )

        value = pipeline.process(self.image_path).to_dict()

        self.assertEqual(value["requested_detector"], "qwen")
        self.assertEqual(value["actual_detector"], "opencv")
        self.assertEqual(value["fallback_path"], ["qwen", "opencv"])
        self.assertEqual(value["status"], "complete")

        attempts = value["attempts"]
        self.assertEqual(len(attempts), 3)
        self.assertEqual(
            [(attempt["stage"], attempt["status"]) for attempt in attempts],
            [
                ("detection", "empty"),
                ("detection", "complete"),
                ("transcription", "complete"),
            ],
        )
        self.assertEqual(attempts[2]["record_count"], 1)
        self.assertEqual(attempts[2]["records"][0]["typed"], ["Unit 0001"])
        self.assertGreaterEqual(attempts[2]["duration_seconds"], 0.0)

    def test_attempts_preserve_every_validated_record_before_consolidation(self):
        pipeline = BoxTranscriptionPipeline(
            detector=FakeDetector(),
            cropper=OpenCVBoxCropper(),
            transcriber=MultiRecordTranscriber(),
            work_directory=self.root / "crops",
        )

        value = pipeline.process(self.image_path).to_dict()

        transcription_attempt = value["attempts"][1]
        self.assertEqual(transcription_attempt["record_count"], 2)
        self.assertEqual(
            [record["unit"] for record in transcription_attempt["records"]],
            ["0001", "0002"],
        )

    def test_attempts_record_failed_crop_and_full_image_retry(self):
        pipeline = BoxTranscriptionPipeline(
            detector=FakeDetector(),
            cropper=OpenCVBoxCropper(),
            transcriber=FallbackTranscriber(),
            work_directory=self.root / "crops",
        )

        value = pipeline.process(self.image_path).to_dict()

        self.assertEqual(
            [attempt["status"] for attempt in value["attempts"]],
            ["complete", "failed", "complete"],
        )
        self.assertEqual(value["attempts"][1]["mode"], "crop")
        self.assertIn("crop inference failed", value["attempts"][1]["error"])
        self.assertEqual(value["attempts"][2]["mode"], "full-image")
        self.assertEqual(value["attempts"][2]["record_count"], 1)

    def test_detector_attempt_bounds_serialized_proposals(self):
        repeated = [
            DetectedRegion(BoundingBox((40, 20), (120, 180)), 0.9) for _ in range(101)
        ]
        pipeline = BoxTranscriptionPipeline(
            detector=FakeDetector(name="qwen", regions=repeated),
            cropper=OpenCVBoxCropper(),
            transcriber=FakeTranscriber(),
            work_directory=self.root / "crops",
        )

        result = pipeline.process(self.image_path)

        detection_attempt = result.attempts[0]
        self.assertEqual(detection_attempt["proposal_count"], 101)
        self.assertEqual(len(detection_attempt["proposals"]), 100)
        self.assertEqual(detection_attempt["proposals_omitted"], 1)

    def test_warns_when_overlap_suppression_drops_a_proposal(self):
        regions = [
            DetectedRegion(BoundingBox((20, 10), (120, 190)), 1.0),
            DetectedRegion(BoundingBox((25, 15), (115, 185)), 0.9),
        ]
        pipeline = BoxTranscriptionPipeline(
            detector=FakeDetector(name="qwen", regions=regions),
            cropper=OpenCVBoxCropper(),
            transcriber=FakeTranscriber(),
            work_directory=self.root / "crops",
        )

        result = pipeline.process(self.image_path)

        self.assertEqual(len(result.boxes), 1)
        self.assertTrue(
            any("proposed 2 regions; retained 1" in item for item in result.warnings)
        )


if __name__ == "__main__":
    unittest.main()
