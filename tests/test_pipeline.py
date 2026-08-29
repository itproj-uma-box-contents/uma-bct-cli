import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.domain import (
    BoundingBox,
    Box,
    BoxText,
    BoxTranscription,
    DetectedRegion,
    InputImage,
    Label,
)
from src.imaging import OpenCVBoxCropper
from src.pipeline import BoxTranscriptionPipeline


class FakeDetector:
    name = "fake-detector"

    def __init__(self, confidence: float = 0.9) -> None:
        self.confidence = confidence

    def detect(self, image: InputImage) -> list[DetectedRegion]:
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

    def transcribe(self, box: Box) -> BoxTranscription:
        self.seen_boxes.append(box)
        confidence = 91
        return BoxTranscription(
            box_id=box.box_id,
            position=box.position,
            label=Label(box.box_id, "0001", ("Unit 0001",)),
            text_lines=(
                BoxText(1, "Unit 0001", "typed", confidence),
                BoxText(1, "Correspondence", "handwritten", confidence),
            ),
            confidence=confidence,
        )


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
            detector=FakeDetector(confidence=0.5),
            cropper=OpenCVBoxCropper(),
            transcriber=transcriber,
            work_directory=self.root / "crops",
            minimum_detector_confidence=0.6,
        )

        result = pipeline.process(self.image_path)

        self.assertEqual(result.boxes, ())
        self.assertEqual(result.transcriptions, ())
        self.assertEqual(transcriber.seen_boxes, [])


if __name__ == "__main__":
    unittest.main()
