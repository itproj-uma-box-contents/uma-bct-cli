import re
import tempfile
import unittest
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from uma_bct.adapters.qwen_detector import (
    GROUNDING_SCHEMA,
    MAX_DETECTOR_PROPOSALS,
    QwenBoxDetector,
    validate_grounding_proposals,
)
from uma_bct.domain import BoundingBox, InputImage


class FakeVisionModel:
    def __init__(self, result: Any) -> None:
        self.result = result
        self.calls: list[tuple[Path, str, dict[str, Any]]] = []

    def query(
        self,
        image_path: Path,
        prompt: str,
        schema: dict[str, Any],
    ) -> Any:
        self.calls.append((image_path, prompt, schema))
        return self.result


class QwenBoxDetectorTest(unittest.TestCase):
    def test_validator_reports_specific_malformed_coordinates(self):
        cases = (
            ([0, 0, True, 1], "finite numbers"),
            ([-1, 0, 1, 1], "coordinates from 0 to 1000"),
            ([1, 0, 0, 1], "invalid corners"),
        )
        for coordinates, message in cases:
            with (
                self.subTest(coordinates=coordinates),
                self.assertRaisesRegex(ValueError, re.escape(message)),
            ):
                validate_grounding_proposals({"boxes": [{"bbox_2d": coordinates}]})

    def test_schema_and_validator_bound_detector_proposals(self):
        self.assertEqual(
            GROUNDING_SCHEMA["properties"]["boxes"]["maxItems"],
            MAX_DETECTOR_PROPOSALS,
        )
        oversized = {
            "boxes": [
                {"bbox_2d": [0, 0, 1, 1]} for _ in range(MAX_DETECTOR_PROPOSALS + 1)
            ]
        }
        with self.assertRaisesRegex(ValueError, "at most 100"):
            validate_grounding_proposals(oversized)

    def test_converts_normalized_qwen_coordinates_to_shared_pixel_regions(self):
        model = FakeVisionModel({"boxes": [{"bbox_2d": [100, 200, 500, 800]}]})
        detector = QwenBoxDetector(model)
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        image_path = Path(temporary_directory.name) / "shelf.jpg"
        self.assertTrue(
            cv2.imwrite(str(image_path), np.zeros((100, 200, 3), dtype=np.uint8))
        )
        image = InputImage(image_path, width=200, height=100)

        regions = detector.detect(image)

        self.assertEqual(len(regions), 1)
        self.assertEqual(
            regions[0].bounding_box,
            BoundingBox((20, 20), (100, 80)),
        )
        sanitized_path = model.calls[0][0]
        self.assertNotEqual(sanitized_path, image.file_path)
        self.assertEqual(sanitized_path.suffix, ".png")
        self.assertFalse(sanitized_path.exists())

    def test_rejects_an_empty_grounding_result_like_the_reference(self):
        detector = QwenBoxDetector(FakeVisionModel({"boxes": []}))
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        image_path = Path(temporary_directory.name) / "shelf.jpg"
        self.assertTrue(
            cv2.imwrite(str(image_path), np.zeros((100, 200, 3), dtype=np.uint8))
        )
        image = InputImage(image_path, width=200, height=100)

        with self.assertRaisesRegex(ValueError, "contained no boxes"):
            detector.detect(image)

    def test_retains_valid_qwen_proposals_and_warns_about_malformed_ones(self):
        model = FakeVisionModel(
            {
                "boxes": [
                    {"bbox_2d": [100, 200, 50, 800]},
                    {"bbox_2d": [100, 200, 500, 800]},
                ]
            }
        )
        detector = QwenBoxDetector(model)
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        image_path = Path(temporary_directory.name) / "shelf.jpg"
        self.assertTrue(
            cv2.imwrite(str(image_path), np.zeros((100, 200, 3), dtype=np.uint8))
        )
        image = InputImage(image_path, width=200, height=100)

        regions = detector.detect(image)

        self.assertEqual(len(regions), 1)
        self.assertEqual(len(detector.dropped_proposal_warnings), 1)
        self.assertIn("invalid corners", detector.dropped_proposal_warnings[0])


if __name__ == "__main__":
    unittest.main()
