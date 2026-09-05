import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from helpers import make_box, make_image, make_transcription

from uma_bct.adapters.opencv_detector import detect_regions, plausible_bbox
from uma_bct.domain import BoundingBox, DetectedRegion, InputImage
from uma_bct.imaging import OpenCVBoxCropper, load_input_image
from uma_bct.transcriptions import ObservedBox, deduplicate


class ImagingTest(unittest.TestCase):
    def test_domain_values_reject_invalid_geometry_and_confidence(self):
        cases = (
            lambda: BoundingBox((-1, 0), (10, 10)),
            lambda: BoundingBox((10, 0), (10, 10)),
            lambda: InputImage(Path("x.jpg"), 0, 10),
            lambda: DetectedRegion(BoundingBox((0, 0), (10, 10)), 1.1),
        )
        for build in cases:
            with self.subTest(build=build), self.assertRaises(ValueError):
                build()

    def test_cropper_writes_reviewable_crops(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "shelf.jpg"
            cv2.imwrite(str(image_path), np.full((200, 200, 3), 220, dtype=np.uint8))
            image = load_input_image(image_path)
            regions = [
                DetectedRegion(BoundingBox((20, 20), (180, 80)), 1.0),
                DetectedRegion(BoundingBox((50, 100), (120, 170)), 1.0),
            ]
            boxes = OpenCVBoxCropper().create_boxes(image, regions, root / "review")
            self.assertEqual(len(boxes), 2)
            self.assertTrue(all(box.image_file_path.is_file() for box in boxes))
            self.assertNotEqual(boxes[0].box_id, boxes[1].box_id)

    def test_unreadable_image_has_a_clear_error(self):
        with (
            patch("uma_bct.imaging.cv2.imread", return_value=None),
            self.assertRaisesRegex(ValueError, "Could not read image"),
        ):
            load_input_image("missing.jpg")

    def test_opencv_fallback_finds_two_synthetic_boxes(self):
        image = np.full((800, 1200, 3), 230, dtype=np.uint8)
        for left, right in ((150, 550), (650, 1050)):
            cv2.rectangle(image, (left, 100), (right, 700), (20, 20, 20), 12)
            cv2.line(image, (left, 180), (right, 180), (20, 20, 20), 8)
        regions = detect_regions(image)
        self.assertEqual(len(regions), 2)
        self.assertTrue(
            all(plausible_bbox(region.bounding_box, 1200, 800) for region in regions)
        )

    def test_deduplication_only_merges_overlapping_independent_crops(self):
        image = make_image()
        first = make_box(image=image, box_id="first")
        overlapping = make_box(
            image=image,
            box_id="second",
            bbox=BoundingBox((12, 22), (92, 182)),
        )
        separate = make_box(
            image=image,
            box_id="third",
            bbox=BoundingBox((110, 20), (190, 180)),
        )
        observations = [
            ObservedBox(first, make_transcription(first), 1),
            ObservedBox(overlapping, make_transcription(overlapping), 2),
            ObservedBox(separate, make_transcription(separate), 3),
        ]
        self.assertEqual(len(deduplicate(observations)), 2)
        same_response = [
            ObservedBox(first, make_transcription(first), 1),
            ObservedBox(overlapping, make_transcription(overlapping), 1),
        ]
        self.assertEqual(len(deduplicate(same_response)), 2)


if __name__ == "__main__":
    unittest.main()
