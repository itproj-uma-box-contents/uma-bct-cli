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
from uma_bct.path_safety import validate_path_hierarchy
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

    def test_cropper_applies_eight_percent_padding_and_saves_padded_dimensions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            image_path = root / "shelf.jpg"
            cv2.imwrite(str(image_path), np.full((1000, 1000, 3), 220, dtype=np.uint8))
            image = load_input_image(image_path)
            regions = [DetectedRegion(BoundingBox((100, 100), (300, 500)), 1.0)]
            boxes = OpenCVBoxCropper().create_boxes(image, regions, root / "review")
            self.assertEqual(boxes[0].bounding_box, BoundingBox((84, 68), (316, 532)))
            saved = cv2.imread(str(boxes[0].image_file_path))
            self.assertEqual(saved.shape[:2], (464, 232))

    def test_cropper_uses_image_relative_minimum_padding_and_clamps_borders(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            image_path = root / "shelf.jpg"
            cv2.imwrite(str(image_path), np.full((1000, 2000, 3), 220, dtype=np.uint8))
            image = load_input_image(image_path)
            regions = [
                DetectedRegion(BoundingBox((100, 100), (120, 130)), 1.0),
                DetectedRegion(BoundingBox((0, 800), (100, 1000)), 1.0),
            ]
            boxes = OpenCVBoxCropper().create_boxes(image, regions, root / "review")
            self.assertEqual(
                [box.bounding_box for box in boxes], [BoundingBox((95, 95), (125, 135)), BoundingBox((0, 784), (108, 1000))]
            )
            self.assertEqual([cv2.imread(str(box.image_file_path)).shape[:2] for box in boxes], [(40, 30), (216, 108)])
            self.assertEqual(len({box.box_id for box in boxes}), 2)
            self.assertEqual(len({box.image_file_path for box in boxes}), 2)

    def test_cropper_rechecks_a_parent_swapped_after_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            image_path = root / "shelf.jpg"
            cv2.imwrite(str(image_path), np.full((100, 100, 3), 220, dtype=np.uint8))
            image = load_input_image(image_path)
            regions = [DetectedRegion(BoundingBox((10, 10), (90, 90)), 1.0)]
            review = root / "review"
            review.mkdir()
            validate_path_hierarchy(review, "Review-crop root")

            original = root / "review-original"
            outside = root / "outside"
            outside.mkdir()
            review.rename(original)
            review.symlink_to(outside, target_is_directory=True)

            with self.assertRaisesRegex(ValueError, "parent must not be a symbolic"):
                OpenCVBoxCropper().create_boxes(image, regions, review)
            self.assertEqual(tuple(outside.iterdir()), ())

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
        self.assertTrue(all(plausible_bbox(region.bounding_box, 1200, 800) for region in regions))

    def test_deduplication_only_merges_overlapping_independent_crops(self):
        image = make_image()
        first = make_box(image=image, box_id="first")
        overlapping = make_box(image=image, box_id="second", bbox=BoundingBox((12, 22), (92, 182)))
        separate = make_box(image=image, box_id="third", bbox=BoundingBox((110, 20), (190, 180)))
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
