import unittest
from pathlib import Path

from uma_bct.domain import (
    BoundingBox,
    BoxTranscription,
    DetectedRegion,
    InputImage,
)


class DomainValidationTest(unittest.TestCase):
    def test_bounding_boxes_require_nonnegative_ordered_corners(self):
        invalid_corners = (
            ((-1, 0), (1, 1)),
            ((1, 1), (1, 2)),
            ((1, 1), (2, 1)),
        )

        for top_left, bottom_right in invalid_corners:
            with (
                self.subTest(corners=(top_left, bottom_right)),
                self.assertRaises(ValueError),
            ):
                BoundingBox(top_left, bottom_right)

    def test_input_images_require_positive_dimensions(self):
        for width, height in ((0, 1), (1, 0), (-1, 1)):
            with (
                self.subTest(size=(width, height)),
                self.assertRaisesRegex(ValueError, "dimensions must be positive"),
            ):
                InputImage(Path("shelf.jpg"), width, height)

    def test_detected_regions_require_finite_unit_interval_confidence(self):
        box = BoundingBox((0, 0), (1, 1))

        for confidence in (-0.1, 1.1, float("nan")):
            with (
                self.subTest(confidence=confidence),
                self.assertRaisesRegex(ValueError, "between 0 and 1"),
            ):
                DetectedRegion(box, confidence)

    def test_transcriptions_require_percentage_confidence(self):
        for confidence in (-1, 101):
            with (
                self.subTest(confidence=confidence),
                self.assertRaisesRegex(ValueError, "between 0 and 100"),
            ):
                BoxTranscription("box-1", "top-left", None, (), confidence)


if __name__ == "__main__":
    unittest.main()
