import unittest
from unittest.mock import patch

import cv2
import numpy as np

from uma_bct.adapters.opencv_detector import (
    candidate_confidence,
    detect_regions,
    line_support,
    plausible_bbox,
    range_score,
)
from uma_bct.domain import BoundingBox, DetectedRegion


class OpenCVDetectorTest(unittest.TestCase):
    def test_geometry_accepts_square_and_wide_boxes_but_rejects_extremes(self):
        self.assertTrue(plausible_bbox(BoundingBox((100, 100), (400, 400)), 1000, 1000))
        self.assertTrue(plausible_bbox(BoundingBox((100, 100), (500, 450)), 1000, 1000))
        self.assertFalse(
            plausible_bbox(BoundingBox((100, 100), (600, 300)), 1000, 1000)
        )
        self.assertFalse(
            plausible_bbox(BoundingBox((100, 100), (500, 200)), 1000, 1000)
        )

    def test_range_score_has_preferred_plateau_and_tapered_edges(self):
        self.assertEqual(range_score(0.1, 0.2, 0.4, 0.6, 0.8), 0.0)
        self.assertEqual(range_score(0.5, 0.2, 0.4, 0.6, 0.8), 1.0)
        self.assertAlmostEqual(range_score(0.3, 0.2, 0.4, 0.6, 0.8), 0.5)
        self.assertAlmostEqual(range_score(0.7, 0.2, 0.4, 0.6, 0.8), 0.5)

    def test_line_support_and_rectangle_evidence_raise_confidence(self):
        bbox = (20, 20, 100, 160)
        horizontal = [(20, 20, 120, 20), (20, 180, 120, 180)]
        vertical = [(20, 20, 20, 180), (120, 20, 120, 180)]
        contour = np.array(
            [[[20, 20]], [[120, 20]], [[120, 180]], [[20, 180]]],
            dtype=np.int32,
        )

        supported = candidate_confidence(
            contour,
            bbox,
            (300, 300),
            horizontal,
            vertical,
        )
        unsupported = candidate_confidence(contour, bbox, (300, 300), [], [])

        self.assertEqual(line_support(bbox, horizontal, vertical), 1.0)
        self.assertGreater(supported, unsupported)
        self.assertLessEqual(supported, 1.0)

    def test_resized_candidates_are_mapped_back_to_source_coordinates(self):
        original = np.zeros((2800, 2000, 3), dtype=np.uint8)
        candidate = DetectedRegion(BoundingBox((100, 140), (500, 700)), 0.8)

        with patch(
            "uma_bct.adapters.opencv_detector.extract_candidates",
            return_value=[candidate],
        ):
            regions = detect_regions(original)

        self.assertEqual(regions[0].bounding_box, BoundingBox((200, 280), (1000, 1400)))
        self.assertEqual(regions[0].confidence, 0.8)

    def test_synthetic_two_box_shelf_yields_two_nonoverlapping_regions(self):
        image = np.full((800, 1200, 3), 230, dtype=np.uint8)
        for left, right in ((150, 550), (650, 1050)):
            cv2.rectangle(image, (left, 100), (right, 700), (20, 20, 20), 12)
            cv2.line(image, (left, 180), (right, 180), (20, 20, 20), 8)

        regions = detect_regions(image)

        self.assertEqual(len(regions), 2)
        centres = sorted(
            region.bounding_box.x + region.bounding_box.width / 2 for region in regions
        )
        self.assertLess(centres[0], image.shape[1] / 2)
        self.assertGreater(centres[1], image.shape[1] / 2)
        self.assertTrue(all(region.confidence >= 0.45 for region in regions))


if __name__ == "__main__":
    unittest.main()
