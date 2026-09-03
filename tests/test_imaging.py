import os
import stat
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from uma_bct.domain import BoundingBox, DetectedRegion
from uma_bct.imaging import OpenCVBoxCropper, image_key, load_input_image


class OpenCVBoxCropperTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.output_directory = self.root / "run" / "crops"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write_image(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        image = np.full((200, 200, 3), 220, dtype=np.uint8)
        self.assertTrue(cv2.imwrite(str(path), image))

    def test_accepts_horizontal_and_square_qwen_regions(self):
        image_path = self.root / "shelf.jpg"
        self.write_image(image_path)
        image = load_input_image(image_path)
        regions = [
            DetectedRegion(BoundingBox((20, 20), (180, 80)), 1.0),
            DetectedRegion(BoundingBox((50, 100), (120, 170)), 1.0),
        ]

        boxes = OpenCVBoxCropper().create_boxes(image, regions, self.output_directory)

        self.assertEqual(len(boxes), 2)
        self.assertTrue(all(box.image_file_path.is_file() for box in boxes))
        if os.name != "nt":
            self.assertEqual(
                stat.S_IMODE(boxes[0].image_file_path.parent.stat().st_mode),
                0o700,
            )
            self.assertEqual(
                stat.S_IMODE(boxes[0].image_file_path.stat().st_mode),
                0o600,
            )

    def test_same_stem_inputs_have_collision_safe_ids_and_directories(self):
        first_path = self.root / "one" / "shelf.jpg"
        second_path = self.root / "two" / "shelf.png"
        self.write_image(first_path)
        self.write_image(second_path)
        region = DetectedRegion(BoundingBox((40, 20), (120, 180)), 0.9)
        cropper = OpenCVBoxCropper()

        first = cropper.create_boxes(
            load_input_image(first_path), [region], self.output_directory
        )[0]
        second = cropper.create_boxes(
            load_input_image(second_path), [region], self.output_directory
        )[0]

        self.assertNotEqual(first.box_id, second.box_id)
        self.assertNotEqual(first.image_file_path.parent, second.image_file_path.parent)
        self.assertIn("jpg", first.box_id)
        self.assertIn("png", second.box_id)

    def test_image_key_is_stable_when_a_batch_moves(self):
        first = image_key(self.root / "first-location" / "shelf.jpg")
        second = image_key(self.root / "second-location" / "shelf.jpg")

        self.assertEqual(first, second)

    def test_image_key_distinguishes_case_sensitive_linux_filenames(self):
        lower = image_key(Path("photo.jpg"))
        upper = image_key(Path("photo.JPG"))

        self.assertNotEqual(lower, upper)

    def test_image_key_bounds_long_filename_components_without_collisions(self):
        first = image_key(Path(f"{'a' * 240}x.jpg"))
        second = image_key(Path(f"{'a' * 240}y.jpg"))

        self.assertLessEqual(len(first.encode("utf-8")), 120)
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
