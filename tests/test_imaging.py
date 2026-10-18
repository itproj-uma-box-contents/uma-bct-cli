import base64
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import tifffile
from helpers import make_box, make_image, make_transcription
from ollama import Image as OllamaImage
from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngInfo
from pillow_heif import register_heif_opener

from uma_bct.adapters.opencv_detector import detect_regions, plausible_bbox
from uma_bct.adapters.qwen_detector import sanitized_detection_image
from uma_bct.domain import BoundingBox, DetectedRegion, InputImage
from uma_bct.imaging import OpenCVBoxCropper, load_input_image, prepare_input_image, read_image, read_input_image
from uma_bct.path_safety import check_path
from uma_bct.transcriptions import ObservedBox, deduplicate


def write_synthetic_dng(path: Path) -> None:
    raw_pixels = np.tile(np.array([[400, 800], [800, 1200]], dtype=np.uint16), (32, 48))
    tags = (
        (274, "H", 1, 6, False),
        (50706, "B", 4, (1, 4, 0, 0), False),
        (50707, "B", 4, (1, 3, 0, 0), False),
        (50708, "s", 0, "UMA synthetic DNG", False),
        (33421, "H", 2, (2, 2), False),
        (33422, "B", 4, (0, 1, 1, 2), False),
        (50710, "B", 3, (0, 1, 2), False),
        (50711, "H", 1, 1, False),
        (50714, "2I", 1, (0, 1), False),
        (50717, "I", 1, 4095, False),
        (50721, "2i", 9, ((1, 1), (0, 1), (0, 1), (0, 1), (1, 1), (0, 1), (0, 1), (0, 1), (1, 1)), False),
        (50728, "2I", 3, ((1, 1), (1, 1), (1, 1)), False),
        (50778, "H", 1, 21, False),
    )
    tifffile.imwrite(path, raw_pixels, photometric=32803, metadata=None, extratags=tags)


def model_payload_pixels(path: Path) -> np.ndarray:
    """Decode the real client's serialized image bytes without a request."""

    encoded = OllamaImage(value=str(path)).model_dump()
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as payload:
        rgb = ImageOps.exif_transpose(payload).convert("RGB")
        return cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)


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
            check_path(review, "Review-crop root")

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

    def test_windows_reads_unicode_image_paths_and_preserves_jpeg_orientation(self):
        with tempfile.TemporaryDirectory() as directory:
            for extension, expected_shape in (("png", (20, 40, 3)), ("jpg", (40, 20, 3))):
                with self.subTest(extension=extension):
                    source = Path(directory) / f"Shelf é 📦.{extension}"
                    image = Image.new("RGB", (40, 20), (23, 77, 131))
                    exif = Image.Exif()
                    if extension == "jpg":
                        exif[274] = 6
                    image.save(source, exif=exif)
                    with patch("sys.platform", "win32"), patch("uma_bct.imaging.cv2.imread", return_value=None):
                        pixels = read_image(source)
                    self.assertEqual(pixels.shape, expected_shape)
                    self.assertEqual(pixels.dtype, np.uint8)
                    if extension == "png":
                        np.testing.assert_array_equal(pixels, np.full(expected_shape, (131, 77, 23), dtype=np.uint8))

    def test_windows_unreadable_images_have_a_clear_error(self):
        with tempfile.TemporaryDirectory() as directory:
            for name, content in (("missing.png", None), ("empty.png", b""), ("invalid.png", b"not an image")):
                with self.subTest(name=name):
                    source = Path(directory) / name
                    if content is not None:
                        source.write_bytes(content)
                    with patch("sys.platform", "win32"), self.assertRaisesRegex(ValueError, "Could not read image"):
                        read_image(source)

    def test_prepares_new_pillow_format_once_and_removes_private_temporary_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root / "shelf.bmp"
            Image.new("RGB", (13, 7), (10, 20, 30)).save(source)

            with prepare_input_image(source) as image:
                processing_path = image.processing_file_path
                self.assertEqual(image.file_path, source)
                self.assertEqual((image.width, image.height), (13, 7))
                self.assertIsNotNone(processing_path)
                self.assertTrue(processing_path.is_file())
                self.assertEqual(read_input_image(image).shape, (7, 13, 3))
            self.assertFalse(processing_path.exists())

    def test_existing_formats_keep_the_direct_opencv_path(self):
        with tempfile.TemporaryDirectory() as directory:
            for extension in ("jpg", "png", "webp"):
                with self.subTest(extension=extension):
                    source = Path(directory).resolve() / f"shelf.{extension}"
                    cv2.imwrite(str(source), np.full((7, 13, 3), 120, dtype=np.uint8))

                    with prepare_input_image(source) as image:
                        self.assertEqual(image.readable_file_path, source)
                        self.assertIsNone(image.processing_file_path)
                        self.assertEqual(read_input_image(image).shape, (7, 13, 3))

    def test_tiff_orientation_preserves_all_landmarks_for_every_orientation(self):
        rgb = np.zeros((20, 40, 3), dtype=np.uint8)
        rgb[:10, :20] = (10, 20, 30)
        rgb[:10, 20:] = (40, 50, 60)
        rgb[10:, :20] = (70, 80, 90)
        rgb[10:, 20:] = (100, 110, 120)
        with tempfile.TemporaryDirectory() as directory:
            for orientation in range(1, 9):
                with self.subTest(orientation=orientation):
                    source = Path(directory).resolve() / f"Shelf é {orientation}.tiff"
                    image = Image.fromarray(rgb)
                    exif = Image.Exif()
                    exif[274] = orientation
                    image.save(source, exif=exif)
                    image.getexif()[274] = orientation
                    expected = cv2.cvtColor(np.asarray(ImageOps.exif_transpose(image)), cv2.COLOR_RGB2BGR)

                    with prepare_input_image(source) as prepared:
                        pixels = read_input_image(prepared)
                        self.assertEqual((prepared.width, prepared.height), (expected.shape[1], expected.shape[0]))
                        np.testing.assert_array_equal(pixels, expected)
                        with sanitized_detection_image(prepared) as model_image:
                            np.testing.assert_array_equal(model_payload_pixels(model_image), expected)
                        region = DetectedRegion(BoundingBox((0, 0), (prepared.width, prepared.height)), 1.0)
                        crop = OpenCVBoxCropper().create_boxes(prepared, [region], source.parent / "review")[0]
                        np.testing.assert_array_equal(model_payload_pixels(crop.image_file_path), expected)
                        np.testing.assert_array_equal(read_image(crop.image_file_path), expected)

    def test_opaque_sixteen_bit_png_and_tiff_keep_the_existing_eight_bit_values(self):
        rgb = np.array([[[0, 16384, 32768], [65535, 49152, 256]]], dtype=np.uint16)
        with tempfile.TemporaryDirectory() as directory:
            for extension in ("png", "tiff"):
                with self.subTest(extension=extension):
                    source = Path(directory).resolve() / f"shelf.{extension}"
                    self.assertTrue(cv2.imwrite(str(source), rgb))
                    expected = cv2.imread(str(source), cv2.IMREAD_COLOR)
                    with prepare_input_image(source) as prepared:
                        self.assertIsNone(prepared.processing_file_path)
                        self.assertEqual(read_input_image(prepared).dtype, np.uint8)
                        np.testing.assert_array_equal(read_input_image(prepared), expected)

    def test_transparent_png_and_webp_share_oriented_metadata_free_model_and_crop_pixels(self):
        rgba = np.zeros((20, 40, 4), dtype=np.uint8)
        rgba[2:8, 3:12] = (0, 0, 0, 255)
        rgba[12:18, 25:37] = (240, 40, 20, 128)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for extension in ("png", "webp"):
                with self.subTest(extension=extension):
                    source = root / f"Shelf é.{extension}"
                    exif = Image.Exif()
                    exif[274] = 6
                    exif[270] = "synthetic private metadata"
                    image = Image.fromarray(rgba)
                    image.save(source, exif=exif, lossless=True)
                    oriented = Image.fromarray(rgba).transpose(Image.Transpose.ROTATE_270)
                    white = Image.new("RGBA", oriented.size, "white")
                    white.alpha_composite(oriented)
                    expected = cv2.cvtColor(np.asarray(white.convert("RGB")), cv2.COLOR_RGB2BGR)

                    with prepare_input_image(source) as prepared:
                        normalized = prepared.processing_file_path
                        self.assertIsNotNone(normalized)
                        self.assertEqual(prepared.file_path, source)
                        self.assertEqual((prepared.width, prepared.height), (20, 40))
                        np.testing.assert_array_equal(read_input_image(prepared), expected)
                        if sys.platform != "win32":
                            self.assertEqual(normalized.stat().st_mode & 0o777, 0o600)
                            self.assertEqual(normalized.parent.stat().st_mode & 0o777, 0o700)
                        with Image.open(normalized) as clean:
                            self.assertFalse(clean.getexif())
                            self.assertNotIn("exif", clean.info)
                        with sanitized_detection_image(prepared) as model_image:
                            np.testing.assert_array_equal(read_image(model_image), expected)
                            np.testing.assert_array_equal(model_payload_pixels(model_image), expected)
                        region = DetectedRegion(BoundingBox((0, 0), (20, 40)), 1.0)
                        crop = OpenCVBoxCropper().create_boxes(prepared, [region], root / "review")[0]
                        np.testing.assert_array_equal(read_image(crop.image_file_path), expected)
                        np.testing.assert_array_equal(model_payload_pixels(crop.image_file_path), expected)
                    self.assertFalse(normalized.exists())

    def test_palette_transparency_uses_white_and_discards_private_png_text(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory).resolve() / "palette.png"
            image = Image.new("P", (8, 6), 0)
            image.putpalette([0, 0, 0, 240, 20, 20] + [0, 0, 0] * 254)
            image.putpixel((4, 3), 1)
            metadata = PngInfo()
            metadata.add_text("private", "synthetic metadata")
            image.save(source, transparency=0, pnginfo=metadata)

            with prepare_input_image(source) as prepared:
                normalized = prepared.processing_file_path
                self.assertIsNotNone(normalized)
                pixels = read_input_image(prepared)
                self.assertEqual(pixels[0, 0].tolist(), [255, 255, 255])
                self.assertEqual(pixels[3, 4].tolist(), [20, 20, 240])
                with Image.open(normalized) as clean:
                    self.assertNotIn("private", clean.info)
            self.assertFalse(normalized.exists())

    def test_transparent_input_working_copy_is_removed_after_processing_or_write_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory).resolve() / "transparent.png"
            Image.new("RGBA", (8, 6), (0, 0, 0, 0)).save(source)
            with self.assertRaisesRegex(RuntimeError, "synthetic processing failure"):
                with prepare_input_image(source) as prepared:
                    normalized = prepared.processing_file_path
                    self.assertIsNotNone(normalized)
                    raise RuntimeError("synthetic processing failure")
            self.assertFalse(normalized.parent.exists())

            attempted = []

            def failed_write(path, pixels):
                attempted.append(path)
                return False

            with patch("uma_bct.imaging.write_png", side_effect=failed_write):
                with self.assertRaisesRegex(OSError, "Could not create normalized image"):
                    with prepare_input_image(source):
                        self.fail("Failed normalization must not enter the pipeline")
            self.assertEqual(len(attempted), 1)
            self.assertFalse(attempted[0].parent.exists())

    def test_corrupt_supported_image_has_a_clear_error(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory).resolve() / "corrupt.heic"
            source.write_bytes(b"not an image")
            with self.assertRaisesRegex(ValueError, "Could not read image"):
                with prepare_input_image(source):
                    self.fail("A corrupt image should not enter the pipeline")

    def test_heic_uses_the_primary_image_from_a_multi_image_container(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory).resolve() / "shelf.heic"
            first = Image.new("RGB", (12, 8), (240, 20, 20))
            primary = Image.new("RGB", (12, 8), (20, 240, 20))
            register_heif_opener()
            first.save(source, format="HEIF", save_all=True, append_images=[primary], primary_index=1, quality=100)

            with prepare_input_image(source) as image:
                blue, green, red = (int(value) for value in read_input_image(image)[4, 6])

            self.assertGreater(green, 200)
            self.assertLess(max(red, blue), 60)

    def test_gif_uses_only_the_first_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory).resolve() / "shelf.gif"
            first = Image.new("RGB", (12, 8), (240, 20, 20))
            second = Image.new("RGB", (12, 8), (20, 240, 20))
            first.save(source, save_all=True, append_images=[second])

            with prepare_input_image(source) as image:
                blue, green, red = (int(value) for value in read_input_image(image)[4, 6])

            self.assertGreater(red, 200)
            self.assertLess(max(green, blue), 60)

    def test_new_format_transparency_is_flattened_onto_white(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory).resolve() / "transparent.gif"
            image = Image.new("P", (8, 6), 0)
            image.putpalette([0, 0, 0, 240, 20, 20] + [0, 0, 0] * 254)
            image.info["transparency"] = 0
            image.putpixel((4, 3), 1)
            image.save(source, transparency=0)

            with prepare_input_image(source) as prepared:
                pixels = read_input_image(prepared)

            self.assertEqual(pixels[0, 0].tolist(), [255, 255, 255])
            self.assertGreater(int(pixels[3, 4, 2]), 200)

    def test_avif_applies_exif_orientation(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory).resolve() / "shelf.avif"
            exif = Image.Exif()
            exif[274] = 6
            Image.new("RGB", (12, 8), (10, 20, 30)).save(source, exif=exif, quality=100)

            with prepare_input_image(source) as image:
                self.assertEqual((image.width, image.height), (8, 12))

    def test_dng_uses_full_size_oriented_camera_development(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory).resolve() / "shelf.dng"
            source.write_bytes(b"synthetic raw")
            raw = MagicMock()
            raw.postprocess.return_value = np.full((7, 13, 3), (30, 20, 10), dtype=np.uint8)
            context = MagicMock()
            context.__enter__.return_value = raw

            with patch("uma_bct.imaging.rawpy.imread", return_value=context) as imread:
                with prepare_input_image(source) as image:
                    pixels = read_input_image(image)

            imread.assert_called_once_with(str(source))
            raw.postprocess.assert_called_once()
            options = raw.postprocess.call_args.kwargs
            self.assertFalse(options["half_size"])
            self.assertTrue(options["use_camera_wb"])
            self.assertEqual(options["output_bps"], 8)
            self.assertEqual(pixels[0, 0].tolist(), [10, 20, 30])

    def test_synthetic_dng_decodes_through_real_libraw_with_orientation(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory).resolve() / "synthetic.dng"
            write_synthetic_dng(source)

            with prepare_input_image(source) as image:
                pixels = read_input_image(image)

            self.assertEqual((image.width, image.height), (64, 96))
            self.assertEqual(pixels.shape, (96, 64, 3))
            self.assertEqual(pixels.dtype, np.uint8)
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
