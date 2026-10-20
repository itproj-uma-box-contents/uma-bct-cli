"""Synthetic cross-platform checks for paths, output, and local model selection."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from uma_bct.adapters.ollama_client import OllamaVisionModel
from uma_bct.adapters.qwen_detector import sanitized_detection_image
from uma_bct.domain import BoundingBox, DetectedRegion
from uma_bct.imaging import OpenCVBoxCropper, create_full_image_box, prepare_input_image, read_input_image
from uma_bct.setup_command import run_setup
from uma_bct.terminal import TerminalProgress, print_run_summary


class WindowsImagePathTest(unittest.TestCase):
    def test_every_png_output_path_supports_unicode_without_opencv_filename_io(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve() / "Photos 王 📦 #50%"
            root.mkdir()
            source = root / "shelf.bmp"
            Image.new("RGB", (40, 30), (20, 70, 130)).save(source)

            # Exercise the Windows implementation on every CI host. OpenCV
            # must receive pixels, not a filename passed to narrow fopen.
            with (
                patch("sys.platform", "win32"),
                patch("cv2.imwrite", side_effect=AssertionError("Unicode-unsafe filename write")),
                patch("tempfile.tempdir", str(root)),
                prepare_input_image(source) as image,
            ):
                expected = np.full((30, 40, 3), (130, 70, 20), dtype=np.uint8)
                self.assertTrue(np.array_equal(read_input_image(image), expected))
                with sanitized_detection_image(image) as detection:
                    self.assertTrue(detection.is_file())
                    with Image.open(detection) as saved:
                        self.assertEqual(saved.size, (40, 30))
                        self.assertFalse(saved.getexif())

                regions = [DetectedRegion(BoundingBox((5, 5), (30, 25)), 0.9)]
                crop = OpenCVBoxCropper().create_boxes(image, regions, root / "Box Images")[0]
                full = create_full_image_box(image, root / "Recovery")
                self.assertTrue(crop.image_file_path.is_file())
                self.assertTrue(full.image_file_path.is_file())

    def test_windows_encoding_failure_stays_an_image_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "shelf.bmp"
            Image.new("RGB", (10, 10), "white").save(source)
            with patch("sys.platform", "win32"), patch("cv2.imencode", return_value=(False, None)):
                with self.assertRaisesRegex(OSError, "normalized image"):
                    with prepare_input_image(source):
                        pass

    def test_windows_write_errors_are_not_reported_as_success(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "shelf.bmp"
            Image.new("RGB", (10, 10), "white").save(source)
            with patch("sys.platform", "win32"), patch.object(Path, "write_bytes", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    with prepare_input_image(source):
                        pass


class NarrowTerminalTest(unittest.TestCase):
    def test_json_and_human_summary_survive_legacy_output_encodings(self):
        summary = {"status": "complete", "catalogue_output": "C:/Users/王/Results/Catalogue.csv"}
        for human in (False, True):
            with self.subTest(human=human), io.BytesIO() as buffer:
                with io.TextIOWrapper(buffer, encoding="cp1252", errors="strict", write_through=True) as output:
                    with contextlib.redirect_stdout(output):
                        print_run_summary(summary, human=human, elapsed_seconds=0)
                    text = buffer.getvalue().decode("cp1252")
                    if human:
                        self.assertIn("Finished", text)
                        self.assertIn("Catalogue.csv", text)
                    else:
                        self.assertEqual(json.loads(text), summary)

    def test_captured_progress_does_not_fail_a_unicode_photo(self):
        with io.BytesIO() as buffer:
            with io.TextIOWrapper(buffer, encoding="ascii", errors="strict", write_through=True) as output:
                with TerminalProgress(1, stream=output, force_terminal=False) as progress:
                    progress.photo_started(1, Path("王 📦.png"))
                    progress.photo_finished(1, "complete", 1)
                self.assertIn(b"Photo 1 finished: complete", buffer.getvalue())

    def test_setup_json_survives_a_unicode_directory(self):
        with tempfile.TemporaryDirectory() as directory, io.BytesIO() as buffer:
            root = Path(directory).resolve() / "王"
            with io.TextIOWrapper(buffer, encoding="cp1252", errors="strict", write_through=True) as output:
                with (contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()),
                      patch.object(OllamaVisionModel, "ensure_model")):
                    self.assertEqual(run_setup(root=root), 0)
                self.assertEqual(json.loads(buffer.getvalue())["input_directory"], str(root / "inputs"))


class LocalModelSelectionTest(unittest.TestCase):
    def test_cloud_tags_are_rejected_before_any_ollama_client_is_created(self):
        for model in ("qwen3-vl:235b-cloud", "gpt-oss:120b-cloud", "ollama.com/gpt-oss:120b-cloud", "qwen:cloud"):
            with self.subTest(model=model), patch.object(OllamaVisionModel, "client") as client:
                with self.assertRaisesRegex(ValueError, "cloud"):
                    OllamaVisionModel(model=model)
                client.assert_not_called()

    def test_regular_local_model_names_remain_supported(self):
        for model in ("qwen3-vl:8b-instruct-q4_K_M", "qwen-test", "cloud-notes:8b"):
            with self.subTest(model=model):
                self.assertEqual(OllamaVisionModel(model=model).model, model)
