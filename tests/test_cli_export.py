import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.cli import main
from src.domain import (
    BoundingBox,
    Box,
    BoxText,
    BoxTranscription,
    InputImage,
    Label,
)
from src.pipeline import ImageResult


class FakePipeline:
    def process(self, image_path: Path) -> ImageResult:
        input_image = InputImage(image_path, 1000, 500)
        box = Box(
            box_id="box-1",
            source_image=input_image,
            bounding_box=BoundingBox((10, 20), (110, 220)),
            image_file_path=image_path.parent / "crops" / "box-1.png",
            position="top-left",
            detector_confidence=0.9,
        )
        confidence = 92
        transcription = BoxTranscription(
            box_id=box.box_id,
            position=box.position,
            label=Label(box.box_id, "0001", ("1974.0056", "Unit 0001")),
            text_lines=(
                BoxText(1, "1974.0056", "typed", confidence),
                BoxText(2, "Unit 0001", "typed", confidence),
                BoxText(1, "Correspondence", "handwritten", confidence),
            ),
            confidence=confidence,
        )
        return ImageResult(
            input_image=input_image,
            detector="fake-detector",
            transcriber="fake-transcriber",
            boxes=(box,),
            transcriptions=(transcription,),
        )


class CliExportTest(unittest.TestCase):
    def test_output_argument_writes_catalogue_and_reports_record_count(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            image_path = root / "shelf.JPG"
            image_path.touch()
            output_path = root / "exports" / "catalogue.csv"
            stdout = io.StringIO()

            with (
                patch("src.cli.build_pipeline", return_value=FakePipeline()),
                contextlib.redirect_stdout(stdout),
            ):
                exit_code = main(["-i", str(image_path), "-o", str(output_path)])

            summary = json.loads(stdout.getvalue())
            written = pd.read_csv(output_path, keep_default_na=False)
            self.assertEqual(exit_code, 0)
            self.assertEqual(summary["catalogue_output"], str(output_path))
            self.assertEqual(summary["catalogue_records"], 1)
            self.assertEqual(written.loc[0, "Node Title"], "1974.0056 Unit 0001")

    def test_rejects_output_without_a_supported_extension_before_processing(self):
        with self.assertRaisesRegex(SystemExit, "Output path must end"):
            main(["-i", "unused.jpg", "-o", "catalogue.json"])


if __name__ == "__main__":
    unittest.main()
