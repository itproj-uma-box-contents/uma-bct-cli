import tempfile
import unittest
from pathlib import Path

import pandas as pd

from constants import OUTPUT_COLUMNS
from src.catalogue import map_catalogue_record
from src.domain import (
    BoundingBox,
    Box,
    BoxText,
    BoxTranscription,
    InputImage,
    Label,
)
from src.export import catalogue_dataframe, export_catalogue
from src.pipeline import ImageResult


class CatalogueMappingTest(unittest.TestCase):
    def test_maps_label_annotations_and_image_to_current_contract(self):
        transcription = self.transcription(
            "box-1",
            "0007",
            ("1974.0056", "Unit 0007", "Printed note"),
            ("Correspondence", "Restricted"),
        )

        record = map_catalogue_record(transcription, Path("crops/box-1.png"))

        self.assertEqual(list(record), OUTPUT_COLUMNS)
        self.assertEqual(record["Node Type"], "Unit")
        self.assertEqual(record["Node Title"], "1974.0056 Unit 0007")
        self.assertEqual(
            record["Description"],
            "Printed note\nCorrespondence\nRestricted",
        )
        self.assertEqual(record["Date Range"], "")
        self.assertEqual(record["Image"], "crops/box-1.png")
        self.assertEqual(record["Flags"], "")

    def test_preserves_text_but_leaves_title_blank_when_unit_is_missing(self):
        transcription = self.transcription(
            "box-1", "", ("1974.0056",), ("Unclear unit",)
        )

        record = map_catalogue_record(transcription, "box.png")

        self.assertEqual(record["Node Title"], "")
        self.assertEqual(record["Description"], "1974.0056\nUnclear unit")
        self.assertEqual(record["Flags"], "Missing unit")

    def test_adds_extracted_unit_when_typed_lines_omit_it(self):
        transcription = self.transcription(
            "box-1", "0042", ("1974.0056",), ("Minutes",)
        )

        record = map_catalogue_record(transcription, "box.png")

        self.assertEqual(record["Node Title"], "1974.0056 Unit 0042")
        self.assertEqual(record["Description"], "Minutes")

    @staticmethod
    def transcription(
        box_id: str,
        unit: str,
        typed: tuple[str, ...],
        handwritten: tuple[str, ...],
    ) -> BoxTranscription:
        confidence = 90
        text_lines = tuple(
            BoxText(position, text, "typed", confidence)
            for position, text in enumerate(typed, start=1)
        ) + tuple(
            BoxText(position, text, "handwritten", confidence)
            for position, text in enumerate(handwritten, start=1)
        )
        return BoxTranscription(
            box_id=box_id,
            position="top-left",
            label=Label(box_id, unit, typed),
            text_lines=text_lines,
            confidence=confidence,
        )


class CatalogueExportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_flags_duplicates_across_images_and_writes_csv(self):
        results = [self.result("shelf-a", "box-a"), self.result("shelf-b", "box-b")]
        output_path = self.root / "nested" / "catalogue.csv"

        dataframe = export_catalogue(results, output_path)
        written = pd.read_csv(output_path, keep_default_na=False)

        self.assertEqual(list(written.columns), OUTPUT_COLUMNS)
        self.assertEqual(len(dataframe), 2)
        self.assertEqual(
            written["Flags"].tolist(),
            ["Potential Duplicate", "Potential Duplicate"],
        )

    def test_writes_xlsx_with_the_same_column_contract(self):
        output_path = self.root / "catalogue.xlsx"

        export_catalogue([self.result("shelf-a", "box-a")], output_path)
        written = pd.read_excel(output_path, keep_default_na=False)

        self.assertEqual(list(written.columns), OUTPUT_COLUMNS)
        self.assertEqual(written.loc[0, "Node Title"], "1974.0056 Unit 0001")
        self.assertEqual(written.loc[0, "Description"], "Correspondence")

    def test_empty_batch_still_writes_the_required_headers(self):
        output_path = self.root / "empty.csv"

        dataframe = export_catalogue([], output_path)
        written = pd.read_csv(output_path, keep_default_na=False)

        self.assertTrue(dataframe.empty)
        self.assertEqual(list(written.columns), OUTPUT_COLUMNS)

    def test_rejects_unsupported_output_extension(self):
        with self.assertRaisesRegex(ValueError, "Unsupported output extension"):
            export_catalogue([], self.root / "catalogue.json")

    def test_rejects_a_box_without_a_matching_transcription(self):
        result = self.result("shelf-a", "box-a")
        invalid = ImageResult(
            input_image=result.input_image,
            detector=result.detector,
            transcriber=result.transcriber,
            boxes=result.boxes,
            transcriptions=(),
        )

        with self.assertRaisesRegex(ValueError, "No transcription found"):
            catalogue_dataframe([invalid])

    def result(self, shelf_name: str, box_id: str) -> ImageResult:
        input_image = InputImage(self.root / f"{shelf_name}.jpg", 1000, 500)
        box = Box(
            box_id=box_id,
            source_image=input_image,
            bounding_box=BoundingBox((10, 20), (110, 220)),
            image_file_path=self.root / "crops" / f"{box_id}.png",
            position="top-left",
            detector_confidence=0.9,
        )
        transcription = CatalogueMappingTest.transcription(
            box_id,
            "0001",
            ("1974.0056", "Unit 0001"),
            ("Correspondence",),
        )
        return ImageResult(
            input_image=input_image,
            detector="fake-detector",
            transcriber="fake-transcriber",
            boxes=(box,),
            transcriptions=(transcription,),
        )


if __name__ == "__main__":
    unittest.main()
