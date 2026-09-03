import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from uma_bct.catalogue import map_catalogue_record
from uma_bct.constants import OUTPUT_COLUMNS
from uma_bct.domain import (
    BoundingBox,
    Box,
    BoxText,
    BoxTranscription,
    InputImage,
    Label,
)
from uma_bct.export import catalogue_dataframe, export_catalogue
from uma_bct.pipeline import ImageResult


class CatalogueMappingTest(unittest.TestCase):
    def test_maps_client_contract_and_review_fields(self):
        transcription = self.transcription(
            "box-1",
            "7",
            ("Australian Pulp and Paper Mills", "1974.0056", "Unit 7"),
            ("Correspondence", "Restricted"),
            identifiers=("1974.0056",),
            auxiliary=("Water damaged",),
        )

        record = map_catalogue_record(transcription, Path("crops/box-1.png"))

        self.assertEqual(list(record), OUTPUT_COLUMNS)
        self.assertEqual(record["Node ID"], "")
        self.assertEqual(record["Node Type"], "Unit")
        self.assertEqual(record["Node Title"], "1974.0056 Unit 0007")
        self.assertEqual(
            record["Description"],
            "Correspondence; Restricted; Water damaged",
        )
        self.assertEqual(record["Date Range"], "")
        self.assertEqual(record["Confidence Score"], 90)
        self.assertEqual(record["Image"], "crops/box-1.png")
        self.assertEqual(record["Flags"], "")

    def test_recognises_every_supported_identifier_format(self):
        cases = (
            ("1960.0009", "1960.0009 Unit 0319"),
            ("96/9", "96/9 Unit 0319"),
            ("uma-ac-000000330", "UMA-AC-000000330 Unit 0319"),
            ("uma-sre-20160038", "UMA-SRE-20160038 Unit 0319"),
        )
        for identifier, expected in cases:
            with self.subTest(identifier=identifier):
                transcription = self.transcription(
                    "box-1",
                    "Unit 319",
                    (f"Records {identifier}", "Unit 319"),
                    (),
                    identifiers=(identifier,),
                )

                record = map_catalogue_record(transcription, "box.png")

                self.assertEqual(record["Node Title"], expected)
                self.assertEqual(record["Flags"], "")

    def test_uses_first_of_multiple_identifiers_and_flags_for_review(self):
        transcription = self.transcription(
            "box-1",
            "8",
            ("1974.0029", "1974.0030", "1997.0138", "Unit 8"),
            (),
            identifiers=("1974.0029", "1974.0030", "1997.0138"),
        )

        record = map_catalogue_record(transcription, "box.png")

        self.assertEqual(record["Node Title"], "1974.0029 Unit 0008")
        self.assertEqual(record["Flags"], "Multiple identifiers")

    def test_blanks_identity_when_typed_text_has_unreported_extra_identifiers(self):
        transcription = self.transcription(
            "box-1",
            "8",
            ("1974.0029", "1974.0030", "1997.0138", "Unit 8"),
            (),
            identifiers=("1974.0029",),
        )

        record = map_catalogue_record(transcription, "box.png")

        self.assertEqual(record["Node Title"], "")
        self.assertEqual(
            record["Flags"],
            "Ambiguous identifier; Transcription mismatch",
        )

    def test_blanks_identity_when_structured_identifier_order_conflicts(self):
        transcription = self.transcription(
            "box-1",
            "8",
            ("1974.0029", "1974.0030", "Unit 8"),
            (),
            identifiers=("1974.0030", "1974.0029"),
        )

        record = map_catalogue_record(transcription, "box.png")

        self.assertEqual(record["Node Title"], "")
        self.assertEqual(
            record["Flags"],
            "Ambiguous identifier; Transcription mismatch",
        )

    def test_uses_structured_order_across_typed_and_handwritten_text(self):
        transcription = self.transcription(
            "box-1",
            "8",
            ("1997.0138", "Unit 8"),
            ("1974.0029",),
            identifiers=("1974.0029", "1997.0138"),
        )

        record = map_catalogue_record(transcription, "box.png")

        self.assertEqual(record["Node Title"], "1974.0029 Unit 0008")
        self.assertEqual(record["Flags"], "Multiple identifiers")

    def test_date_like_annotation_does_not_override_structured_identifier(self):
        transcription = self.transcription(
            "box-1",
            "8",
            ("1974.0029", "Unit 8"),
            ("Meeting 12/5",),
            identifiers=("1974.0029",),
        )

        record = map_catalogue_record(transcription, "box.png")

        self.assertEqual(record["Node Title"], "1974.0029 Unit 0008")
        self.assertEqual(record["Description"], "Meeting 12/5")
        self.assertEqual(record["Flags"], "")

    def test_never_infers_identifier_from_date_like_raw_text(self):
        transcription = self.transcription(
            "box-1",
            "8",
            ("Unit 8",),
            ("Meeting 12/5",),
        )

        record = map_catalogue_record(transcription, "box.png")

        self.assertEqual(record["Node Title"], "")
        self.assertEqual(record["Flags"], "Missing identifier")

    def test_mixed_valid_and_invalid_structured_identifiers_are_ambiguous(self):
        transcription = self.transcription(
            "box-1",
            "8",
            ("1974.0029", "1974.OO30", "Unit 8"),
            (),
            identifiers=("1974.0029", "1974.OO30"),
        )

        record = map_catalogue_record(transcription, "box.png")

        self.assertEqual(record["Node Title"], "")
        self.assertEqual(
            record["Flags"],
            "Ambiguous identifier; Transcription mismatch",
        )

    def test_flags_missing_and_ambiguous_identifiers_without_guessing(self):
        missing = self.transcription("box-1", "42", ("Sheehan Family", "Unit 42"), ())
        ambiguous = self.transcription(
            "box-2",
            "42",
            ("Unit 42",),
            (),
            identifiers=("1974.OO29 or 1974.002B",),
        )

        missing_record = map_catalogue_record(missing, "missing.png")
        ambiguous_record = map_catalogue_record(ambiguous, "ambiguous.png")

        self.assertEqual(missing_record["Node Title"], "")
        self.assertEqual(missing_record["Flags"], "Missing identifier")
        self.assertEqual(ambiguous_record["Node Title"], "")
        self.assertEqual(
            ambiguous_record["Flags"],
            "Ambiguous identifier; Transcription mismatch",
        )

    def test_flags_missing_or_invalid_units_without_guessing(self):
        missing = self.transcription(
            "box-1",
            "",
            ("1974.0056",),
            ("Unclear unit",),
            identifiers=("1974.0056",),
        )
        invalid = self.transcription(
            "box-2",
            "7 or 8",
            ("1974.0056",),
            ("Crossed out",),
            identifiers=("1974.0056",),
        )

        missing_record = map_catalogue_record(missing, "missing.png")
        invalid_record = map_catalogue_record(invalid, "invalid.png")

        self.assertEqual(missing_record["Node Title"], "")
        self.assertEqual(missing_record["Description"], "Unclear unit")
        self.assertEqual(missing_record["Flags"], "Missing unit")
        self.assertEqual(invalid_record["Node Title"], "")
        self.assertEqual(invalid_record["Flags"], "Invalid unit")

    def test_blanks_uncorroborated_structured_identity_and_flags_mismatch(self):
        transcription = self.transcription(
            "box-1",
            "30",
            ("Melbourne College Advanced Education",),
            ("Staff 1970s",),
            confidence=95,
            identifiers=("1990.0003",),
            needs_review=True,
        )

        record = map_catalogue_record(transcription, "box.png")

        self.assertEqual(record["Node Title"], "")
        self.assertEqual(
            record["Flags"],
            "Ambiguous identifier; Invalid unit; Transcription mismatch",
        )

    def test_does_not_corroborate_unit_from_a_year_or_identifier_suffix(self):
        year = self.transcription(
            "box-1",
            "1970",
            ("1974.0056",),
            ("Staff 1970s",),
            identifiers=("1974.0056",),
        )
        slash_identifier = self.transcription(
            "box-2",
            "9",
            ("96/9",),
            (),
            identifiers=("96/9",),
        )

        self.assertEqual(map_catalogue_record(year, "year.png")["Node Title"], "")
        self.assertEqual(
            map_catalogue_record(slash_identifier, "slash.png")["Node Title"], ""
        )

    def test_accepts_an_identifier_corroborated_by_handwriting(self):
        transcription = self.transcription(
            "box-1",
            "7",
            ("Unit 7",),
            ("1974.0056",),
            identifiers=("1974.0056",),
        )

        record = map_catalogue_record(transcription, "box.png")

        self.assertEqual(record["Node Title"], "1974.0056 Unit 0007")

    def test_accepts_older_handwritten_unit_formats(self):
        for handwritten in (("No. 30",), ("30 C1",)):
            with self.subTest(handwritten=handwritten):
                transcription = self.transcription(
                    "box-1",
                    "30",
                    ("1974.0056",),
                    handwritten,
                    identifiers=("1974.0056",),
                )

                record = map_catalogue_record(transcription, "box.png")

                self.assertEqual(record["Node Title"], "1974.0056 Unit 0030")

    def test_conflicting_contextual_units_are_blank_and_review_flagged(self):
        transcription = self.transcription(
            "box-1",
            "7",
            ("1974.0056", "Unit 7"),
            ("Box 8",),
            identifiers=("1974.0056",),
        )

        record = map_catalogue_record(transcription, "box.png")

        self.assertEqual(record["Node Title"], "")
        self.assertEqual(record["Flags"], "Invalid unit; Transcription mismatch")

    def test_excludes_identity_only_handwriting_from_description(self):
        transcription = self.transcription(
            "box-1",
            "30",
            ("Unit 30",),
            ("1974.0056", "30", "30 C1"),
            identifiers=("1974.0056",),
            auxiliary=("Fragile",),
        )

        record = map_catalogue_record(transcription, "box.png")

        self.assertEqual(record["Node Title"], "1974.0056 Unit 0030")
        self.assertEqual(record["Description"], "30 C1; Fragile")

    def test_existing_model_review_state_reaches_the_controlled_export_flag(self):
        transcription = self.transcription(
            "box-1",
            "7",
            ("1974.0056", "Unit 7"),
            ("[illegible]",),
            identifiers=("1974.0056",),
            needs_review=True,
        )

        record = map_catalogue_record(transcription, "box.png")

        self.assertEqual(record["Node Title"], "1974.0056 Unit 0007")
        self.assertEqual(record["Flags"], "Transcription mismatch")

    def test_escapes_formula_prefixes_in_user_controlled_cells(self):
        transcription = self.transcription(
            "box-1",
            "1",
            ("1974.0056", "Unit 1"),
            ('=HYPERLINK("https://example.test")',),
            auxiliary=("  +SUM(1,1)",),
        )

        record = map_catalogue_record(transcription, "@box.png")

        self.assertEqual(
            record["Description"],
            '\'=HYPERLINK("https://example.test"); +SUM(1,1)',
        )
        self.assertEqual(record["Image"], "'@box.png")

    def test_joins_embedded_model_line_breaks_with_semicolons(self):
        transcription = self.transcription(
            "box-1",
            "1",
            ("1974.0056", "Unit 1"),
            ("First line\nSecond line",),
            auxiliary=("Third line\r\nFourth line",),
        )

        record = map_catalogue_record(transcription, "box.png")

        self.assertEqual(
            record["Description"],
            "First line; Second line; Third line; Fourth line",
        )

    def test_rejects_review_images_on_another_windows_drive(self):
        transcription = self.transcription(
            "box-1",
            "1",
            ("1974.0056", "Unit 1"),
            (),
        )

        with (
            patch("uma_bct.catalogue.os.path.relpath", side_effect=ValueError),
            self.assertRaisesRegex(ValueError, "same drive"),
        ):
            map_catalogue_record(
                transcription,
                "D:/crops/box.png",
                catalogue_path="C:/output/catalogue.csv",
            )

    @staticmethod
    def transcription(
        box_id: str,
        unit: str,
        typed: tuple[str, ...],
        handwritten: tuple[str, ...],
        confidence: int = 90,
        *,
        identifiers: tuple[str, ...] = (),
        auxiliary: tuple[str, ...] = (),
        needs_review: bool = False,
    ) -> BoxTranscription:
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
            label=Label(box_id, unit, typed, record_identifiers=identifiers),
            text_lines=text_lines,
            confidence=confidence,
            needs_review=needs_review,
            auxiliary_annotations=auxiliary,
        )


class CatalogueExportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_flags_duplicates_without_removing_physical_rows_and_writes_csv(self):
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
        self.assertEqual(
            written["Image"].tolist(),
            ["../crops/box-a.png", "../crops/box-b.png"],
        )

    def test_writes_xlsx_with_the_same_client_contract(self):
        output_path = self.root / "catalogue.xlsx"

        export_catalogue(
            [self.result("shelf-a", "box-a", handwritten=('Māori "records"',))],
            output_path,
        )
        written = pd.read_excel(output_path, keep_default_na=False)

        self.assertEqual(list(written.columns), OUTPUT_COLUMNS)
        self.assertEqual(written.loc[0, "Node ID"], "")
        self.assertEqual(written.loc[0, "Node Type"], "Unit")
        self.assertEqual(written.loc[0, "Node Title"], "1974.0056 Unit 0001")
        self.assertEqual(written.loc[0, "Description"], 'Māori "records"')
        self.assertEqual(written.loc[0, "Date Range"], "")

    def test_applies_configurable_low_confidence_threshold(self):
        results = [self.result("shelf-a", "box-a", confidence=79)]

        at_default = catalogue_dataframe(results)
        at_custom_threshold = catalogue_dataframe(results, minimum_confidence=70)

        self.assertEqual(at_default.loc[0, "Flags"], "Low confidence")
        self.assertEqual(at_custom_threshold.loc[0, "Flags"], "")

    def test_flags_rows_produced_after_detector_fallback(self):
        original = self.result("shelf-a", "box-a")
        fallback = ImageResult(
            input_image=original.input_image,
            detector="qwen",
            transcriber=original.transcriber,
            boxes=original.boxes,
            transcriptions=original.transcriptions,
            requested_detector="qwen",
            actual_detector="opencv",
            fallback_path=("qwen", "opencv"),
        )

        dataframe = catalogue_dataframe([fallback])

        self.assertEqual(dataframe.loc[0, "Flags"], "Fallback used")

    def test_sorts_complete_titles_and_keeps_incomplete_rows_last_stably(self):
        results = [
            self.result("shelf-z", "box-z", identifier="1997.0138", unit="2"),
            self.result("shelf-missing-a", "box-missing-a", identifier="", unit="1"),
            self.result("shelf-a", "box-a", identifier="1974.0029", unit="12"),
            self.result("shelf-missing-b", "box-missing-b", identifier="", unit="2"),
        ]

        dataframe = catalogue_dataframe(results)

        self.assertEqual(
            dataframe["Node Title"].tolist(),
            ["1974.0029 Unit 0012", "1997.0138 Unit 0002", "", ""],
        )
        self.assertEqual(
            [Path(value).name for value in dataframe["Image"].tolist()],
            ["box-a.png", "box-z.png", "box-missing-a.png", "box-missing-b.png"],
        )

    def test_empty_batch_still_writes_the_required_headers(self):
        output_path = self.root / "empty.csv"

        dataframe = export_catalogue([], output_path)
        written = pd.read_csv(output_path, keep_default_na=False)

        self.assertTrue(dataframe.empty)
        self.assertEqual(list(written.columns), OUTPUT_COLUMNS)

    def test_rejects_unsupported_output_extension(self):
        with self.assertRaisesRegex(ValueError, "Unsupported output extension"):
            export_catalogue([], self.root / "catalogue.json")

    def test_preserves_existing_destination_if_atomic_write_fails(self):
        output_path = self.root / "catalogue.csv"
        output_path.write_text("existing\n", encoding="utf-8")

        with (
            patch.object(pd.DataFrame, "to_csv", side_effect=OSError("disk full")),
            self.assertRaisesRegex(OSError, "disk full"),
        ):
            export_catalogue([self.result("shelf-a", "box-a")], output_path)

        self.assertEqual(output_path.read_text(encoding="utf-8"), "existing\n")
        self.assertEqual(list(self.root.glob(".catalogue-*")), [])

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

    def result(
        self,
        shelf_name: str,
        box_id: str,
        confidence: int = 90,
        *,
        identifier: str = "1974.0056",
        unit: str = "1",
        handwritten: tuple[str, ...] = ("Correspondence",),
    ) -> ImageResult:
        input_image = InputImage(self.root / f"{shelf_name}.jpg", 1000, 500)
        box = Box(
            box_id=box_id,
            source_image=input_image,
            bounding_box=BoundingBox((10, 20), (110, 220)),
            image_file_path=self.root / "crops" / f"{box_id}.png",
            position="top-left",
            detector_confidence=0.9,
        )
        typed = ((identifier,) if identifier else ()) + (f"Unit {unit}",)
        transcription = CatalogueMappingTest.transcription(
            box_id,
            unit,
            typed,
            handwritten,
            confidence,
            identifiers=(identifier,) if identifier else (),
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
