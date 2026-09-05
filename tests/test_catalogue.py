import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from helpers import make_box, make_result, make_transcription

from uma_bct.catalogue import map_catalogue_record
from uma_bct.constants import OUTPUT_COLUMNS
from uma_bct.domain import BoxResult
from uma_bct.export import catalogue_records, export_catalogue


class CatalogueRulesTest(unittest.TestCase):
    def test_builds_titles_for_every_supported_identifier(self):
        for identifier in (
            "1960.0009",
            "96/9",
            "UMA-AC-000000330",
            "UMA-SRE-20160038",
        ):
            with self.subTest(identifier=identifier):
                box = make_box()
                transcription = make_transcription(
                    box,
                    identifiers=(identifier,),
                    unit="7",
                    typed=(f"Record ID: {identifier}", "Unit 7"),
                )
                record = map_catalogue_record(transcription, box.image_file_path)
                self.assertEqual(
                    list(record)[:5],
                    ["Node ID", "Node Type", "Node Title", "Description", "Date Range"],
                )
                self.assertEqual(record["Node Type"], "Unit")
                self.assertEqual(record["Node Title"], f"{identifier} Unit 0007")
                self.assertEqual(record["Node ID"], "")
                self.assertEqual(record["Date Range"], "")

    def test_uses_first_identifier_and_flags_multiple(self):
        box = make_box()
        transcription = make_transcription(
            box,
            identifiers=("1990.0001", "UMA-SRE-20160038"),
            typed=("1990.0001; UMA-SRE-20160038", "Unit 30"),
            unit="30",
        )
        record = map_catalogue_record(transcription, box.image_file_path)
        self.assertEqual(record["Node Title"], "1990.0001 Unit 0030")
        self.assertIn("Multiple identifiers", record["Flags"])

    def test_blanks_missing_or_uncorroborated_identity(self):
        box = make_box()
        cases = (
            (
                make_transcription(
                    box, identifiers=(), typed=("Record origin", "Unit 4")
                ),
                "Missing identifier",
            ),
            (
                make_transcription(
                    box,
                    identifiers=("1960.0009",),
                    typed=("1960.0010", "Unit 4"),
                ),
                "Ambiguous identifier",
            ),
            (
                make_transcription(
                    box,
                    identifiers=("1960.009",),
                    typed=("Record ID: 1960.009", "Unit 4"),
                ),
                "Ambiguous identifier",
            ),
        )
        for transcription, flag in cases:
            with self.subTest(flag=flag):
                record = map_catalogue_record(transcription, box.image_file_path)
                self.assertEqual(record["Node Title"], "")
                self.assertIn(flag, record["Flags"])

    def test_does_not_promote_a_slash_date_from_prose(self):
        box = make_box()
        transcription = make_transcription(
            box,
            identifiers=("12/5",),
            typed=("Meeting 12/5", "Unit 4"),
        )
        record = map_catalogue_record(transcription, box.image_file_path)
        self.assertEqual(record["Node Title"], "")
        self.assertIn("Transcription mismatch", record["Flags"])

    def test_requires_one_unambiguous_unit(self):
        box = make_box()
        accepted = make_transcription(
            box,
            unit="7",
            typed=("1960.0009",),
            handwritten=("7",),
        )
        conflicting = make_transcription(
            box,
            unit="7",
            typed=("1960.0009", "Unit 7", "Unit 8"),
        )
        self.assertEqual(
            map_catalogue_record(accepted, box.image_file_path)["Node Title"],
            "1960.0009 Unit 0007",
        )
        rejected = map_catalogue_record(conflicting, box.image_file_path)
        self.assertEqual(rejected["Node Title"], "")
        self.assertIn("Invalid unit", rejected["Flags"])

    def test_description_excludes_identity_and_escapes_formulas(self):
        box = make_box()
        transcription = make_transcription(
            box,
            handwritten=("1960.0009", "=SUM(A1:A2)"),
            auxiliary=("Fragile",),
        )
        record = map_catalogue_record(transcription, box.image_file_path)
        self.assertEqual(record["Description"], "'=SUM(A1:A2); Fragile")

    def test_preserves_and_flags_duplicate_physical_boxes(self):
        first = make_box(box_id="shelf:001")
        second = make_box(box_id="shelf:002")
        records = catalogue_records(
            [
                make_result(
                    BoxResult(first, make_transcription(first, confidence=70)),
                    BoxResult(second, make_transcription(second)),
                )
            ]
        )
        self.assertEqual(len(records), 2)
        self.assertIn("Potential Duplicate", records[0]["Flags"])
        self.assertIn("Potential Duplicate", records[1]["Flags"])
        self.assertIn("Low confidence", records[0]["Flags"])


class CsvExportTest(unittest.TestCase):
    def test_writes_exact_headers_sorted_records_and_relative_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "outputs" / "catalogue.csv"
            first = make_box(box_id="shelf:001", crop_path=root / "outputs/crops/a.png")
            incomplete = make_box(
                box_id="shelf:002", crop_path=root / "outputs/crops/b.png"
            )
            result = make_result(
                BoxResult(first, make_transcription(first)),
                BoxResult(
                    incomplete,
                    make_transcription(
                        incomplete, identifiers=(), typed=("Unit 2",), unit="2"
                    ),
                ),
            )
            export_catalogue([result], output)

            with output.open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
                self.assertEqual(source.seek(0), 0)
                self.assertEqual(next(csv.reader(source)), OUTPUT_COLUMNS)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["Node Title"], "1960.0009 Unit 0004")
            self.assertEqual(rows[0]["Image"], "crops/a.png")
            self.assertEqual(rows[1]["Node Title"], "")

    def test_atomic_failure_preserves_existing_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "catalogue.csv"
            output.write_text("old\n", encoding="utf-8")
            box = make_box(crop_path=output.parent / "crops/box.png")
            result = make_result(BoxResult(box, make_transcription(box)))
            with (
                patch(
                    "uma_bct.export.csv.DictWriter.writerows",
                    side_effect=OSError("disk full"),
                ),
                self.assertRaisesRegex(OSError, "disk full"),
            ):
                export_catalogue([result], output)
            self.assertEqual(output.read_text(encoding="utf-8"), "old\n")
            self.assertEqual(tuple(output.parent.glob(".uma-bct-catalogue-*")), ())

    def test_csv_is_the_only_output_format(self):
        with self.assertRaisesRegex(ValueError, "must end in .csv"):
            export_catalogue([], "catalogue.xlsx")


if __name__ == "__main__":
    unittest.main()
