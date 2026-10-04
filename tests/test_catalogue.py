import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from helpers import make_box, make_image, make_result, make_transcription

from uma_bct.atomic import rename_exclusive
from uma_bct.catalogue import map_catalogue_record
from uma_bct.constants import OUTPUT_COLUMNS
from uma_bct.domain import BoxResult
from uma_bct.export import catalogue_records, export_catalogue, missing_box_records


class CatalogueRulesTest(unittest.TestCase):
    def test_builds_unit_type_for_known_and_arbitrary_identifiers(self):
        for identifier in ("1960.0009", "96/9", "UMA-AC-000000330", "UMA-SRE-20160038", "BHP"):
            with self.subTest(identifier=identifier):
                box = make_box()
                transcription = make_transcription(box, identifiers=(identifier,), unit="7", typed=(f"{identifier} Unit 7",))
                record = map_catalogue_record(transcription, box.image_file_path)
                self.assertEqual(list(record)[:6], ["Unit Type", "Unit Number", "Description", "Dates", "Earliest Date", "Latest Date"])
                self.assertEqual(record["Unit Type"], identifier)
                self.assertEqual(record["Unit Number"], "0007")
                self.assertEqual(record["Dates"], "No Dates")
                self.assertEqual(record["Earliest Date"], "None")
                self.assertEqual(record["Latest Date"], "None")

    def test_accepts_a_prefix_on_the_line_before_unit_number(self):
        box = make_box()
        transcription = make_transcription(box, identifiers=("BHP",), unit="24", typed=("BHP", "Unit 24"))
        record = map_catalogue_record(transcription, box.image_file_path)
        self.assertEqual(record["Unit Type"], "BHP")
        self.assertEqual(record["Unit Number"], "0024")

    def test_accepts_arbitrary_identifier_prefix_before_unit(self):
        box = make_box()
        identifier = "BHP124-SFDSF33"
        transcription = make_transcription(box, identifiers=(identifier,), unit="30", typed=(f"{identifier} Unit 30",))
        record = map_catalogue_record(transcription, box.image_file_path)
        self.assertEqual(record["Unit Type"], identifier)
        self.assertEqual(record["Unit Number"], "0030")

    def test_builds_unit_type_from_all_printed_label_text(self):
        box = make_box()
        cases = (
            (
                ("HOME, Roderick Weir", "2008.0036", "Unit 7"),
                ("2008.0036",),
                "7",
                "HOME, Roderick Weir 2008.0036",
                "0007",
            ),
            (
                (
                    "THE UNIVERSITY OF MELBOURNE ARCHIVES",
                    "Creator:",
                    "Victorian Chamber of Mines",
                    "107/56",
                    "Box 7 of 9",
                ),
                ("107/56",),
                "7",
                "THE UNIVERSITY OF MELBOURNE ARCHIVES Creator: Victorian Chamber of Mines 107/56",
                "0007",
            ),
            (
                ("BURNET, Sir Frank Macfarlane", "1986.0107 (1989.0034)", "Unit 22"),
                ("1986.0107", "1989.0034"),
                "22",
                "BURNET, Sir Frank Macfarlane 1986.0107 (1989.0034)",
                "0022",
            ),
            (
                (
                    "Printing & Allied Trades",
                    "Employers' Association",
                    "of Victoria",
                    "1975.0064 Unit 8",
                ),
                ("1975.0064",),
                "8",
                "Printing & Allied Trades Employers' Association of Victoria 1975.0064",
                "0008",
            ),
        )
        for typed, identifiers, unit, expected_type, expected_unit in cases:
            with self.subTest(expected_type=expected_type):
                transcription = make_transcription(box, identifiers=identifiers, typed=typed, unit=unit)
                record = map_catalogue_record(transcription, box.image_file_path)
                self.assertEqual(record["Unit Type"], expected_type)
                self.assertEqual(record["Unit Number"], expected_unit)

    def test_preserves_multiple_identifiers_and_flags_review(self):
        box = make_box()
        transcription = make_transcription(
            box, identifiers=("1990.0001", "UMA-SRE-20160038"), typed=("1990.0001; UMA-SRE-20160038", "Unit 30"), unit="30"
        )
        record = map_catalogue_record(transcription, box.image_file_path)
        self.assertEqual(record["Unit Type"], "1990.0001; UMA-SRE-20160038")
        self.assertEqual(record["Unit Number"], "0030")
        self.assertIn("Multiple identifiers", record["Review Flags"])

    def test_identifier_order_mismatch_keeps_printed_unit_type_for_review(self):
        box = make_box()
        identifiers = ("1990.0001", "1990.0002")
        accepted = make_transcription(box, identifiers=identifiers, typed=("1990.0001; 1990.0002", "Unit 4"))
        accepted_record = map_catalogue_record(accepted, box.image_file_path)
        self.assertEqual(accepted_record["Unit Type"], "1990.0001; 1990.0002")
        self.assertNotIn("Ambiguous identifier", accepted_record["Review Flags"])

        for typed, handwritten, expected_type in (
            (("1990.0002; 1990.0001", "Unit 4"), (), "1990.0002; 1990.0001"),
            (("1990.0001; 1990.0002", "Unit 4"), ("1990.0002; 1990.0001",), "1990.0001; 1990.0002"),
        ):
            with self.subTest(typed=typed, handwritten=handwritten):
                transcription = make_transcription(
                    box, identifiers=identifiers, typed=typed, handwritten=handwritten
                )
                record = map_catalogue_record(transcription, box.image_file_path)
                self.assertEqual(record["Unit Type"], expected_type)
                self.assertEqual(record["Unit Number"], "0004")
                self.assertIn("Ambiguous identifier", record["Review Flags"])
                self.assertIn("Transcription mismatch", record["Review Flags"])

    def test_preserves_printed_unit_type_when_identifier_evidence_needs_review(self):
        box = make_box()
        cases = (
            (
                make_transcription(box, identifiers=("1960.0009",), typed=("1960.0010", "Unit 4")),
                "1960.0010",
            ),
            (
                make_transcription(box, identifiers=("1960.009",), typed=("Record ID: 1960.009", "Unit 4")),
                "Record ID: 1960.009",
            ),
        )
        for transcription, expected_type in cases:
            with self.subTest(expected_type=expected_type):
                record = map_catalogue_record(transcription, box.image_file_path)
                self.assertEqual(record["Unit Type"], expected_type)
                self.assertEqual(record["Unit Number"], "0004")
                self.assertIn("Ambiguous identifier", record["Review Flags"])
                self.assertIn("Transcription mismatch", record["Review Flags"])

    def test_flags_a_malformed_dotted_identifier_on_the_unit_line(self):
        box = make_box()
        transcription = make_transcription(box, identifiers=("1960.009",), typed=("1960.009 Unit 4",), unit="4")
        record = map_catalogue_record(transcription, box.image_file_path)
        self.assertEqual(record["Unit Type"], "1960.009")
        self.assertEqual(record["Unit Number"], "0004")
        self.assertIn("Ambiguous identifier", record["Review Flags"])
        self.assertIn("Transcription mismatch", record["Review Flags"])

    def test_accepts_printed_unit_type_without_a_structured_identifier(self):
        box = make_box()
        transcription = make_transcription(box, identifiers=(), typed=("Record origin", "Unit 4"))
        record = map_catalogue_record(transcription, box.image_file_path)
        self.assertEqual(record["Unit Type"], "Record origin")
        self.assertEqual(record["Unit Number"], "0004")
        self.assertNotIn("Missing identifier", record["Review Flags"])

    def test_flags_a_missing_unit_type_when_the_label_only_has_a_unit_number(self):
        box = make_box()
        transcription = make_transcription(box, identifiers=(), typed=("Unit 4",))
        record = map_catalogue_record(transcription, box.image_file_path)
        self.assertEqual(record["Unit Type"], "")
        self.assertEqual(record["Unit Number"], "0004")
        self.assertIn("Missing identifier", record["Review Flags"])

    def test_does_not_promote_a_slash_date_from_prose(self):
        box = make_box()
        transcription = make_transcription(box, identifiers=("12/5",), typed=("Meeting 12/5", "Unit 4"))
        record = map_catalogue_record(transcription, box.image_file_path)
        self.assertEqual(record["Unit Type"], "Meeting 12/5")
        self.assertEqual(record["Unit Number"], "0004")
        self.assertIn("Transcription mismatch", record["Review Flags"])

    def test_requires_one_unambiguous_unit(self):
        box = make_box()
        accepted = make_transcription(box, unit="7", typed=("1960.0009",), handwritten=("7",))
        conflicting = make_transcription(box, unit="7", typed=("1960.0009", "Unit 7", "Unit 8"))
        accepted_record = map_catalogue_record(accepted, box.image_file_path)
        self.assertEqual(accepted_record["Unit Type"], "1960.0009")
        self.assertEqual(accepted_record["Unit Number"], "0007")
        rejected = map_catalogue_record(conflicting, box.image_file_path)
        self.assertEqual(rejected["Unit Type"], "1960.0009")
        self.assertEqual(rejected["Unit Number"], "")
        self.assertIn("Invalid unit", rejected["Review Flags"])

    def test_description_excludes_identity_and_escapes_formulas(self):
        box = make_box()
        transcription = make_transcription(box, handwritten=("1960.0009", "=SUM(A1:A2)"), auxiliary=("Fragile",))
        record = map_catalogue_record(transcription, box.image_file_path)
        self.assertEqual(record["Description"], "'=SUM(A1:A2); Fragile")

    def test_untrusted_crop_text_is_escaped_and_only_managed_paths_become_formulas(self):
        box = make_box(crop_path=Path("=OPEN-UNTRUSTED.png"))
        transcription = make_transcription(box)
        direct = map_catalogue_record(transcription, box.image_file_path)
        self.assertEqual(direct["Cropped Box Image Link"], "'=OPEN-UNTRUSTED.png")

        managed = map_catalogue_record(
            transcription,
            Path("batch/Box Images/=source - Box 01.png"),
            catalogue_path=Path("batch/Catalogue.csv"),
            trusted_crop_link=True,
        )
        self.assertEqual(
            managed["Cropped Box Image Link"],
            '=HYPERLINK("Box Images/=source - Box 01.png","=source - Box 01.png")',
        )
        with self.assertRaisesRegex(ValueError, "inside Box Images"):
            map_catalogue_record(
                transcription,
                Path("batch/outside.png"),
                catalogue_path=Path("batch/Catalogue.csv"),
                trusted_crop_link=True,
            )

    def test_preserves_and_flags_duplicate_physical_boxes(self):
        first = make_box(box_id="shelf:001")
        second = make_box(box_id="shelf:002")
        records = catalogue_records(
            [
                make_result(
                    BoxResult(first, make_transcription(first, confidence=70)), BoxResult(second, make_transcription(second))
                )
            ]
        )
        self.assertEqual(len(records), 2)
        self.assertIn("Potential Duplicate", records[0]["Review Flags"])
        self.assertIn("Potential Duplicate", records[1]["Review Flags"])
        self.assertIn("Low confidence", records[0]["Review Flags"])

    def test_adds_blank_rows_for_missing_boxes(self):
        first = make_box(box_id="shelf:001")
        third = make_box(box_id="shelf:003")
        records = catalogue_records(
            [
                make_result(
                    BoxResult(
                        first, make_transcription(first, identifiers=("1974.0056",), typed=("1974.0056", "Unit 1"), unit="1")
                    ),
                    BoxResult(
                        third, make_transcription(third, identifiers=("1974.0056",), typed=("1974.0056", "Unit 3"), unit="3")
                    ),
                )
            ]
        )

        missing = records[1]
        self.assertEqual(missing["Unit Type"], "1974.0056")
        self.assertEqual(missing["Unit Number"], "0002")
        self.assertEqual(missing["Review Notes"], "Missing Box")
        self.assertEqual(missing["Crop Image"], "")

    def test_detects_gaps_within_each_prefix_and_can_be_disabled(self):
        first_a = make_box(box_id="a:001")
        third_a = make_box(box_id="a:003")
        first_b = make_box(box_id="b:001")
        third_b = make_box(box_id="b:003")
        results = [
            make_result(
                BoxResult(
                    first_b, make_transcription(first_b, identifiers=("2000.0002",), typed=("2000.0002", "Unit 1"), unit="1")
                ),
                BoxResult(
                    third_a, make_transcription(third_a, identifiers=("1990.0001",), typed=("1990.0001", "Unit 3"), unit="3")
                ),
                BoxResult(
                    first_a, make_transcription(first_a, identifiers=("1990.0001",), typed=("1990.0001", "Unit 1"), unit="1")
                ),
                BoxResult(
                    third_b, make_transcription(third_b, identifiers=("2000.0002",), typed=("2000.0002", "Unit 3"), unit="3")
                ),
            )
        ]

        records = catalogue_records(results)
        self.assertEqual(
            [(record["Unit Type"], record["Unit Number"]) for record in records],
            [
                ("1990.0001", "0001"),
                ("1990.0001", "0002"),
                ("1990.0001", "0003"),
                ("2000.0002", "0001"),
                ("2000.0002", "0002"),
                ("2000.0002", "0003"),
            ],
        )
        self.assertEqual([record["Review Notes"] for record in records], ["", "Missing Box", "", "", "Missing Box", ""])

        without_detection = catalogue_records(results, detect_missing_boxes=False)
        self.assertEqual(
            [(record["Unit Type"], record["Unit Number"]) for record in without_detection],
            [("1990.0001", "0001"), ("1990.0001", "0003"), ("2000.0002", "0001"), ("2000.0002", "0003")],
        )

    def test_gap_suggestions_group_case_insensitively_and_skip_large_gaps(self):
        observed = (
            {"Unit Type": "home", "Unit Number": "0003"},
            {"Unit Type": "HOME", "Unit Number": "0001"},
            {"Unit Type": "HOME", "Unit Number": "0200"},
        )
        suggestions = missing_box_records(observed)
        self.assertEqual(
            [(row["Unit Type"], row["Unit Number"], row["Review Flags"]) for row in suggestions],
            [("HOME", "0002", "Missing Box")],
        )

    def test_low_confidence_threshold_is_strict(self):
        box = make_box()
        result = make_result(BoxResult(box, make_transcription(box, confidence=70)))
        below_threshold = catalogue_records([result], minimum_confidence=71)
        at_threshold = catalogue_records([result], minimum_confidence=70)

        self.assertIn("Low confidence", below_threshold[0]["Review Flags"])
        self.assertNotIn("Low confidence", at_threshold[0]["Review Flags"])


class CsvExportTest(unittest.TestCase):
    def test_writes_exact_headers_in_sorted_order_with_relative_crop_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            output = root / "outputs" / "catalogue.csv"
            shelf_b = make_image(root / "inputs/shelf-b.jpg")
            shelf_a = make_image(root / "inputs/shelf-a.jpg")
            first_image = make_box(image=shelf_b, box_id="shelf-b:001", crop_path=root / "outputs/crops/b-001.png")
            first_image_second_crop = make_box(
                image=shelf_b, box_id="shelf-b:002", crop_path=root / "outputs/crops/b-002.png"
            )
            second_image = make_box(image=shelf_a, box_id="shelf-a:001", crop_path=root / "outputs/crops/a-001.png")
            first_result = make_result(
                BoxResult(
                    first_image,
                    make_transcription(first_image, identifiers=("2000.0002",), typed=("2000.0002", "Unit 2"), unit="2"),
                ),
                BoxResult(
                    first_image_second_crop,
                    make_transcription(first_image_second_crop, identifiers=(), typed=("Unit 3",), unit="3"),
                ),
            )
            second_result = make_result(
                BoxResult(
                    second_image,
                    make_transcription(second_image, identifiers=("1990.0001",), typed=("1990.0001", "Unit 1"), unit="1"),
                )
            )
            export_catalogue([first_result, second_result], output)

            with output.open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
                self.assertEqual(source.seek(0), 0)
                self.assertEqual(next(csv.reader(source)), OUTPUT_COLUMNS)
            self.assertEqual(
                OUTPUT_COLUMNS,
                [
                    "Unit Type",
                    "Unit Number",
                    "Description",
                    "Dates",
                    "Earliest Date",
                    "Latest Date",
                    "Transcription Confidence (%)",
                    "Review Flags",
                    "Cropped Box Image Link",
                ],
            )
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["Unit Type"], "1990.0001")
            self.assertEqual(rows[0]["Unit Number"], "0001")
            self.assertEqual(rows[0]["Cropped Box Image Link"], "crops/a-001.png")
            self.assertEqual(rows[1]["Unit Type"], "2000.0002")
            self.assertEqual(rows[1]["Unit Number"], "0002")
            self.assertEqual(rows[1]["Cropped Box Image Link"], "crops/b-001.png")
            self.assertEqual(rows[2]["Unit Type"], "")
            self.assertEqual(rows[2]["Unit Number"], "0003")
            self.assertEqual(rows[2]["Cropped Box Image Link"], "crops/b-002.png")

    def test_atomic_failure_preserves_existing_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory).resolve() / "catalogue.csv"
            output.write_text("old\n", encoding="utf-8")
            box = make_box(crop_path=output.parent / "crops/box.png")
            result = make_result(BoxResult(box, make_transcription(box)))
            with (
                patch("uma_bct.export.csv.DictWriter.writerows", side_effect=OSError("disk full")),
                self.assertRaisesRegex(OSError, "disk full"),
            ):
                export_catalogue([result], output)
            self.assertEqual(output.read_text(encoding="utf-8"), "old\n")
            self.assertEqual(tuple(output.parent.glob(".uma-bct-catalogue-*")), ())

    def test_does_not_replace_an_output_that_appeared_during_processing(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory).resolve() / "catalogue.csv"
            output.write_text("other process\n", encoding="utf-8")
            box = make_box(crop_path=output.parent / "crops/box.png")
            result = make_result(BoxResult(box, make_transcription(box)))

            with self.assertRaisesRegex(FileExistsError, "already exists"):
                export_catalogue([result], output, replace_existing=False)
            self.assertEqual(output.read_text(encoding="utf-8"), "other process\n")

    def test_does_not_replace_an_output_appearing_at_the_publication_syscall(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory).resolve() / "catalogue.csv"
            box = make_box(crop_path=output.parent / "crops/box.png")
            result = make_result(BoxResult(box, make_transcription(box)))

            def late_destination(source, destination):
                destination.write_text("late output\n", encoding="utf-8")
                return rename_exclusive(source, destination)

            with (
                patch("uma_bct.atomic.rename_exclusive", side_effect=late_destination),
                self.assertRaisesRegex(FileExistsError, "already exists"),
            ):
                export_catalogue([result], output, replace_existing=False)

            self.assertEqual(output.read_text(encoding="utf-8"), "late output\n")

    def test_csv_is_the_only_output_format(self):
        with self.assertRaisesRegex(ValueError, "must end in .csv"):
            export_catalogue([], "catalogue.xlsx")


if __name__ == "__main__":
    unittest.main()
