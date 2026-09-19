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
from uma_bct.export import catalogue_records, export_catalogue


# Group the catalogue rules checks.
class CatalogueRulesTest(unittest.TestCase):
    # Check that the code builds titles for every supported identifier.
    def test_builds_titles_for_every_supported_identifier(self):
        for identifier in ("1960.0009", "96/9", "UMA-AC-000000330", "UMA-SRE-20160038"):
            with self.subTest(identifier=identifier):
                box = make_box()
                transcription = make_transcription(
                    box, identifiers=(identifier,), unit="7", typed=(f"Record ID: {identifier}", "Unit 7")
                )
                record = map_catalogue_record(transcription, box.image_file_path)
                self.assertEqual(list(record)[:5], ["Node ID", "Node Type", "Node Title", "Description", "Date Range"])
                self.assertEqual(record["Node Type"], "Unit")
                self.assertEqual(record["Node Title"], f"{identifier} Unit 0007")
                self.assertEqual(record["Node ID"], "")
                self.assertEqual(record["Date Range"], "")

    # Check that the code uses first identifier and flags multiple.
    def test_uses_first_identifier_and_flags_multiple(self):
        box = make_box()
        transcription = make_transcription(
            box, identifiers=("1990.0001", "UMA-SRE-20160038"), typed=("1990.0001; UMA-SRE-20160038", "Unit 30"), unit="30"
        )
        record = map_catalogue_record(transcription, box.image_file_path)
        self.assertEqual(record["Node Title"], "1990.0001 Unit 0030")
        self.assertIn("Multiple identifiers", record["Review Flags"])

    # Check that the code blanks missing or uncorroborated identity.
    def test_blanks_missing_or_uncorroborated_identity(self):
        box = make_box()
        cases = (
            (make_transcription(box, identifiers=(), typed=("Record origin", "Unit 4")), "Missing identifier"),
            (make_transcription(box, identifiers=("1960.0009",), typed=("1960.0010", "Unit 4")), "Ambiguous identifier"),
            (
                make_transcription(box, identifiers=("1960.009",), typed=("Record ID: 1960.009", "Unit 4")),
                "Ambiguous identifier",
            ),
        )
        for transcription, flag in cases:
            with self.subTest(flag=flag):
                record = map_catalogue_record(transcription, box.image_file_path)
                self.assertEqual(record["Node Title"], "")
                self.assertIn(flag, record["Review Flags"])

    # Check that the code does not promote a slash date from prose.
    def test_does_not_promote_a_slash_date_from_prose(self):
        box = make_box()
        transcription = make_transcription(box, identifiers=("12/5",), typed=("Meeting 12/5", "Unit 4"))
        record = map_catalogue_record(transcription, box.image_file_path)
        self.assertEqual(record["Node Title"], "")
        self.assertIn("Transcription mismatch", record["Review Flags"])

    # Check that the code requires one unambiguous unit.
    def test_requires_one_unambiguous_unit(self):
        box = make_box()
        accepted = make_transcription(box, unit="7", typed=("1960.0009",), handwritten=("7",))
        conflicting = make_transcription(box, unit="7", typed=("1960.0009", "Unit 7", "Unit 8"))
        self.assertEqual(map_catalogue_record(accepted, box.image_file_path)["Node Title"], "1960.0009 Unit 0007")
        rejected = map_catalogue_record(conflicting, box.image_file_path)
        self.assertEqual(rejected["Node Title"], "")
        self.assertIn("Invalid unit", rejected["Review Flags"])

    # Check that description excludes identity and escapes formulas.
    def test_description_excludes_identity_and_escapes_formulas(self):
        box = make_box()
        transcription = make_transcription(box, handwritten=("1960.0009", "=SUM(A1:A2)"), auxiliary=("Fragile",))
        record = map_catalogue_record(transcription, box.image_file_path)
        self.assertEqual(record["Description"], "'=SUM(A1:A2); Fragile")

    # Check that untrusted crop text is escaped and only managed paths become formulas.
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

    # Check that the code preserves and flags duplicate physical boxes.
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

    # Check that low confidence threshold is strict.
    def test_low_confidence_threshold_is_strict(self):
        box = make_box()
        result = make_result(BoxResult(box, make_transcription(box, confidence=70)))
        below_threshold = catalogue_records([result], minimum_confidence=71)
        at_threshold = catalogue_records([result], minimum_confidence=70)

        self.assertIn("Low confidence", below_threshold[0]["Review Flags"])
        self.assertNotIn("Low confidence", at_threshold[0]["Review Flags"])


# Group the CSV export checks.
class CsvExportTest(unittest.TestCase):
    # Check that the code writes exact headers in physical order with relative crop images.
    def test_writes_exact_headers_in_physical_order_with_relative_crop_images(self):
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
                    "Node ID",
                    "Node Type",
                    "Node Title",
                    "Description",
                    "Date Range",
                    "Transcription Confidence (%)",
                    "Review Flags",
                    "Cropped Box Image Link",
                ],
            )
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["Node Title"], "2000.0002 Unit 0002")
            self.assertEqual(rows[0]["Cropped Box Image Link"], "crops/b-001.png")
            self.assertEqual(rows[1]["Node Title"], "")
            self.assertEqual(rows[1]["Cropped Box Image Link"], "crops/b-002.png")
            self.assertEqual(rows[2]["Node Title"], "1990.0001 Unit 0001")
            self.assertEqual(rows[2]["Cropped Box Image Link"], "crops/a-001.png")

    # Check that atomic failure preserves existing CSV.
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

    # Check that the code does not replace an output that appeared during processing.
    def test_does_not_replace_an_output_that_appeared_during_processing(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory).resolve() / "catalogue.csv"
            output.write_text("other process\n", encoding="utf-8")
            box = make_box(crop_path=output.parent / "crops/box.png")
            result = make_result(BoxResult(box, make_transcription(box)))

            with self.assertRaisesRegex(FileExistsError, "already exists"):
                export_catalogue([result], output, replace_existing=False)
            self.assertEqual(output.read_text(encoding="utf-8"), "other process\n")

    # Check that the code does not replace an output appearing at the publication syscall.
    def test_does_not_replace_an_output_appearing_at_the_publication_syscall(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory).resolve() / "catalogue.csv"
            box = make_box(crop_path=output.parent / "crops/box.png")
            result = make_result(BoxResult(box, make_transcription(box)))

            # Create the competing destination used by this race check.
            def late_destination(source, destination):
                destination.write_text("late output\n", encoding="utf-8")
                return rename_exclusive(source, destination)

            with (
                patch("uma_bct.atomic.rename_exclusive", side_effect=late_destination),
                self.assertRaisesRegex(FileExistsError, "already exists"),
            ):
                export_catalogue([result], output, replace_existing=False)

            self.assertEqual(output.read_text(encoding="utf-8"), "late output\n")

    # Check that CSV is the only output format.
    def test_csv_is_the_only_output_format(self):
        with self.assertRaisesRegex(ValueError, "must end in .csv"):
            export_catalogue([], "catalogue.xlsx")


if __name__ == "__main__":
    unittest.main()
