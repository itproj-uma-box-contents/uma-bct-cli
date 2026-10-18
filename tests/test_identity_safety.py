"""Reject partial claims from compound unit labels without losing their text."""

import unittest

from helpers import make_box, make_transcription

from uma_bct.catalogue import map_catalogue_record
from uma_bct.identity import identity_only_line, resolve_unit_claim
from uma_bct.transcriptions import ObservedBox, deduplicate


class CompoundUnitTest(unittest.TestCase):
    def test_compound_units_stay_visible_and_never_export_a_partial_number(self):
        labels = ("Unit 7/8", "Unit 7-8", "Unit 7–8", "Unit 7.5", "Unit 7, 8",
                  "Unit 7 & 8", "Unit 7 and 8", "Unit 7 to 8", "Unit 7 and Unit 8")
        box = make_box()
        for label in labels:
            with self.subTest(label=label):
                typed = ("1960.0009", label)
                row = map_catalogue_record(make_transcription(box, unit="7", typed=typed), "crop.png")
                self.assertEqual(row["Unit Number"], "")
                self.assertIn(label, row["Unit Type"])
                self.assertIn("Invalid unit", row["Review Flags"])
                self.assertEqual(resolve_unit_claim("", typed, ()), ("", True))

    def test_sequence_totals_and_ordinary_annotations_still_corroborate_one_unit(self):
        box = make_box()
        for label in ("Box 7 of 8", "Unit 7, damaged", "Unit 7 - correspondence", "Unit 7."):
            with self.subTest(label=label):
                row = map_catalogue_record(make_transcription(box, unit="7", typed=("1960.0009", label)), "crop.png")
                self.assertEqual(row["Unit Number"], "0007")
                self.assertNotIn("Invalid unit", row["Review Flags"])

    def test_identifier_after_a_unit_is_not_misread_as_another_unit(self):
        box = make_box()
        for label in ("Unit 7 - 1960.0009", "Unit 7, 1960.0009", "Unit 7 / 1960.0009"):
            with self.subTest(label=label):
                row = map_catalogue_record(make_transcription(box, unit="7", typed=(label,)), "crop.png")
                self.assertEqual(row["Unit Number"], "0007")
                self.assertNotIn("Invalid unit", row["Review Flags"])

    def test_ambiguous_compound_units_do_not_merge_separate_crop_readings(self):
        first = make_box(box_id="first")
        second = make_box(box_id="second")
        a = make_transcription(first, unit="7", typed=("1960.0009", "Unit 7/8"))
        b = make_transcription(second, unit="7", typed=("1960.0009", "Unit 7"))
        self.assertEqual(len(deduplicate([ObservedBox(first, a, 1), ObservedBox(second, b, 2)])), 2)


class IdentityTextTest(unittest.TestCase):
    def test_identity_filter_preserves_annotations_and_conflicting_numbers(self):
        identifiers = ("1960.0009", "BHP")
        for line, expected, legacy in (
            ("4", True, True),
            ("7", False, True),
            ("1950", False, True),
            ("Fragile Box 4", False, True),
            ("Mould on Unit 4", False, True),
            ("1960.0009 Unit 4", True, True),
            ("BHP Unit 4", True, True),
            ("Box 7", False, False),
            ("Fragile Box 7", False, False),
        ):
            with self.subTest(line=line):
                self.assertEqual(identity_only_line(line, identifiers, "4"), expected)
                self.assertEqual(identity_only_line(line, identifiers, "4", legacy_filter=True), legacy)

    def test_date_text_excludes_only_explicit_corroborated_identity(self):
        box = make_box()
        handwritten = (
            "Unit 1950\n1950.1960",
            "Record ID: 1950.1960",
            "1950.1960 Unit 1950",
            "Record ID: 1950.1960 Box 4",
            "1950",
            "Files from 1950.1960",
            "Files from 1950.1960 Unit 1950",
            "Fragile Box 1950",
            "Box 1950 of 1952",
            "Unit 1950/1951",
        )
        transcription = make_transcription(
            box, identifiers=("1950.1960",), typed=("1950.1960", "Unit 4"), handwritten=handwritten,
        )

        self.assertEqual(transcription.get_date_text(),
                         "1950\nFiles from 1950.1960\nFiles from 1950.1960 Unit 1950\n"
                         "Fragile Box 1950\nUnit 1950/1951")
        self.assertEqual(transcription.get_handwritten_text(), "\n".join(handwritten))
        self.assertEqual(transcription.handwritten_lines, handwritten)

    def test_date_text_keeps_identifiers_without_both_kinds_of_evidence(self):
        box = make_box()
        for identifiers, typed in (
            ((), ("1950.1960", "Unit 4")),
            (("1950.1960",), ("1951.1961", "Unit 4")),
        ):
            with self.subTest(identifiers=identifiers, typed=typed):
                transcription = make_transcription(box, identifiers=identifiers, typed=typed, handwritten=("1950.1960",))
                self.assertEqual(transcription.get_date_text(), "1950.1960")

    def test_date_text_never_promotes_a_bare_year_to_identity(self):
        box = make_box()
        transcription = make_transcription(
            box, identifiers=("1950",), typed=("1950 Unit 4",), handwritten=("1950", "1950 Unit 4"),
        )
        self.assertEqual(transcription.get_date_text(), "1950\n1950 Unit 4")
