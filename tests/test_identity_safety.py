"""Reject partial claims from compound unit labels without losing their text."""

import unittest

from helpers import make_box, make_transcription

from uma_bct.catalogue import map_catalogue_record
from uma_bct.identity import resolve_unit_claim
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
