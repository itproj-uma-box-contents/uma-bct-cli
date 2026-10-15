"""Keep malformed dates from hiding successful box transcriptions."""

import copy
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import Mock

import test_pipeline
from helpers import make_box, make_result, make_transcription, read_excel

from uma_bct.adapters.qwen_date_detector import QwenDateDetector, get_result_schema
from uma_bct.adapters.regex_date_detector import RegexDateDetector
from uma_bct.domain import BoxResult
from uma_bct.export import catalogue_records, export_catalogue
from uma_bct.managed_batch import deserialize_result, serialize_result
from uma_bct.ports import ModelResponse
from uma_bct.util.date_util import Date
from uma_bct.verification import reading, reconcile, verify_image


def model_date(**changes):
    value = {"raw_text": "1950", "starting_index": 0, "start_date": {"year": 1950}, "end_date": {}}
    return value | changes


class DateValidationTest(unittest.TestCase):
    def extract(self, entries, text="1950"):
        client = Mock()
        client.query.return_value = ModelResponse(value={"dates": entries})
        return QwenDateDetector(client).extract_dates(text)

    def test_invalid_components_are_not_treated_as_missing(self):
        for field in ("day", "month", "year"):
            for value in (0, -1, True, False, "1", 1.5):
                with self.subTest(field=field, value=value):
                    self.assertFalse(Date(**{field: value}).is_valid())
        self.assertTrue(Date(day=29, month=2).is_valid())
        self.assertTrue(Date().is_valid())

    def test_invalid_model_dates_are_rejected_without_echoing_private_text(self):
        bad_entries = (
            model_date(start_date={"day": 31, "month": 2, "year": 1950}),
            model_date(start_date={"year": True}),
            model_date(start_date={"year": "1950"}),
            model_date(start_date={"year": 0}),
            model_date(start_date={"year": 2051}),
            model_date(start_date={}),
            model_date(start_date={"year": 1950, "extra": 1}),
            model_date(end_date={"year": 1949}),
            model_date(starting_index=-1),
            model_date(starting_index=True),
            model_date(starting_index=10),
            model_date(raw_text="private invented date"),
            model_date(raw_text=""),
        )
        for entry in bad_entries:
            with self.subTest(entry=entry), self.assertRaises(ValueError) as caught:
                self.extract([entry])
            self.assertNotIn("private invented date", str(caught.exception))

    def test_invalid_response_shapes_and_limits_are_rejected(self):
        for entries in (None, {}, "1950", [None], [model_date()] * 101):
            with self.subTest(entries=type(entries)), self.assertRaises(ValueError):
                self.extract(entries)

    def test_overlapping_source_spans_are_rejected(self):
        with self.assertRaises(ValueError):
            self.extract([model_date(), model_date()])

    def test_valid_entries_keep_source_order_and_partial_precision(self):
        entries = [model_date(raw_text="1952", starting_index=9, start_date={"year": 1952}), model_date()]
        self.assertEqual(self.extract(entries, "1950 and 1952"), (Date(year=1950), Date(year=1952)))

    def test_month_name_ranges_keep_their_complete_source_span(self):
        text = "1 January 1950 to 31 December 1951"
        entry = model_date(raw_text=text, start_date={"day": 1, "month": 1, "year": 1950},
                           end_date={"day": 31, "month": 12, "year": 1951})
        dates = self.extract([entry], text)
        self.assertEqual(dates[0].to_string(), "01/01/1950 - 31/12/1951")
        with self.assertRaises(ValueError):
            self.extract([model_date(raw_text="x" * 101)], "x" * 101)

    def test_empty_text_does_not_request_a_model(self):
        client = Mock()
        for text in ("", " \n\t"):
            self.assertEqual(QwenDateDetector(client).extract_dates(text), ())
        client.query.assert_not_called()

    def test_schema_uses_effective_array_and_numeric_constraints(self):
        array = get_result_schema("1950")["properties"]["dates"]
        self.assertEqual(array["maxItems"], 100)
        index = array["items"]["properties"]["starting_index"]
        self.assertEqual((index["minimum"], index["maximum"]), (0, 3))

    def test_regex_does_not_extract_years_inside_longer_numbers(self):
        detector = RegexDateDetector()
        for text in ("Box 20010002", "Reference 12008.00360", "01950", "19500"):
            with self.subTest(text=text):
                self.assertEqual(detector.extract_dates(text), ())
        self.assertEqual(detector.extract_dates("Records (1950), filed 1952."), (Date(year=1950), Date(year=1952)))


class DateFailureTest(unittest.TestCase):
    def failed_result(self, *, full_image=False):
        fixture = test_pipeline.PipelineTest()
        fixture.setUp()
        detector = test_pipeline.FakeDetector("qwen", [] if full_image else [fixture.region])
        transcriber = test_pipeline.ScriptedTranscriber(["record"], full="record")
        pipeline = fixture.pipeline(detector, transcriber)
        pipeline.date_detector = Mock()
        pipeline.date_detector.name = "qwen"
        pipeline.date_detector.extract_dates.side_effect = RuntimeError("synthetic date failure")
        return fixture.process(pipeline)

    def test_date_failure_preserves_rows_but_marks_partial_and_requires_review(self):
        for full_image in (False, True):
            with self.subTest(full_image=full_image):
                result = self.failed_result(full_image=full_image)
                self.assertEqual(result.status, "partial")
                self.assertEqual(len(result.records), 1)
                self.assertTrue(result.records[0].transcription.date_extraction_failed)
                self.assertIn("Date extraction failed", catalogue_records([result])[0]["Review Flags"])
                self.assertTrue(result.to_dict()["boxes"][0]["date_extraction_failed"])

    def test_date_failure_survives_checkpoints_and_old_records_remain_readable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            box = make_box(crop_path=root / "Box Images/box.png")
            transcription = replace(make_transcription(box), date_extraction_failed=True)
            original = make_result(BoxResult(box, transcription), status="partial")
            saved = serialize_result(original, root)
            self.assertEqual(deserialize_result(saved, root), original)
            self.assertTrue(reading(asdict(transcription)).date_extraction_failed)

            # Older checkpoints predate the flag and must continue to load.
            legacy = copy.deepcopy(saved)
            del legacy["records"][0]["transcription"]["date_extraction_failed"]
            self.assertFalse(deserialize_result(legacy, root).records[0].transcription.date_extraction_failed)

    def test_date_failure_flag_is_identical_in_csv_and_excel(self):
        import csv

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self.failed_result()
            export_catalogue([result], root / "catalogue.csv")
            export_catalogue([result], root / "catalogue.xlsx")
            with (root / "catalogue.csv").open(encoding="utf-8-sig", newline="") as source:
                csv_row = next(csv.DictReader(source))
            rows, _ = read_excel(root / "catalogue.xlsx")
            excel_row = dict(zip(rows[0], rows[1], strict=True))
            self.assertEqual(csv_row["Review Flags"], excel_row["Review Flags"])
            self.assertIn("Date extraction failed", csv_row["Review Flags"])

    def test_verification_date_failure_keeps_original_reading_and_evidence(self):
        box = make_box()
        first = make_transcription(box, handwritten=("Original 1950",), dates=(Date(year=1950),))
        second = replace(first, handwritten_lines=("Changed 1951",))
        transcriber = Mock()
        transcriber.verify.return_value = [second]
        detector = Mock()
        detector.extract_dates.side_effect = RuntimeError("synthetic date failure")
        result = verify_image(make_result(BoxResult(box, first)), transcriber, on_stage=lambda _: None,
                              date_detector=detector)
        self.assertEqual(result.status, "partial")
        self.assertEqual(result.records[0].transcription.handwritten_lines, first.handwritten_lines)
        self.assertEqual(result.records[0].transcription.dates, first.dates)
        self.assertEqual(result.records[0].verification["second"][0]["handwritten_lines"], second.handwritten_lines)

    def test_corroborated_handwriting_correction_carries_its_date_outcome(self):
        box = make_box()
        first = replace(make_transcription(box, handwritten=("Records 1950",)), date_extraction_failed=True)
        second = replace(first, handwritten_lines=("Records 1951",), dates=(Date(year=1951),), date_extraction_failed=False)
        chosen, _, unresolved = reconcile(first, second, second)
        self.assertFalse(unresolved)
        self.assertFalse(chosen.date_extraction_failed)
        self.assertEqual(chosen.dates, second.dates)
