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
from uma_bct.util.date_util import Date, DateRange
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

    def test_model_components_and_precision_must_have_source_evidence(self):
        cases = (
            ("1950", {"year": 2049}, {}),
            ("May 1950", {"day": 31, "month": 12, "year": 1950}, {}),
            ("May 1950", {"day": 1, "month": 5, "year": 1950}, {}),
            ("5/1950", {"day": 5, "month": 5, "year": 1950}, {}),
            ("no date", {"year": 2049}, {}),
            ("Series1950", {"year": 1950}, {}),
            ("1950", {"year": 1950}, {"year": 1951}),
            ("2021-23", {"year": 2021}, {"year": 2024}),
            ("12 May 1950", {"month": 5, "year": 2012}, {}),
            ("12 May 1950 - 15 June 1951", {"month": 5, "year": 2012}, {"month": 6, "year": 2015}),
        )
        for text, start, end in cases:
            with self.subTest(text=text, start=start, end=end):
                with self.assertRaises(ValueError) as caught:
                    self.extract([model_date(raw_text=text, start_date=start, end_date=end)], text)
                self.assertNotIn(text, str(caught.exception))

    def test_source_evidence_preserves_shared_components_and_ai_range_syntax(self):
        def components(value):
            return {key: part for key, part in asdict(value).items() if part is not None}

        cases = (
            ("12/05/1990", Date(day=12, month=5, year=1990), None),
            ("12May1950", Date(day=12, month=5, year=1950), None),
            ("May1950", Date(month=5, year=1950), None),
            ("12thMay1950", Date(day=12, month=5, year=1950), None),
            ("12Sept1950", Date(day=12, month=9, year=1950), None),
            ("Sept1950", Date(month=9, year=1950), None),
            ("12-15/May", Date(day=12, month=5), Date(day=15, month=5)),
            ("12-12/May/1950", Date(day=12, month=5, year=1950), Date(day=12, month=5, year=1950)),
            ("05-06/1990", Date(month=5, year=1990), Date(month=6, year=1990)),
            ("May-Jun/1950", Date(month=5, year=1950), Date(month=6, year=1950)),
            ("from May until July in 1950", Date(month=5, year=1950), Date(month=7, year=1950)),
            ("2021-23", Date(year=2021), Date(year=2023)),
            ("1942-9", Date(year=1942), Date(year=1949)),
            ("21st September 1950", Date(day=21, month=9, year=1950), None),
        )
        for text, start, end in cases:
            entry = model_date(raw_text=text, start_date=components(start), end_date=components(end) if end else {})
            with self.subTest(text=text):
                self.assertEqual(self.extract([entry], text), (DateRange(start, end) if end else start,))

    def test_source_guard_omits_unanchored_short_years_for_review(self):
        for year in (50, 1950, 2050):
            with self.subTest(year=year):
                text = "May 50"
                entry = model_date(raw_text=text, start_date={"month": 5, "year": year})
                client = Mock()
                client.query.return_value = ModelResponse(value={"dates": [entry]})
                outcome = QwenDateDetector(client).extract_dates_with_review(text)
                self.assertEqual(outcome.dates, ())
                self.assertTrue(outcome.needs_review)

    def test_regex_century_ambiguity_keeps_other_dates_and_full_year_ranges(self):
        detector = RegexDateDetector()
        for text in ("12 May 50", "12 May 25", "May 50", "05/90", "02/03/04", "05-06/90", "12May50",
                     "12-15/May/50", "12-15/05/50"):
            with self.subTest(text=text):
                outcome = detector.extract_dates_with_review(text + "; filed 1952")
                self.assertEqual(outcome.dates, (Date(year=1952),))
                self.assertTrue(outcome.needs_review)
                self.assertEqual(detector.extract_dates(text), ())
        for text, expected in (
            ("2021-23", DateRange(Date(year=2021), Date(year=2023))),
            ("1942-9", DateRange(Date(year=1942), Date(year=1949))),
            ("12-15/May/1950", DateRange(Date(day=12, month=5, year=1950), Date(day=15, month=5, year=1950))),
        ):
            with self.subTest(text=text):
                outcome = detector.extract_dates_with_review(text)
                self.assertEqual(outcome.dates, (expected,))
                self.assertFalse(outcome.needs_review)

    def test_ai_omits_ambiguous_entries_and_flags_even_an_empty_reply(self):
        text = "12 May 50; filed 1952"
        entries = [model_date(raw_text="12 May 50", start_date={"day": 12, "month": 5, "year": 1950}),
                   model_date(raw_text="1952", starting_index=text.index("1952"), start_date={"year": 1952})]
        for values, expected in ((entries, (Date(year=1952),)), ([], ())):
            with self.subTest(values=values):
                client = Mock()
                client.query.return_value = ModelResponse(value={"dates": values})
                outcome = QwenDateDetector(client).extract_dates_with_review(text)
                self.assertEqual(outcome.dates, expected)
                self.assertTrue(outcome.needs_review)
                client.query.assert_called_once()

    def test_ai_only_short_year_syntax_is_reviewed_but_four_digit_years_are_kept(self):
        cases = (
            ("from May until July in 50", {"month": 5, "year": 1950}, {"month": 7, "year": 1950}, (), True),
            ("0050", {"year": 50}, {}, (Date(year=50),), False),
            ("2021-23", {"year": 2021}, {"year": 2023}, (DateRange(Date(year=2021), Date(year=2023)),), False),
        )
        for text, start, end, expected, review in cases:
            with self.subTest(text=text):
                client = Mock()
                client.query.return_value = ModelResponse(value={"dates": [
                    model_date(raw_text=text, start_date=start, end_date=end),
                ]})
                outcome = QwenDateDetector(client).extract_dates_with_review(text)
                self.assertEqual((outcome.dates, outcome.needs_review), (expected, review))

    def test_ambiguous_years_do_not_hide_malformed_model_components(self):
        text = "May 50"
        with self.assertRaises(ValueError):
            self.extract([model_date(raw_text=text, start_date={"month": 13, "year": 1950})], text)

    def test_early_year_ranges_keep_their_explicit_four_digit_anchor(self):
        cases = (
            ("0050-51", {"year": 50}, {"year": 51}),
            ("0001-05", {"year": 1}, {"year": 5}),
            ("May0050-Jun51", {"month": 5, "year": 50}, {"month": 6, "year": 51}),
            ("May0001-Jun05", {"month": 5, "year": 1}, {"month": 6, "year": 5}),
        )
        for text, start, end in cases:
            with self.subTest(text=text):
                client = Mock()
                client.query.return_value = ModelResponse(value={"dates": [
                    model_date(raw_text=text, start_date=start, end_date=end),
                ]})
                outcome = QwenDateDetector(client).extract_dates_with_review(text)
                self.assertEqual(outcome.dates, (DateRange(Date(**start), Date(**end)),))
                self.assertFalse(outcome.needs_review)
                client.query.assert_called_once()

        # A printed full-year anchor does not authorize a different suffix.
        with self.assertRaises(ValueError):
            self.extract([model_date(raw_text="0050-52", start_date={"year": 50}, end_date={"year": 51})], "0050-52")

    def test_ai_anchors_only_the_short_range_end_and_keeps_other_ambiguity(self):
        for text, start, end in (
            ("May 1950-Jun 51", {"month": 5, "year": 1950}, {"month": 6, "year": 1951}),
            ("12May1950-15June51", {"day": 12, "month": 5, "year": 1950},
             {"day": 15, "month": 6, "year": 1951}),
        ):
            for extra in ("", "; May 50"):
                with self.subTest(text=text, extra=extra):
                    client = Mock()
                    client.query.return_value = ModelResponse(value={"dates": [
                        model_date(raw_text=text, start_date=start, end_date=end),
                    ]})
                    outcome = QwenDateDetector(client).extract_dates_with_review(text + extra)
                    self.assertEqual(outcome.dates, (DateRange(Date(**start), Date(**end)),))
                    self.assertEqual(outcome.needs_review, bool(extra))

        text = "May 50; filed 1950"
        client = Mock()
        client.query.return_value = ModelResponse(value={"dates": [
            model_date(raw_text="May 50", start_date={"month": 5, "year": 1950}),
            model_date(raw_text="1950", starting_index=text.index("1950"), start_date={"year": 1950}),
        ]})
        outcome = QwenDateDetector(client).extract_dates_with_review(text)
        self.assertEqual(outcome.dates, (Date(year=1950),))
        self.assertTrue(outcome.needs_review)

        text = "Filed 1950; May 50 to June 51"
        client.query.return_value = ModelResponse(value={"dates": [
            model_date(raw_text=text, start_date={"month": 5, "year": 1950},
                       end_date={"month": 6, "year": 1951}),
        ]})
        outcome = QwenDateDetector(client).extract_dates_with_review(text)
        self.assertEqual(outcome.dates, ())
        self.assertTrue(outcome.needs_review)

    def test_verification_dates_share_the_filtered_input_and_preserve_raw_readings(self):
        box = make_box()
        first = make_transcription(
            box, identifiers=("1950.1960",), unit="1950", typed=("1950.1960", "Unit 1950"),
            handwritten=("Unit 1950", "1950.1960", "1951", "Minutes 1952"),
            dates=(Date(year=1951), Date(year=1952)),
        )
        transcriber = Mock()
        transcriber.verify.return_value = [first]
        detector = Mock(spec=RegexDateDetector, wraps=RegexDateDetector())
        result = verify_image(make_result(BoxResult(box, first)), transcriber, on_stage=lambda _: None,
                              date_detector=detector)
        detector.extract_dates_with_review.assert_called_once_with("1951\nMinutes 1952")
        self.assertEqual(result.records[0].transcription.handwritten_lines, first.handwritten_lines)
        self.assertEqual(result.records[0].transcription.dates, first.dates)
        second = result.records[0].verification["second"][0]
        self.assertEqual(second["handwritten_lines"], first.handwritten_lines)
        self.assertEqual(reading(second).dates, first.dates)

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

    def test_regex_does_not_extract_a_prefix_or_suffix_of_a_dotted_identifier(self):
        detector = RegexDateDetector()
        for text in ("1980.0950", "12008.1990", "Reference 1980.0950"):
            with self.subTest(text=text):
                self.assertEqual(detector.extract_dates(text), ())

        # Full calendar dates and configured dotted ranges still consume the
        # whole token; only partial matches inside dotted numbers are blocked.
        for text, expected in (
            ("12.05.1990", Date(day=12, month=5, year=1990)),
            ("12/5", Date(day=12, month=5)),
            ("1950.1960", DateRange(Date(year=1950), Date(year=1960))),
            ("1.May 1950", Date(month=5, year=1950)),
            ("1.January 1950 - December 1950",
             DateRange(Date(month=1, year=1950), Date(month=12, year=1950))),
            ("12-15/May", DateRange(Date(day=12, month=5), Date(day=15, month=5))),
        ):
            with self.subTest(text=text):
                self.assertEqual(detector.extract_dates(text), (expected,))

        box = make_box()
        transcription = make_transcription(box, identifiers=("1980.0950",), unit="7", handwritten=("1980.0950",))
        transcription = replace(transcription, dates=detector.extract_dates(transcription.get_handwritten_text()))
        row = catalogue_records([make_result(BoxResult(box, transcription))])[0]
        self.assertEqual((row["Description"], row["Dates"], row["Earliest Date"], row["Latest Date"]),
                         ("", "No Dates", "None", "None"))


class DateFailureTest(unittest.TestCase):
    def failed_result(self, *, full_image=False):
        fixture = test_pipeline.PipelineTest()
        fixture.setUp()
        detector = test_pipeline.FakeDetector("qwen", [] if full_image else [fixture.region])
        transcriber = test_pipeline.ScriptedTranscriber(["record"], full="record")
        pipeline = fixture.pipeline(detector, transcriber)
        pipeline.date_detector = Mock(spec=["name", "extract_dates"])
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

    def test_date_review_flag_roundtrips_and_is_optional_in_legacy_readings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            box = make_box(crop_path=root / "Box Images/box.png")
            transcription = replace(make_transcription(box), date_needs_review=True, dates=(Date(year=1952),))
            original = make_result(BoxResult(box, transcription))
            saved = serialize_result(original, root)
            self.assertEqual(deserialize_result(saved, root), original)
            self.assertTrue(reading(asdict(transcription)).date_needs_review)
            legacy = copy.deepcopy(saved)
            del legacy["records"][0]["transcription"]["date_needs_review"]
            self.assertFalse(deserialize_result(legacy, root).records[0].transcription.date_needs_review)

    def test_date_failure_flag_is_identical_in_csv_and_excel(self):
        import csv

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Windows CI keeps the checkout and temp directory on different
            # drives. A portable catalogue and its crop must share one root.
            box = make_box(crop_path=root / "Box Images/box.png")
            transcription = replace(make_transcription(box), date_extraction_failed=True)
            result = make_result(BoxResult(box, transcription), status="partial")
            export_catalogue([result], root / "catalogue.csv")
            export_catalogue([result], root / "catalogue.xlsx")
            with (root / "catalogue.csv").open(encoding="utf-8-sig", newline="") as source:
                csv_row = next(csv.DictReader(source))
            rows, _ = read_excel(root / "catalogue.xlsx")
            excel_row = dict(zip(rows[0], rows[1], strict=True))
            self.assertEqual(csv_row["Review Flags"], excel_row["Review Flags"])
            self.assertIn("Date extraction failed", csv_row["Review Flags"])
            self.assertEqual(csv_row["Cropped Box Image Link"], "Box Images/box.png")

    def test_verification_date_failure_keeps_original_reading_and_evidence(self):
        box = make_box()
        first = make_transcription(box, handwritten=("Original 1950",), dates=(Date(year=1950),))
        second = replace(first, handwritten_lines=("Changed 1951",))
        transcriber = Mock()
        transcriber.verify.return_value = [second]
        detector = Mock(spec=["extract_dates"])
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
