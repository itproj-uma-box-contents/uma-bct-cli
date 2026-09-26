import argparse
import io
import os
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import Mock, patch

from helpers import make_box, make_result, make_transcription

from uma_bct.adapters.qwen_transcriber import QwenTextTranscriber
from uma_bct.domain import BoxResult
from uma_bct.export import catalogue_records
from uma_bct.ports import ModelResponse
from uma_bct.verification import VerificationSetupError, choose_verification, configured_mode, reconcile, verify_image


class VerificationTest(unittest.TestCase):
    def setUp(self):
        self.box = make_box()
        self.first = make_transcription(self.box)
        self.result = make_result(BoxResult(self.box, self.first))

    def run_verification(self, second, third=None):
        transcriber = Mock()
        transcriber.verify.return_value = second
        transcriber.resolve_discrepancy.return_value = third or []
        result = verify_image(self.result, transcriber, on_stage=lambda _stage: None)
        return result, transcriber

    def test_equivalent_whitespace_and_unit_padding_do_not_need_a_third_reading(self):
        second = replace(self.first, unit_number="0004", typed_lines=("1960.0009   Unit 4",), confidence=89)
        result, transcriber = self.run_verification([second])
        transcriber.resolve_discrepancy.assert_not_called()
        self.assertEqual(result.records[0].transcription.typed_lines, self.first.typed_lines)
        self.assertEqual(result.records[0].transcription.confidence, 89)

    def test_corroborated_identity_changes_together_and_remains_flagged(self):
        second = replace(self.first, unit_number="5", typed_lines=("1960.0009", "Unit 5"))
        result, transcriber = self.run_verification([second], [second])
        self.assertEqual(result.records[0].transcription.unit_number, "5")
        self.assertEqual(result.records[0].transcription.typed_lines, second.typed_lines)
        self.assertTrue(result.records[0].transcription.needs_review)
        self.assertEqual(transcriber.resolve_discrepancy.call_count, 1)
        self.assertIn("Verification discrepancy", catalogue_records([result])[0]["Review Flags"])

    def test_three_different_readings_keep_original(self):
        second = replace(self.first, handwritten_lines=("Ledger",))
        third = replace(self.first, handwritten_lines=("Letter",))
        result, _ = self.run_verification([second], [third])
        self.assertEqual(result.records[0].transcription.handwritten_lines, ())
        self.assertIn("Verification unresolved", catalogue_records([result])[0]["Review Flags"])

    def test_two_blank_readings_do_not_erase_original_text(self):
        first = replace(self.first, handwritten_lines=("Visible note",))
        selected, _, unresolved = reconcile(first, self.first, self.first)
        self.assertEqual(selected.handwritten_lines, ("Visible note",))
        self.assertEqual(unresolved, ["handwritten_lines"])

    def test_inconsistent_structured_unit_is_not_promoted(self):
        second = replace(self.first, unit_number="5")
        selected, _, unresolved = reconcile(self.first, second, second)
        self.assertEqual(selected.unit_number, "4")
        self.assertIn("unit_number", unresolved)

    def test_partial_omissions_keep_original_fields(self):
        cases = (
            ("typed_lines", ("1960.0009", "Unit 4", "Department of Chemistry"), ("1960.0009", "Unit 4")),
            ("handwritten_lines", ("Minutes", "Restricted until 2055"), ("Minutes",)),
            ("handwritten_lines", ("Minutes not for release",), ("Minutes for release",)),
            ("handwritten_lines", ("Minutes", "Minutes"), ("Minutes",)),
            ("handwritten_lines", ("Minutes restricted until 2055",), ("Letters until 2055",)),
            ("auxiliary_annotations", ("Water damage", "Mould"), ("Water damage",)),
            ("auxiliary_annotations", ("Severe water damage",), ("Water damage",)),
        )
        for field, original, candidate in cases:
            with self.subTest(field=field, original=original):
                first = replace(self.first, **{field: original})
                second = replace(first, **{field: candidate})
                selected, _, unresolved = reconcile(first, second, second)
                self.assertEqual(getattr(selected, field), original)
                self.assertIn(field, unresolved)
                self.assertTrue(selected.needs_review)

    def test_omitted_identifier_keeps_all_printed_identity(self):
        first = replace(self.first, record_identifiers=("1960.0009", "1970.0001"),
                        typed_lines=("1960.0009", "1970.0001", "Unit 4"))
        selected, _, unresolved = reconcile(first, self.first, self.first)
        self.assertEqual(selected.record_identifiers, first.record_identifiers)
        self.assertEqual(selected.typed_lines, first.typed_lines)
        self.assertIn("record_identifiers", unresolved)

    def test_adapter_handwritten_units_are_reconciled_together(self):
        def response(unit):
            return ModelResponse({"boxes": [{
                "position": "middle-left", "record_identifiers": ["1960.0009"], "unit": unit,
                "typed": ["1960.0009"], "handwritten": [f"Unit {unit}", "Minutes"],
                "auxiliary_annotations": [],
            }]})

        client = Mock()
        client.query.side_effect = [response("4"), response("5"), response("5")]
        transcriber = QwenTextTranscriber(client)
        first = transcriber.transcribe(self.box)[0]
        result = verify_image(make_result(BoxResult(self.box, first)), transcriber, on_stage=lambda _: None)
        chosen = result.records[0].transcription
        self.assertEqual(chosen.unit_number, "5")
        self.assertEqual(chosen.handwritten_lines, ("Unit 5", "Minutes"))
        self.assertEqual(catalogue_records([result])[0]["Unit Number"], "0005")
        self.assertTrue(chosen.needs_review)

    def test_conflicting_handwritten_unit_cannot_blank_valid_original(self):
        second = replace(self.first, handwritten_lines=("Unit 5",))
        result, _ = self.run_verification([second], [second])
        self.assertEqual(result.records[0].transcription.handwritten_lines, ())
        self.assertEqual(catalogue_records([result])[0]["Unit Number"], "0004")
        self.assertIn("handwritten_lines", result.records[0].verification["unresolved"])

    def test_handwritten_identifiers_follow_structured_identity(self):
        first = replace(self.first, typed_lines=("Unit 4",), handwritten_lines=("1960.0009", "Minutes"))
        second = replace(first, record_identifiers=("1970.0001",), handwritten_lines=("1970.0001", "Minutes"))
        selected, _, unresolved = reconcile(first, second, second)
        self.assertEqual(selected.record_identifiers, second.record_identifiers)
        self.assertEqual(selected.handwritten_lines, second.handwritten_lines)
        self.assertEqual(unresolved, [])

    def test_identity_correction_with_omission_preserves_entire_original_group(self):
        first = replace(self.first, typed_lines=("1960.0009",),
                        handwritten_lines=("Unit 4", "Minutes restricted until 2055"))
        second = replace(first, unit_number="5", handwritten_lines=("Unit 5", "Minutes"))
        chosen, _, unresolved = reconcile(first, second, second)
        result = make_result(BoxResult(self.box, chosen))
        self.assertEqual(chosen.unit_number, "4")
        self.assertEqual(chosen.handwritten_lines, first.handwritten_lines)
        self.assertIn("handwritten_lines", unresolved)
        self.assertEqual(catalogue_records([result])[0]["Unit Number"], "0004")

    def test_corroborated_token_correction_without_omission_is_allowed(self):
        first = replace(self.first, handwritten_lines=("Meeting minuets",))
        second = replace(first, handwritten_lines=("Meeting minutes",))
        chosen, _, unresolved = reconcile(first, second, second)
        self.assertEqual(chosen.handwritten_lines, second.handwritten_lines)
        self.assertEqual(unresolved, [])
        self.assertTrue(chosen.needs_review)

    def cached_result(self, second, third=None):
        audit = {"version": 1, "status": "pending", "original": [asdict(self.first)],
                 "second": [asdict(second)]}
        if third is not None:
            audit["third"] = [asdict(third)]
        return make_result(BoxResult(self.box, self.first, audit))

    def test_cached_readings_do_not_load_transcriber(self):
        changed = replace(self.first, handwritten_lines=("Minutes",))
        for result, expected in ((self.cached_result(self.first), ()),
                                 (self.cached_result(changed, changed), ("Minutes",))):
            with self.subTest(expected=expected):
                factory = Mock(side_effect=AssertionError("must not load"))
                verified = verify_image(result, None, on_stage=lambda _: None, get_transcriber=factory)
                factory.assert_not_called()
                self.assertEqual(verified.records[0].transcription.handwritten_lines, expected)
                self.assertEqual(verified.records[0].verification["status"], "complete")

    def test_missing_requests_load_transcriber_once(self):
        changed = replace(self.first, handwritten_lines=("Minutes",))
        for cached_second in (False, True):
            with self.subTest(cached_second=cached_second):
                result = self.cached_result(changed) if cached_second else self.result
                transcriber = Mock()
                transcriber.verify.return_value = [changed]
                transcriber.resolve_discrepancy.return_value = [changed]
                factory = Mock(return_value=transcriber)
                verified = verify_image(result, None, on_stage=lambda _: None, get_transcriber=factory)
                factory.assert_called_once_with()
                self.assertEqual(transcriber.verify.call_count, 0 if cached_second else 1)
                transcriber.resolve_discrepancy.assert_called_once()
                self.assertEqual(verified.records[0].transcription.handwritten_lines, ("Minutes",))

    def test_factory_failure_propagates_without_marking_reading_failed(self):
        changed = replace(self.first, handwritten_lines=("Minutes",))
        for result in (self.result, self.cached_result(changed)):
            with self.subTest(cached=bool(result.records[0].verification)):
                failure = ValueError("model identity changed")
                factory = Mock(side_effect=failure)
                saved = []
                with self.assertRaises(VerificationSetupError) as caught:
                    verify_image(result, None, on_stage=lambda _: None, checkpoint=saved.append,
                                 get_transcriber=factory)
                self.assertIs(caught.exception.__cause__, failure)
                self.assertEqual(saved[-1].records[0].verification["status"], "pending")
                self.assertNotIn("error", saved[-1].records[0].verification)
                self.assertEqual(saved[-1].records[0].transcription, self.first)

    def test_failure_preserves_first_reading_and_partial_status_in_checkpoint(self):
        transcriber = Mock()
        transcriber.verify.side_effect = RuntimeError("Ollama stopped")
        saved = []
        result = verify_image(self.result, transcriber, on_stage=lambda _: None, checkpoint=saved.append)
        self.assertEqual(result.status, "partial")
        self.assertEqual(saved[-1].status, "partial")
        self.assertEqual(result.records[0].transcription.typed_lines, self.first.typed_lines)
        self.assertIn("Verification failed", catalogue_records([result])[0]["Review Flags"])

    def test_resume_uses_second_reading_checkpoint(self):
        second = replace(self.first, handwritten_lines=("Notes",))
        transcriber = Mock()
        transcriber.verify.return_value = [second]
        transcriber.resolve_discrepancy.side_effect = KeyboardInterrupt
        saved = []
        with self.assertRaises(KeyboardInterrupt):
            verify_image(self.result, transcriber, on_stage=lambda _: None, checkpoint=saved.append)
        transcriber.verify.reset_mock()
        transcriber.resolve_discrepancy.side_effect = None
        transcriber.resolve_discrepancy.return_value = [second]
        resumed = verify_image(saved[-1], transcriber, on_stage=lambda _: None)
        transcriber.verify.assert_not_called()
        self.assertEqual(resumed.records[0].transcription.handwritten_lines, ("Notes",))

    def test_ambiguous_multiple_boxes_preserve_all_rows(self):
        second_box = replace(self.box, box_id="other")
        self.result = make_result(BoxResult(self.box, self.first),
                                  BoxResult(second_box, replace(self.first, box_id="other")))
        result, transcriber = self.run_verification([self.first, self.first])
        self.assertEqual(len(result.records), 2)
        self.assertTrue(all(record.transcription.needs_review for record in result.records))
        transcriber.resolve_discrepancy.assert_not_called()

    def test_checkpoint_failure_stops_before_inference(self):
        from uma_bct.verification import VerificationCheckpointError
        transcriber = Mock()
        with self.assertRaises(VerificationCheckpointError):
            verify_image(self.result, transcriber, on_stage=lambda _: None,
                         checkpoint=Mock(side_effect=OSError("disk full")))
        transcriber.verify.assert_not_called()

    def test_completed_verification_is_not_repeated(self):
        result, transcriber = self.run_verification([self.first])
        transcriber.reset_mock()
        resumed = verify_image(result, transcriber, on_stage=lambda _: None)
        self.assertEqual(resumed, result)
        transcriber.verify.assert_not_called()

    def test_discrepancy_request_failure_does_not_replace_the_first_reading(self):
        second = replace(self.first, handwritten_lines=("New reading",))
        transcriber = Mock()
        transcriber.verify.return_value = [second]
        transcriber.resolve_discrepancy.side_effect = TimeoutError("request timed out")
        result = verify_image(self.result, transcriber, on_stage=lambda _: None)
        self.assertEqual(result.status, "partial")
        self.assertEqual(result.records[0].transcription.handwritten_lines, self.first.handwritten_lines)
        self.assertIn("Verification failed", catalogue_records([result])[0]["Review Flags"])

    def test_unattended_and_json_runs_do_not_prompt(self):
        with patch("sys.stdin", io.StringIO("y\n")):
            self.assertFalse(choose_verification(argparse.Namespace(json=False), "ask", has_records=True))
        with patch("sys.stdin") as stream:
            self.assertFalse(choose_verification(argparse.Namespace(json=True), "ask", has_records=True))
            stream.readline.assert_not_called()
            self.assertTrue(choose_verification(argparse.Namespace(json=True), "always", has_records=True))

    def test_settings_precedence_and_interactive_always(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            args = argparse.Namespace(results_directory=Path(directory), verification=None, json=False)
            self.assertEqual(configured_mode(args), "ask")
            with patch("sys.stdin") as stdin, patch("sys.stderr") as stderr:
                stdin.isatty.return_value = stderr.isatty.return_value = True
                stdin.readline.return_value = "a\n"
                self.assertTrue(choose_verification(args, "ask", has_records=True))
            self.assertEqual(configured_mode(args), "always")
            with patch.dict(os.environ, {"UMA_BCT_VERIFICATION": "never"}):
                self.assertEqual(configured_mode(args), "never")
                args.verification = "ask"
                self.assertEqual(configured_mode(args), "ask")
