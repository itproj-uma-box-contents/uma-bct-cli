import argparse
import io
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

from helpers import make_box, make_result, make_transcription

from uma_bct.domain import BoxResult
from uma_bct.export import catalogue_records
from uma_bct.verification import choose_verification, configured_mode, reconcile, verify_image


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
