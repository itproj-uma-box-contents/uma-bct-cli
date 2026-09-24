import contextlib
import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from uma_bct.terminal import TerminalProgress, human_summary, print_run_summary


class TerminalProgressTest(unittest.TestCase):
    # Check that captured output stays short and the final numbers remain honest.
    def test_captured_progress_and_human_report_use_plain_committed_counts(self):
        moment = [100.0]
        output = io.StringIO()
        with TerminalProgress(2, stream=output, clock=lambda: moment[0]) as reporter:
            reporter.photo_started(1, Path("shelf\nphoto.jpg"))
            reporter.phase("Transcribing box 1 of 2")
            moment[0] = 131.0
            reporter.photo_finished(1, "complete", 2)
            reporter.phase("Saving the catalogue")

        progress = output.getvalue()
        self.assertIn("Processing Photo 1 of 2: shelf?photo.jpg", progress)
        self.assertIn("Photo 1 of 2: shelf?photo.jpg | Transcribing box 1 of 2", progress)
        self.assertIn("Photo 1 finished: complete | 2 boxes found", progress)
        self.assertNotIn("Still working:", progress)

        summary = {
            "status": "partial", "images_total": 2, "images_complete": 1,
            "images_partial": 0, "images_failed": 1, "catalogue_records": 2,
            "catalogue_output": "/private/results/Catalogue.csv",
            "run_manifest": "/private/results/Run Details.json",
        }
        report = human_summary(summary, elapsed_seconds=31)
        self.assertIn("Finished with issues", report)
        self.assertIn("Photos: 1 complete | 0 partial | 1 failed", report)
        self.assertIn("Catalogue: 2 rows", report)
        self.assertIn("Time this launch: 0m 31s", report)

        # A preflight failure must not describe untouched photos as failures.
        summary.update(status="failed", images_complete=0, images_failed=2, catalogue_output=None)
        early_report = human_summary(summary, elapsed_seconds=1, attempted=0)
        self.assertIn("Photos finished: 0 of 2", early_report)
        self.assertNotIn("2 failed", early_report)

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            print_run_summary(summary, human=False, elapsed_seconds=1)
        self.assertEqual(json.loads(stdout.getvalue()), summary)

    # Check that a real terminal redraws one live status instead of appending heartbeats.
    def test_live_progress_redraws_in_place(self):
        output = io.StringIO()
        with TerminalProgress(1, stream=output, force_terminal=True) as reporter:
            reporter.photo_started(1, Path("shelf.jpg"))
            reporter.phase("Transcribing box 1 of 6")
            print("Warning: check this photo", file=sys.stderr)
            reporter.photo_finished(1, "complete", 6)
        self.assertIn("Transcribing box 1 of 6", output.getvalue())
        self.assertIn("Warning: check this photo", output.getvalue())
        self.assertIn("\x1b[", output.getvalue())
        self.assertNotIn("Still working:", output.getvalue())

    # Older offline bundles keep working if their runtime lacks Rich.
    def test_missing_rich_uses_milestones(self):
        output = io.StringIO()
        with patch("uma_bct.terminal.Console", None):
            with TerminalProgress(1, stream=output, force_terminal=True) as reporter:
                reporter.photo_started(1, Path("shelf.jpg"))
                reporter.photo_finished(1, "complete", 1)
        self.assertIn("Processing Photo 1 of 1: shelf.jpg", output.getvalue())
        self.assertNotIn("\x1b[", output.getvalue())

    # A display error must not turn a successful photo into a failed one.
    def test_live_display_error_falls_back_to_milestones(self):
        output = io.StringIO()
        live = Mock()
        live.update.side_effect = OSError("terminal unavailable")
        with TerminalProgress(1, stream=output) as reporter:
            reporter._progress = live
            reporter._task_id = 1
            reporter.phase("Transcribing box 1 of 2")
            reporter.photo_finished(1, "complete", 2)
        live.stop.assert_called_once_with()
        self.assertIn("Live display unavailable", output.getvalue())
        self.assertIn("Transcribing box 1 of 2", output.getvalue())
        self.assertIn("Photo 1 finished: complete", output.getvalue())

    # Startup and cleanup errors in Rich must leave the transcription usable.
    def test_live_start_and_stop_errors_do_not_abort(self):
        output = io.StringIO()
        with patch("uma_bct.terminal.Progress") as make_progress:
            live = make_progress.return_value
            live.add_task.return_value = 1
            live.start.side_effect = OSError("terminal unavailable")
            with TerminalProgress(1, stream=output, force_terminal=True) as reporter:
                reporter.photo_started(1, Path("shelf.jpg"))
            live.stop.assert_called_once_with()
        self.assertIn("Live display unavailable", output.getvalue())
        self.assertIn("Processing Photo 1 of 1", output.getvalue())

        output = io.StringIO()
        with patch("uma_bct.terminal.Progress") as make_progress:
            live = make_progress.return_value
            live.add_task.return_value = 1
            live.stop.side_effect = OSError("cannot clear")
            with TerminalProgress(1, stream=output, force_terminal=True):
                pass
        self.assertIn("could not cleanly stop", output.getvalue())
