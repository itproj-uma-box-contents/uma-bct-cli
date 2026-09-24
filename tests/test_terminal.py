import contextlib
import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

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
