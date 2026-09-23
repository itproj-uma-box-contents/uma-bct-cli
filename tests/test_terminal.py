import contextlib
import io
import json
import unittest
from pathlib import Path

from uma_bct.terminal import TerminalProgress, human_summary, print_run_summary


class TerminalProgressTest(unittest.TestCase):
    # Check that a slow photo stays visible and the final numbers remain honest.
    def test_heartbeat_and_human_report_use_plain_committed_counts(self):
        moment = [100.0]
        output = io.StringIO()
        with TerminalProgress(2, stream=output, clock=lambda: moment[0]) as reporter:
            self.assertEqual(reporter.heartbeat_seconds, 30)
            reporter.photo_started(1, Path("shelf\nphoto.jpg"))
            moment[0] = 131.0
            reporter.heartbeat()
            reporter.photo_finished(1, "complete", 2)
            reporter.phase("Saving the catalogue...")

        progress = output.getvalue()
        self.assertIn("Processing photo 1 of 2: shelf?photo.jpg", progress)
        self.assertIn("Still working: photo 1 of 2", progress)
        self.assertIn("0 of 2 finished | 0m 31s elapsed", progress)
        self.assertIn("1 of 2 finished", progress)

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
