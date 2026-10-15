import contextlib
import io
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from uma_bct.terminal import TerminalProgress, human_summary, print_run_summary, print_selected_model


class TerminalProgressTest(unittest.TestCase):
    def test_selected_model_uses_stderr_and_sanitizes_control_characters(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            print_selected_model(
                {"model": "example\nmodel", "backend": "hosted\x1b", "api_key": "secret"},
                fallback_model="fallback",
            )
            print_selected_model({}, fallback_model="local-model")
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(
            stderr.getvalue(), "Selected model: example?model (hosted?)\nSelected model: local-model (ollama)\n"
        )
        self.assertNotIn("secret", stderr.getvalue())

    def setUp(self):
        terminal = patch.dict(os.environ, {"TERM": "xterm-256color"})
        terminal.start()
        self.addCleanup(terminal.stop)

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

    def test_reused_history_report_lists_all_results_and_retains_failures(self):
        summary = {
            "already_complete": True, "status": "partial", "images_total": 2,
            "images_complete": 1, "images_partial": 0, "images_failed": 1,
            "catalogue_records": 2, "current_input_records": 1, "catalogue_output": None,
            "catalogue_outputs": ["/results/second/Catalogue.csv"],
            "batch_directories": ["/results/second", "/results/first"],
            "run_manifests": ["/results/second/Run Details.json", "/results/first/Run Details.json"],
        }
        report = human_summary(summary, elapsed_seconds=1)
        self.assertIn("Already complete with issues", report)
        self.assertIn("1 complete | 0 partial | 1 failed", report)
        self.assertIn("Catalogue: 2 recorded rows", report)
        self.assertIn("Transcribed boxes in current photos: 1", report)
        self.assertNotIn("Catalogue: not created", report)
        for path in (*summary["catalogue_outputs"], *summary["run_manifests"]):
            self.assertIn(path, report)

    def test_recovery_report_lists_only_retained_recovery_files(self):
        summary = {"status": "partial", "catalogue_output": None, "catalogue_records": 0,
                   "recovery_path": "/results/old-catalogue.csv",
                   "recovery_paths": ["/results/old-catalogue.csv"]}
        report = human_summary(summary, elapsed_seconds=1)
        self.assertIn("Recovery file: /results/old-catalogue.csv", report)
        self.assertEqual(report.count("Recovery file:"), 1)
        self.assertIn("Catalogue: not created", report)

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

    def test_missing_rich_uses_milestones(self):
        output = io.StringIO()
        with patch("uma_bct.terminal.Console", None):
            with TerminalProgress(1, stream=output, force_terminal=True) as reporter:
                reporter.photo_started(1, Path("shelf.jpg"))
                reporter.photo_finished(1, "complete", 1)
        self.assertIn("Processing Photo 1 of 1: shelf.jpg", output.getvalue())
        self.assertNotIn("\x1b[", output.getvalue())

    def test_dumb_terminal_uses_milestones_instead_of_silent_live_updates(self):
        for total in (None, 1):
            with self.subTest(total=total), patch.dict(os.environ, {"TERM": "dumb"}):
                output = io.StringIO()
                with patch("uma_bct.terminal.Progress") as make_progress:
                    with TerminalProgress(total, stream=output, force_terminal=True) as reporter:
                        reporter.phase("Verifying crop 1 of 1")
                        if total is not None:
                            reporter.photo_started(1, Path("shelf.jpg"))
                            reporter.photo_finished(1, "complete", 1)
                    make_progress.assert_not_called()
                self.assertIn("Verifying crop 1 of 1", output.getvalue())
                self.assertNotIn("\x1b[", output.getvalue())
                if total is None:
                    self.assertIn("Preparing UMA BCT", output.getvalue())
                else:
                    self.assertIn("Photo 1 finished: complete | 1 box found", output.getvalue())

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
