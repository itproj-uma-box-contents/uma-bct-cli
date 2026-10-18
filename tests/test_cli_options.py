"""Check public option spellings, validation, and routing without model work."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from uma_bct.cli import main, parse_args
from uma_bct.model_config import HostedModelConfig


class CliOptionsTest(unittest.TestCase):
    def test_value_options_keep_aliases_and_attached_forms(self):
        cases = (
            ("-i", "Photos to read", "path_to_inputs", Path("Photos to read")),
            ("--path-to-inputs", "Photos to read", "path_to_inputs", Path("Photos to read")),
            ("-o", "Review.xlsx", "path_to_outputs", Path("Review.xlsx")),
            ("--path-to-output", "Review.xlsx", "path_to_outputs", Path("Review.xlsx")),
            ("--path-to-outputs", "Review.xlsx", "path_to_outputs", Path("Review.xlsx")),
            ("-b", "Review crops", "path_to_box_images", Path("Review crops")),
            ("--path-to-box-images", "Review crops", "path_to_box_images", Path("Review crops")),
            ("--work-dir", "Review crops", "path_to_box_images", Path("Review crops")),
            ("--results-directory", "Saved batches", "results_directory", Path("Saved batches")),
            ("--format", "Excel", "file_type", "excel"),
            ("--file-type", "CSV", "file_type", "csv"),
            ("--verification", "never", "verification", "never"),
            ("--verification", "always", "verification", "always"),
            ("--verification", "ask", "verification", "ask"),
            ("-d", "opencv", "detector", "opencv"),
            ("--detector", "qwen", "detector", "qwen"),
            ("-t", "qwen", "transcriber", "qwen"),
            ("--transcriber", "qwen", "transcriber", "qwen"),
            ("-m", "custom:8b", "model", "custom:8b"),
            ("--model", "custom:8b", "model", "custom:8b"),
            ("-c", "80", "min_transcription_confidence", 80),
            ("--min-transcription-confidence", "80", "min_transcription_confidence", 80),
            ("--min-confidence", "0.7", "min_detector_confidence", 0.7),
            ("--min-detector-confidence", "0.7", "min_detector_confidence", 0.7),
            ("--request-timeout", "75.5", "request_timeout", 75.5),
        )
        for option, value, attribute, expected in cases:
            # Long options accept --name=value; short options also accept -mvalue.
            variants = ([option, value], [f"{option}={value}"])
            if len(option) == 2:
                variants += ([option + value],)
            for arguments in variants:
                with self.subTest(arguments=arguments):
                    self.assertEqual(getattr(parse_args(arguments), attribute), expected)

    def test_switches_take_no_value_and_keep_existing_aliases(self):
        for option, attribute, expected in (
            ("--ai-dates", "ai_dates", True), ("--hosted", "hosted", True),
            ("--json", "json", True), ("--human", "human", True),
            ("--reprocess", "reprocess", True), ("-f", "force", True), ("--force", "force", True),
            ("--no-missing-box", "detect_missing_boxes", False),
            ("--no-missing-box-detection", "detect_missing_boxes", False),
        ):
            with self.subTest(option=option):
                self.assertEqual(getattr(parse_args([option]), attribute), expected)

    def test_invalid_numeric_options_fail_before_processing(self):
        cases = (
            ("--min-transcription-confidence", ("-1", "101", "50.5", "nan", "inf"), "between 0 and 100"),
            ("--min-detector-confidence", ("-1", "1.1", "nan", "inf"), "between 0 and 1"),
            ("--request-timeout", ("0", "-1", "nan", "inf", "75s"), "positive number"),
        )
        for option, invalid_values, message in cases:
            for value in invalid_values:
                stderr = io.StringIO()
                with self.subTest(option=option, value=value), contextlib.redirect_stderr(stderr), \
                        patch("uma_bct.cli.keep_awake") as awake, self.assertRaises(SystemExit) as error:
                    main([f"{option}={value}"])
                self.assertEqual(error.exception.code, 2)
                self.assertIn(message, stderr.getvalue())
                awake.assert_not_called()

    def test_help_and_version_do_not_start_processing(self):
        for arguments in (["-h"], ["--help"], ["-hh"], ["--version"], ["setup", "--help"]):
            with self.subTest(arguments=arguments), contextlib.redirect_stdout(io.StringIO()), \
                    patch("uma_bct.cli.keep_awake") as awake, self.assertRaises(SystemExit) as error:
                main(arguments)
            self.assertEqual(error.exception.code, 0)
            awake.assert_not_called()

    def test_hosted_selection_reaches_processing_and_setup(self):
        config = HostedModelConfig(model="anthropic/test-model", api_key="synthetic-key")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            photo = root / "Photo.jpg"
            photo.write_bytes(b"synthetic photo")
            for setup in (False, True):
                options = (["setup", "--hosted"] if setup else
                           ["-i", str(photo), "-o", str(root / "Review.csv"), "--hosted", "--ai-dates"])
                runner = "run_setup" if setup else "run_batch"
                with self.subTest(setup=setup), \
                        patch("uma_bct.cli.hosted_model_from_environment", return_value=config), \
                        patch("uma_bct.adapters.hosted_client.HostedVisionModel"), \
                        patch(f"uma_bct.cli.{runner}", return_value=0) as run, \
                        patch("uma_bct.cli.keep_awake", return_value=contextlib.nullcontext()):
                    self.assertEqual(main(options), 0)
                if setup:
                    run.assert_called_once_with(hosted_model=config)
                else:
                    self.assertIs(run.call_args.args[0].hosted_model, config)
                    self.assertTrue(run.call_args.args[0].ai_dates)

    def test_processing_options_reach_both_output_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            photo = root / "Photo ü #.jpg"
            photo.write_bytes(b"synthetic photo")
            common = [
                "-i", str(photo), "--format", "Excel", "--ai-dates", "-mcustom:8b",
                "-d", "opencv", "-t", "qwen", "-c", "80", "--min-confidence", "0.7",
                "--request-timeout", "75", "--verification", "always", "--no-missing-box", "--json",
            ]
            # Validate real CLI paths, then inspect the arguments handed to each runner.
            for managed in (False, True):
                runner = "run_managed_batch" if managed else "run_batch"
                paths = (["--results-directory", str(root / "Results"), "--reprocess"] if managed else
                         ["-o", str(root / "Review.xlsx"), "-b", str(root / "Crops"), "--force"])
                with self.subTest(managed=managed), \
                        patch(f"uma_bct.cli.{runner}", return_value=0) as run, \
                        patch("uma_bct.cli.keep_awake", return_value=contextlib.nullcontext()):
                    self.assertEqual(main(common + paths), 0)
                arguments = run.call_args.args[0]
                expected = {
                    "ai_dates": True, "file_type": "excel", "model": "custom:8b", "detector": "opencv",
                    "transcriber": "qwen", "min_transcription_confidence": 80, "min_detector_confidence": 0.7,
                    "request_timeout": 75.0, "verification": "always", "detect_missing_boxes": False,
                    "json": True, "reprocess": managed, "force": not managed,
                }
                for attribute, value in expected.items():
                    self.assertEqual(getattr(arguments, attribute), value, attribute)


if __name__ == "__main__":
    unittest.main()
