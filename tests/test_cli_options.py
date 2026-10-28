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
            ("--inputs", "Photos to read", "path_to_inputs", Path("Photos to read")),
            ("-o", "Review.xlsx", "path_to_outputs", Path("Review.xlsx")),
            ("--outputs", "Review.xlsx", "path_to_outputs", Path("Review.xlsx")),
            ("-co", "Review crops", "path_to_box_images", Path("Review crops")),
            ("--crop-outputs", "Review crops", "path_to_box_images", Path("Review crops")),
            ("--batch-outputs", "Saved batches", "results_directory", Path("Saved batches")),
            ("--format", "Excel", "file_type", "excel"),
            ("--file-type", "CSV", "file_type", "csv"),
            ("--verification", "never", "verification", "never"),
            ("--verification", "always", "verification", "always"),
            ("--verification", "ask", "verification", "ask"),
            ("-m", "custom:8b", "model", "custom:8b"),
            ("--model", "custom:8b", "model", "custom:8b"),
            ("-c", "80", "min_transcription_confidence", 80),
            ("--confidence-threshold", "80", "min_transcription_confidence", 80),
            ("--min-confidence", "0.7", "min_detector_confidence", 0.7),
            ("--min-detector-confidence", "0.7", "min_detector_confidence", 0.7),
            ("--timeout", "75.5", "request_timeout", 75.5),
            ("-t", "75.5", "request_timeout", 75.5),
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
            ("--json", "json", True),
            ("--reprocess", "reprocess", True), ("-f", "force", True), ("--force", "force", True),
            ("-r", "reprocess", True),
            ("--disable-missing-box-rows", "detect_missing_boxes", False),
            ("-mb", "detect_missing_boxes", False),
        ):
            with self.subTest(option=option):
                self.assertEqual(getattr(parse_args([option]), attribute), expected)

    def test_invalid_numeric_options_fail_before_processing(self):
        cases = (
            ("--confidence-threshold", ("-1", "101", "50.5", "nan", "inf"), "between 0 and 100"),
            ("--min-detector-confidence", ("-1", "1.1", "nan", "inf"), "between 0 and 1"),
            ("--timeout", ("0", "-1", "nan", "inf", "75s"), "positive number"),
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
                           ["-i", str(photo), "-o", str(root / "Review.csv"), "--hosted", "--ai-dates", "-mb"])
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
                    self.assertFalse(run.call_args.args[0].detect_missing_boxes)
                    self.assertEqual(run.call_args.args[0].model, parse_args([]).model)

    def test_processing_options_reach_both_output_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            photo = root / "Photo ü #.jpg"
            photo.write_bytes(b"synthetic photo")
            common = [
                "-i", str(photo), "--format", "Excel", "--ai-dates", "-mcustom:8b",
                "-c", "80", "--min-confidence", "0.7",
                "-t", "75", "--verification", "always", "-mb", "--json",
            ]
            # Validate real CLI paths, then inspect the arguments handed to each runner.
            for managed in (False, True):
                runner = "run_managed_batch" if managed else "run_batch"
                paths = (["--batch-outputs", str(root / "Results"), "-r"] if managed else
                         ["-o", str(root / "Review.xlsx"), "-co", str(root / "Crops"), "--force"])
                with self.subTest(managed=managed), \
                        patch(f"uma_bct.cli.{runner}", return_value=0) as run, \
                        patch("uma_bct.cli.keep_awake", return_value=contextlib.nullcontext()):
                    self.assertEqual(main(common + paths), 0)
                arguments = run.call_args.args[0]
                expected = {
                    "ai_dates": True, "file_type": "excel", "model": "custom:8b", "detector": "qwen",
                    "transcriber": "qwen", "min_transcription_confidence": 80, "min_detector_confidence": 0.7,
                    "request_timeout": 75.0, "verification": "always", "detect_missing_boxes": False,
                    "json": True, "reprocess": managed, "force": not managed,
                }
                for attribute, value in expected.items():
                    self.assertEqual(getattr(arguments, attribute), value, attribute)

    def test_retired_options_fail_before_processing(self):
        for options in (
            ["--detector", "opencv"], ["-d", "qwen"], ["--transcriber", "qwen"], ["--human"],
            ["-t", "qwen"], ["-b", "Crops"], ["--path-to-inputs", "Photos"],
            ["--path-to-outputs", "catalogue.csv"], ["--path-to-output", "catalogue.csv"],
            ["--path-to-box-images", "Crops"], ["--work-dir", "Crops"],
            ["--results-directory", "Results"], ["--min-transcription-confidence", "80"],
            ["--no-missing-box"], ["--no-missing-box-detection"],
        ):
            with self.subTest(options=options), contextlib.redirect_stderr(io.StringIO()), \
                    patch("uma_bct.cli.keep_awake") as awake, self.assertRaises(SystemExit) as error:
                main(options)
            self.assertEqual(error.exception.code, 2)
            awake.assert_not_called()

    def test_help_promotes_current_names_and_hides_compatibility_aliases(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(SystemExit):
            main(["--help"])
        text = output.getvalue()
        for name in ("--inputs", "--outputs", "--crop-outputs", "--batch-outputs", "--confidence-threshold",
                     "--timeout", "--disable-missing-box-rows", "-co", "-mb"):
            self.assertIn(name, text)
        for name in ("--file-type", "--request-timeout", "--detector", "--transcriber", "--human"):
            self.assertNotIn(name, text)

    def test_multiletter_crop_option_does_not_bypass_managed_path_conflicts(self):
        for options in (["-co", "Crops"], ["-co=Crops"], ["--crop-outputs=Crops"]):
            with self.subTest(options=options), contextlib.redirect_stderr(io.StringIO()) as errors, \
                    self.assertRaises(SystemExit):
                main(["--batch-outputs", "Results", *options])
            self.assertIn("cannot be combined", errors.getvalue())

    def test_setup_uses_the_same_input_names(self):
        for name in ("-i", "--inputs"):
            with self.subTest(name=name), patch("uma_bct.cli.run_setup", return_value=0) as setup, \
                    patch("uma_bct.cli.keep_awake", return_value=contextlib.nullcontext()):
                self.assertEqual(main(["setup", name, "Photos", "--batch-outputs", "Results"]), 0)
            setup.assert_called_once_with(input_directory=Path("Photos"), results_directory=Path("Results"))


if __name__ == "__main__":
    unittest.main()
