"""CLI output and batch checkpoints must resolve the same application version."""

import contextlib
import io
import unittest
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from unittest.mock import patch

from uma_bct.cli import main, parse_args
from uma_bct.managed_batch import initial_progress
from uma_bct.version import package_version


class PackageVersionTest(unittest.TestCase):
    def assert_cli_and_checkpoint_version(self, expected):
        arguments = parse_args([])
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as raised:
            main(["--version"])
        self.assertEqual(raised.exception.code, 0)
        self.assertEqual(output.getvalue().strip(), expected)
        progress = initial_progress(arguments, Path("Results/Batch example"), [], 600.0)
        self.assertEqual(progress["application"], {"name": "uma-bct-cli", "version": expected})

    def test_installed_version_is_used_by_cli_and_checkpoint(self):
        with patch("uma_bct.version.version", return_value="9.8.7"):
            self.assert_cli_and_checkpoint_version("9.8.7")

    def test_missing_metadata_uses_existing_fallback_in_cli_and_checkpoint(self):
        with patch("uma_bct.version.version", side_effect=PackageNotFoundError("uma-bct-cli")):
            self.assert_cli_and_checkpoint_version("0.1.0")

    def test_other_metadata_errors_are_not_hidden_by_fallback(self):
        with patch("uma_bct.version.version", side_effect=ValueError("invalid metadata")):
            with self.assertRaisesRegex(ValueError, "invalid metadata"):
                package_version()
