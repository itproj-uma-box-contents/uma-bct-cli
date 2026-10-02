"""Hosted-model configuration must never silently run a different model."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from uma_bct.cli import main
from uma_bct.model_config import hosted_model_from_environment


class HostedModelConfigTest(unittest.TestCase):
    def test_empty_key_keeps_local_mode(self):
        self.assertIsNone(hosted_model_from_environment({}))
        self.assertIsNone(hosted_model_from_environment({"UMA_BCT_API_KEY": "  ", "UMA_BCT_API_MODEL": "gemini/example"}))

    def test_key_requires_provider_qualified_model(self):
        key = "secret-test-key"
        for model in ("", "example", "gemini/", "/example", "gemini//example", "gemini/a b"):
            with self.subTest(model=model), self.assertRaisesRegex(ValueError, "UMA_BCT_API_MODEL"):
                hosted_model_from_environment({"UMA_BCT_API_KEY": key, "UMA_BCT_API_MODEL": model})

    def test_valid_configuration_keeps_key_out_of_repr(self):
        key = "secret-test-key"
        config = hosted_model_from_environment({"UMA_BCT_API_KEY": key, "UMA_BCT_API_MODEL": " openrouter/openrouter/free "})
        self.assertIsNotNone(config)
        self.assertEqual(config.model, "openrouter/openrouter/free")
        self.assertEqual(config.provider, "openrouter")
        self.assertEqual(config.api_key, key)
        self.assertNotIn(key, repr(config))

    def test_key_routes_direct_and_setup_entry_points(self):
        environment = {"UMA_BCT_API_KEY": "secret-test-key", "UMA_BCT_API_MODEL": "gemini/vision-test"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "shelf.jpg"
            image.write_bytes(b"unused test image")
            with patch.dict("os.environ", environment), patch("uma_bct.cli.run_batch", return_value=0) as run_batch:
                self.assertEqual(main(["--json", "--path-to-inputs", str(image), "--path-to-outputs", str(root / "out.csv")]), 0)
            selected = run_batch.call_args.args[0].hosted_model
            self.assertEqual(selected.model, "gemini/vision-test")
            self.assertEqual(selected.api_key, "secret-test-key")
            with patch.dict("os.environ", environment), patch("uma_bct.cli.run_setup", return_value=0) as setup:
                self.assertEqual(main(["setup"]), 0)
            self.assertEqual(setup.call_args.kwargs["hosted_model"], selected)

    def test_cli_requires_model_and_rejects_local_model_flag_in_hosted_mode(self):
        key = "secret-test-key"
        for model, argv, expected in (
            ("", [], "UMA_BCT_API_MODEL"),
            ("gemini/example", ["--model", "local"], "--model selects local Ollama"),
            ("openrouter/example", [], "currently support"),
        ):
            with self.subTest(model=model), patch.dict(
                "os.environ", {"UMA_BCT_API_KEY": key, "UMA_BCT_API_MODEL": model}
            ), patch("uma_bct.cli.run_batch") as run_batch, contextlib.redirect_stderr(io.StringIO()) as errors:
                with self.assertRaises(SystemExit) as raised:
                    main(argv)
                self.assertEqual(raised.exception.code, 2)
                self.assertIn(expected, errors.getvalue())
                self.assertNotIn(key, errors.getvalue())
                run_batch.assert_not_called()



if __name__ == "__main__":
    unittest.main()
