"""Hosted-model configuration must never silently run a different model."""

import contextlib
import io
import unittest
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

    def test_cli_rejects_key_before_processing_until_adapter_exists(self):
        key = "secret-test-key"
        for model, expected in (("", "UMA_BCT_API_MODEL"), ("gemini/example", "not available yet")):
            with self.subTest(model=model), patch.dict(
                "os.environ", {"UMA_BCT_API_KEY": key, "UMA_BCT_API_MODEL": model}
            ), patch("uma_bct.cli.run_batch") as run_batch, contextlib.redirect_stderr(io.StringIO()) as errors:
                with self.assertRaises(SystemExit) as raised:
                    main([])
                self.assertEqual(raised.exception.code, 2)
                self.assertIn(expected, errors.getvalue())
                self.assertNotIn(key, errors.getvalue())
                run_batch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
