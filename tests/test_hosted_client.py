"""Hosted-model requests use only sanitized images and safe response handling."""

import base64
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image

from uma_bct.adapters.hosted_client import PROVIDER_API_BASES, HostedVisionModel
from uma_bct.model_config import HostedModelConfig


class HostedVisionModelTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.image_path = Path(self.directory.name) / "clean.png"
        Image.new("RGB", (2, 2), "white").save(self.image_path)
        self.schema = {"type": "object", "properties": {"boxes": {"type": "array"}}, "required": ["boxes"]}

    @staticmethod
    def sdk(content='{"boxes":[]}', *, params=("response_format",), logprobs=None, finish_reason="stop"):
        response = SimpleNamespace(
            choices=[SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=content),
                logprobs=SimpleNamespace(content=logprobs) if logprobs is not None else None,
            )]
        )
        return SimpleNamespace(
            completion=Mock(return_value=response),
            get_supported_openai_params=Mock(return_value=params),
        )

    def test_three_providers_use_their_fixed_https_origins_and_inline_png(self):
        for provider in PROVIDER_API_BASES:
            with self.subTest(provider=provider):
                sdk = self.sdk()
                client = HostedVisionModel(HostedModelConfig(model=f"{provider}/example", api_key="secret-test-key"))
                with patch("uma_bct.adapters.hosted_client._load_litellm", return_value=sdk):
                    result = client.query(self.image_path, "Find boxes.", self.schema, validator=lambda value: value["boxes"])
                self.assertEqual(result.value, {"boxes": []})
                self.assertEqual(result.token_logprobs, ())
                arguments = sdk.completion.call_args.kwargs
                self.assertEqual(arguments["model"], f"{provider}/example")
                self.assertEqual(arguments["api_base"], PROVIDER_API_BASES[provider])
                self.assertEqual(arguments["api_key"], "secret-test-key")
                self.assertEqual(arguments["response_format"], {"type": "json_object"})
                self.assertEqual(arguments["num_retries"], 0)
                content = arguments["messages"][0]["content"]
                self.assertIn(json.dumps(self.schema, separators=(",", ":")), content[0]["text"])
                image_url = content[1]["image_url"]["url"]
                self.assertTrue(image_url.startswith("data:image/png;base64,"))
                self.assertEqual(base64.b64decode(image_url.split(",", 1)[1]), self.image_path.read_bytes())
                self.assertNotIn(str(self.image_path), repr(arguments["messages"]))
                self.assertNotIn("secret-test-key", repr(client))

    def test_unknown_provider_and_invalid_timeout_fail_before_request(self):
        with self.assertRaisesRegex(ValueError, "openai/"):
            HostedVisionModel(HostedModelConfig(model="openrouter/example", api_key="secret"))
        with self.assertRaisesRegex(ValueError, "positive"):
            HostedVisionModel(HostedModelConfig(model="gemini/example", api_key="secret"), timeout_seconds=0)

    def test_unknown_model_capabilities_use_prompt_only(self):
        sdk = self.sdk()
        sdk.get_supported_openai_params.side_effect = ValueError("unknown model")
        client = HostedVisionModel(HostedModelConfig(model="anthropic/new-model", api_key="secret"))
        with patch("uma_bct.adapters.hosted_client._load_litellm", return_value=sdk):
            self.assertEqual(client.query(self.image_path, "Read box.", self.schema).value, {"boxes": []})
        self.assertNotIn("response_format", sdk.completion.call_args.kwargs)
        self.assertNotIn("logprobs", sdk.completion.call_args.kwargs)

    def test_logprobs_are_used_only_when_complete_and_requested(self):
        content = '{"boxes":[]}'
        tokens = [{"token": character, "logprob": -0.1} for character in content]
        sdk = self.sdk(content, params=("response_format", "logprobs"), logprobs=tokens)
        client = HostedVisionModel(HostedModelConfig(model="openai/gpt-test", api_key="secret"))
        with patch("uma_bct.adapters.hosted_client._load_litellm", return_value=sdk):
            result = client.query(self.image_path, "Read box.", self.schema, logprobs=True)
        self.assertEqual(len(result.token_logprobs), len(content))
        self.assertTrue(sdk.completion.call_args.kwargs["logprobs"])
        sdk.completion.return_value.choices[0].logprobs.content = tokens[:-1]
        with patch("uma_bct.adapters.hosted_client._load_litellm", return_value=sdk):
            result = client.query(self.image_path, "Read box.", self.schema, logprobs=True)
        self.assertEqual(result.token_logprobs, ())

    def test_provider_errors_and_invalid_replies_do_not_expose_secrets(self):
        sdk = self.sdk()
        sdk.completion.side_effect = RuntimeError("secret-test-key private-image-text")
        client = HostedVisionModel(HostedModelConfig(model="gemini/example", api_key="secret-test-key"))
        with patch("uma_bct.adapters.hosted_client._load_litellm", return_value=sdk):
            with self.assertRaises(RuntimeError) as raised:
                client.query(self.image_path, "Read box.", self.schema)
        self.assertNotIn("secret-test-key", str(raised.exception))
        self.assertNotIn("private-image-text", str(raised.exception))
        for content, finish_reason in (("{broken", "stop"), ('{"boxes":[]}', "length")):
            sdk = self.sdk(content, finish_reason=finish_reason)
            with self.subTest(content=content, finish_reason=finish_reason), patch(
                "uma_bct.adapters.hosted_client._load_litellm", return_value=sdk
            ), self.assertRaises(ValueError):
                client.query(self.image_path, "Read box.", self.schema)

    def test_non_png_image_is_rejected_before_sdk_load(self):
        client = HostedVisionModel(HostedModelConfig(model="openai/example", api_key="secret"))
        other = self.image_path.with_suffix(".jpg")
        other.write_bytes(self.image_path.read_bytes())
        with patch("uma_bct.adapters.hosted_client._load_litellm") as load:
            with self.assertRaisesRegex(ValueError, "sanitized PNG"):
                client.query(other, "Read box.", self.schema)
            self.image_path.write_bytes(b"not a png")
            with self.assertRaisesRegex(ValueError, "not a PNG"):
                client.query(self.image_path, "Read box.", self.schema)
        load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
