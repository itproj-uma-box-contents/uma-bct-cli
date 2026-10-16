"""Hosted-model requests use only sanitized images and safe response handling."""

import base64
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from helpers import make_box, make_image
from PIL import Image

from uma_bct.adapters.hosted_client import PROVIDER_API_BASES, HostedVisionModel
from uma_bct.adapters.qwen_date_detector import QwenDateDetector
from uma_bct.adapters.qwen_transcriber import QwenTextTranscriber
from uma_bct.model_config import HostedModelConfig
from uma_bct.util.date_util import Date


class ProviderError(Exception):
    def __init__(self, status_code, message="secret-test-key private-image-text"):
        super().__init__(message)
        self.status_code = status_code


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
            supports_response_schema=Mock(return_value=False),
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
                expected_base = (
                    "https://generativelanguage.googleapis.com/v1beta"
                    if provider == "gemini" else PROVIDER_API_BASES[provider]
                )
                self.assertEqual(arguments["api_base"], expected_base)
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

    def test_date_detector_sends_text_only_to_hosted_providers(self):
        response = json.dumps({"dates": [{
            "raw_text": "1950", "starting_index": 0,
            "start_date": {"year": 1950}, "end_date": {},
        }]})
        for provider in PROVIDER_API_BASES:
            with self.subTest(provider=provider):
                sdk = self.sdk(content=response)
                sdk.supports_response_schema.return_value = True
                client = HostedVisionModel(HostedModelConfig(model=f"{provider}/example", api_key="secret-test-key"))
                with patch("uma_bct.adapters.hosted_client._load_litellm", return_value=sdk):
                    dates = QwenDateDetector(client).extract_dates("1950")
                self.assertEqual(dates, (Date(year=1950),))
                content = sdk.completion.call_args.kwargs["messages"][0]["content"]
                self.assertEqual(len(content), 1)
                self.assertEqual(content[0]["type"], "text")
                self.assertIn("1950", content[0]["text"])
                self.assertEqual(sdk.completion.call_args.kwargs["response_format"], {"type": "json_object"})

    def test_schema_capability_uses_portable_strict_structure(self):
        sdk = self.sdk()
        sdk.supports_response_schema.return_value = True
        schema = {
            "type": "object", "properties": {"boxes": {
                "type": "array", "maxItems": 100, "items": {"type": "object", "properties": {
                    "name": {"type": "string", "maxLength": 10},
                    "coords": {"type": "array", "minItems": 4, "maxItems": 4, "items": {"type": "number"}},
                }, "required": ["name", "coords"]},
            }}, "required": ["boxes"],
        }
        client = HostedVisionModel(HostedModelConfig(model="openai/gpt-test", api_key="secret"))
        with patch("uma_bct.adapters.hosted_client._load_litellm", return_value=sdk):
            client.query(self.image_path, "Find boxes.", schema)
        reply_format = sdk.completion.call_args.kwargs["response_format"]
        self.assertEqual(reply_format["type"], "json_schema")
        self.assertTrue(reply_format["json_schema"]["strict"])
        portable = reply_format["json_schema"]["schema"]
        self.assertFalse(portable["additionalProperties"])
        self.assertNotIn("maxItems", portable["properties"]["boxes"])
        item = portable["properties"]["boxes"]["items"]
        self.assertFalse(item["additionalProperties"])
        self.assertNotIn("maxLength", item["properties"]["name"])
        self.assertNotIn("minItems", item["properties"]["coords"])
        self.assertIn("maxItems", schema["properties"]["boxes"])

    def test_schema_lookup_failure_falls_back_to_json_object(self):
        sdk = self.sdk()
        sdk.supports_response_schema.side_effect = ValueError("unsupported")
        client = HostedVisionModel(HostedModelConfig(model="anthropic/new-model", api_key="secret"))
        with patch("uma_bct.adapters.hosted_client._load_litellm", return_value=sdk):
            client.query(self.image_path, "Read box.", self.schema)
        self.assertEqual(sdk.completion.call_args.kwargs["response_format"], {"type": "json_object"})

    def test_preflight_checks_dependency_without_request(self):
        sdk = self.sdk()
        client = HostedVisionModel(HostedModelConfig(model="gemini/example", api_key="secret"))
        with patch("uma_bct.adapters.hosted_client._load_litellm", return_value=sdk):
            metadata = client.preflight()
        self.assertEqual(metadata["model"], "gemini/example")
        self.assertEqual(metadata["backend"], "hosted")
        self.assertEqual(metadata["api_base"], PROVIDER_API_BASES["gemini"])
        sdk.completion.assert_not_called()

    def test_malformed_probability_metadata_keeps_the_valid_reply(self):
        content = '{"boxes":[]}'
        malformed = (42, "invalid metadata", {"token": content},
                     [{"token": content, "logprob": -(10 ** 400)}])
        for provider in PROVIDER_API_BASES:
            for logprobs in malformed:
                with self.subTest(provider=provider, metadata=type(logprobs).__name__):
                    sdk = self.sdk(content=content, logprobs=logprobs, params=("logprobs",))
                    client = HostedVisionModel(HostedModelConfig(model=f"{provider}/example", api_key="synthetic-key"))
                    with patch("uma_bct.adapters.hosted_client._load_litellm", return_value=sdk):
                        result = client.query(self.image_path, "Synthetic check.", self.schema, logprobs=True)
                    self.assertEqual(result.value, {"boxes": []})
                    self.assertEqual(result.token_logprobs, ())

    def test_client_load_suppresses_sdk_stdout_feedback(self):
        sdk = self.sdk()
        sdk.suppress_debug_info = False
        client = HostedVisionModel(HostedModelConfig(model="gemini/example", api_key="secret"))
        with patch.dict("sys.modules", {"litellm": sdk}):
            client.preflight()
        self.assertTrue(sdk.suppress_debug_info)
        sdk.completion.assert_not_called()

    def test_temporary_server_error_retries_the_same_request(self):
        for status in (500, 502, 503, 504):
            sdk = self.sdk()
            sdk.completion.side_effect = [ProviderError(status), sdk.completion.return_value]
            client = HostedVisionModel(HostedModelConfig(model="gemini/example", api_key="secret"))
            with self.subTest(status=status), patch(
                "uma_bct.adapters.hosted_client._load_litellm", return_value=sdk
            ), patch("uma_bct.adapters.hosted_client.time.sleep") as sleep:
                result = client.query(self.image_path, "Read box.", self.schema)
            self.assertEqual(result.value, {"boxes": []})
            self.assertEqual(sdk.completion.call_count, 2)
            self.assertEqual(sdk.completion.call_args_list[0], sdk.completion.call_args_list[1])
            sleep.assert_called_once()
            self.assertGreaterEqual(sleep.call_args.args[0], 2)
            self.assertLessEqual(sleep.call_args.args[0], 3)

    def test_server_retries_are_bounded_and_do_not_expose_reply_text(self):
        sdk = self.sdk()
        sdk.completion.side_effect = ProviderError(503)
        client = HostedVisionModel(HostedModelConfig(model="gemini/example", api_key="secret-test-key"))
        with patch("uma_bct.adapters.hosted_client._load_litellm", return_value=sdk), patch(
            "uma_bct.adapters.hosted_client.time.sleep"
        ) as sleep, self.assertRaises(RuntimeError) as raised:
            client.query(self.image_path, "Read box.", self.schema)
        self.assertEqual(sdk.completion.call_count, 3)
        self.assertEqual(sleep.call_count, 2)
        self.assertIn("HTTP 503", str(raised.exception))
        self.assertIn("high demand", str(raised.exception))
        self.assertNotIn("secret-test-key", str(raised.exception))
        self.assertNotIn("private-image-text", str(raised.exception))

    def test_client_and_quota_errors_are_not_retried(self):
        for status in (400, 401, 403, 404, 429):
            sdk = self.sdk(params=("response_format", "logprobs"))
            sdk.completion.side_effect = ProviderError(status)
            client = HostedVisionModel(HostedModelConfig(model="gemini/example", api_key="secret"))
            with self.subTest(status=status), patch(
                "uma_bct.adapters.hosted_client._load_litellm", return_value=sdk
            ), patch("uma_bct.adapters.hosted_client.time.sleep") as sleep, self.assertRaises(RuntimeError) as raised:
                client.query(self.image_path, "Read box.", self.schema, logprobs=True)
            sdk.completion.assert_called_once()
            sleep.assert_not_called()
            self.assertIn(f"HTTP {status}", str(raised.exception))
            self.assertNotIn("secret-test-key", str(raised.exception))

    def test_hosted_logprobs_rejection_still_transcribes_and_flags_review(self):
        content = json.dumps({"boxes": [{
            "position": "left", "record_identifiers": ["1960.0009"], "unit": "4",
            "typed": ["1960.0009", "Unit 4"], "handwritten": [], "auxiliary_annotations": [],
        }]})
        for provider in PROVIDER_API_BASES:
            for message in (
                "Logprobs is not supported for this model.",
                "Logprobs is not enabled for this model.",
                "Unsupported parameter: 'logprobs' is not supported with this model.",
            ):
                sdk = self.sdk(content, params=("response_format", "logprobs"))
                sdk.completion.side_effect = [ProviderError(400, message), sdk.completion.return_value]
                client = HostedVisionModel(HostedModelConfig(model=f"{provider}/example", api_key="secret"))
                box = make_box(image=make_image(self.image_path), crop_path=self.image_path)
                with self.subTest(provider=provider, message=message), patch(
                    "uma_bct.adapters.hosted_client._load_litellm", return_value=sdk
                ):
                    record = QwenTextTranscriber(client).transcribe(box)[0]
                self.assertEqual(record.unit_number, "4")
                self.assertEqual(record.confidence, 0)
                self.assertTrue(record.needs_review)
                self.assertEqual(sdk.completion.call_count, 2)
                first, second = (call.kwargs for call in sdk.completion.call_args_list)
                self.assertTrue(first["logprobs"])
                self.assertNotIn("logprobs", second)
                self.assertEqual({key: value for key, value in first.items() if key != "logprobs"}, second)

    def test_rejected_logprobs_retry_is_not_repeated(self):
        sdk = self.sdk(params=("response_format", "logprobs"))
        sdk.completion.side_effect = ProviderError(400, "Logprobs is not supported for this model.")
        client = HostedVisionModel(HostedModelConfig(model="gemini/example", api_key="secret"))
        with patch("uma_bct.adapters.hosted_client._load_litellm", return_value=sdk), self.assertRaises(RuntimeError):
            client.query(self.image_path, "Read box.", self.schema, logprobs=True)
        self.assertEqual(sdk.completion.call_count, 2)

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

    def test_missing_provider_logprobs_force_human_review(self):
        content = json.dumps({"boxes": [{
            "position": "left", "record_identifiers": ["1960.0009"], "unit": "4",
            "typed": ["1960.0009", "Unit 4"], "handwritten": [], "auxiliary_annotations": [],
        }]})
        sdk = self.sdk(content, params=("response_format",))
        client = HostedVisionModel(HostedModelConfig(model="anthropic/claude-test", api_key="secret"))
        box = make_box(image=make_image(self.image_path), crop_path=self.image_path)
        with patch("uma_bct.adapters.hosted_client._load_litellm", return_value=sdk):
            record = QwenTextTranscriber(client).transcribe(box)[0]
        self.assertEqual(record.confidence, 0)
        self.assertTrue(record.needs_review)
        self.assertEqual(record.unit_number, "4")
        self.assertNotIn("logprobs", sdk.completion.call_args.kwargs)

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
