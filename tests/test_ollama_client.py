import contextlib
import io
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from uma_bct.adapters.ollama_client import (
    DEFAULT_MODEL,
    OllamaVisionModel,
    normalize_ollama_host,
    timeout_from_environment,
    version_tuple,
)


class OllamaPreflightTest(unittest.TestCase):
    def test_preflight_reports_server_model_and_digest(self):
        client = SimpleNamespace(
            list=Mock(
                return_value=SimpleNamespace(
                    models=[SimpleNamespace(model=DEFAULT_MODEL, digest="sha256:abc")]
                )
            ),
            close=Mock(),
        )
        client_factory = Mock(return_value=client)
        fake_ollama = SimpleNamespace(Client=client_factory)

        with (
            patch.dict(sys.modules, {"ollama": fake_ollama}),
            patch(
                "uma_bct.adapters.ollama_client.fetch_ollama_version",
                return_value="0.12.8",
            ),
        ):
            metadata = OllamaVisionModel(timeout_seconds=42).preflight()

        self.assertEqual(metadata["model"], DEFAULT_MODEL)
        self.assertEqual(metadata["model_digest"], "sha256:abc")
        self.assertEqual(metadata["ollama_version"], "0.12.8")
        client.close.assert_called_once_with()
        client_factory.assert_called_once_with(
            host="http://127.0.0.1:11434",
            timeout=42,
            follow_redirects=False,
            trust_env=False,
        )

    def test_preflight_rejects_missing_model_with_pull_command(self):
        client = SimpleNamespace(
            list=Mock(return_value=SimpleNamespace(models=[])),
            close=Mock(),
        )
        fake_ollama = SimpleNamespace(Client=Mock(return_value=client))

        with (
            patch.dict(sys.modules, {"ollama": fake_ollama}),
            patch(
                "uma_bct.adapters.ollama_client.fetch_ollama_version",
                return_value="0.12.8",
            ),
            self.assertRaisesRegex(RuntimeError, f"ollama pull {DEFAULT_MODEL}"),
        ):
            OllamaVisionModel().preflight()

    def test_preflight_rejects_old_server(self):
        client = SimpleNamespace(
            list=Mock(
                return_value=SimpleNamespace(
                    models=[SimpleNamespace(model=DEFAULT_MODEL, digest="digest")]
                )
            ),
            close=Mock(),
        )
        fake_ollama = SimpleNamespace(Client=Mock(return_value=client))

        with (
            patch.dict(sys.modules, {"ollama": fake_ollama}),
            patch(
                "uma_bct.adapters.ollama_client.fetch_ollama_version",
                return_value="0.11.0",
            ),
            self.assertRaisesRegex(RuntimeError, "0.12.7 or newer"),
        ):
            OllamaVisionModel().preflight()

    def test_stream_progress_goes_to_stderr_only(self):
        chunks = [SimpleNamespace(message={"content": ""}) for _ in range(20)]
        chunks.append(SimpleNamespace(message={"content": '{"boxes": []}'}))
        client = SimpleNamespace(chat=Mock(return_value=iter(chunks)), close=Mock())
        fake_ollama = SimpleNamespace(Client=Mock(return_value=client))
        stdout = io.StringIO()
        stderr = io.StringIO()

        with (
            patch.dict(sys.modules, {"ollama": fake_ollama}),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            value = OllamaVisionModel().query(
                Path("box.png"), "prompt", {"type": "object"}
            )

        self.assertEqual(value, {"boxes": []})
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(stderr.getvalue(), ".")
        client.close.assert_called_once_with()


class OllamaSettingsTest(unittest.TestCase):
    def test_timeout_environment_requires_a_positive_number(self):
        with (
            patch.dict("os.environ", {"UMA_BCT_OLLAMA_TIMEOUT_SECONDS": "0"}),
            self.assertRaisesRegex(ValueError, "positive number"),
        ):
            timeout_from_environment()

    def test_timeout_environment_rejects_non_finite_values(self):
        for value in ("nan", "inf", "-inf"):
            with (
                self.subTest(value=value),
                patch.dict(
                    "os.environ",
                    {"UMA_BCT_OLLAMA_TIMEOUT_SECONDS": value},
                ),
                self.assertRaisesRegex(ValueError, "positive number"),
            ):
                timeout_from_environment()

    def test_direct_client_timeout_must_be_finite(self):
        with self.assertRaisesRegex(ValueError, "positive finite"):
            OllamaVisionModel(timeout_seconds=float("nan"))

    def test_remote_ollama_requires_explicit_privacy_override(self):
        with (
            patch.dict("os.environ", {}, clear=True),
            self.assertRaisesRegex(ValueError, "sends archive image data off-device"),
        ):
            normalize_ollama_host("https://models.example.test:11434")

        with patch.dict(
            "os.environ",
            {"UMA_BCT_ALLOW_REMOTE_OLLAMA": "1"},
            clear=True,
        ):
            self.assertEqual(
                normalize_ollama_host("https://models.example.test:11434"),
                "https://models.example.test:11434",
            )

    def test_only_numeric_loopback_and_normalized_localhost_are_allowed_by_default(
        self,
    ):
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(
                normalize_ollama_host("127.0.0.1:11434"),
                "http://127.0.0.1:11434",
            )
            self.assertEqual(
                normalize_ollama_host("localhost"),
                "http://127.0.0.1:11434",
            )
            self.assertEqual(
                normalize_ollama_host("http://[::1]"),
                "http://[::1]:11434",
            )
            for unsafe_hostname in ("ollama", "host.docker.internal", "x.localhost"):
                with (
                    self.subTest(hostname=unsafe_hostname),
                    self.assertRaises(ValueError),
                ):
                    normalize_ollama_host(unsafe_hostname)

    def test_compose_opt_in_permits_only_the_internal_ollama_hostname(self):
        with patch.dict(
            "os.environ",
            {"UMA_BCT_ALLOW_CONTAINER_OLLAMA": "1"},
            clear=True,
        ):
            self.assertEqual(
                normalize_ollama_host("http://ollama:11434"),
                "http://ollama:11434",
            )
            with self.assertRaises(ValueError):
                normalize_ollama_host("http://host.docker.internal:11434")

    def test_remote_privacy_opt_in_still_requires_https(self):
        with (
            patch.dict(
                "os.environ",
                {"UMA_BCT_ALLOW_REMOTE_OLLAMA": "1"},
                clear=True,
            ),
            self.assertRaisesRegex(ValueError, "must use HTTPS"),
        ):
            normalize_ollama_host("http://models.example.test:11434")

    def test_model_stream_has_character_and_chunk_safety_limits(self):
        client = SimpleNamespace(
            chat=Mock(
                return_value=iter(
                    [
                        SimpleNamespace(message={"content": "1234"}),
                        SimpleNamespace(message={"content": "5678"}),
                    ]
                )
            ),
            close=Mock(),
        )
        fake_ollama = SimpleNamespace(Client=Mock(return_value=client))

        with (
            patch.dict(sys.modules, {"ollama": fake_ollama}),
            patch(
                "uma_bct.adapters.ollama_client.MAX_MODEL_RESPONSE_CHARACTERS",
                5,
            ),
            self.assertRaisesRegex(RuntimeError, "4 MiB safety limit"),
        ):
            OllamaVisionModel().query(Path("box.png"), "prompt", {"type": "object"})

        client.chat.return_value = iter(
            [
                SimpleNamespace(message={"content": ""}),
                SimpleNamespace(message={"content": "{}"}),
            ]
        )
        with (
            patch.dict(sys.modules, {"ollama": fake_ollama}),
            patch("uma_bct.adapters.ollama_client.MAX_MODEL_STREAM_CHUNKS", 1),
            self.assertRaisesRegex(RuntimeError, "stream-chunk limit"),
        ):
            OllamaVisionModel().query(Path("box.png"), "prompt", {"type": "object"})

    def test_ollama_host_rejects_credentials_without_echoing_them(self):
        with self.assertRaisesRegex(
            ValueError, "must not contain credentials"
        ) as raised:
            normalize_ollama_host("http://secret-user:secret-pass@localhost:11434")

        self.assertNotIn("secret-pass", str(raised.exception))

    def test_version_parser_accepts_prefixed_versions(self):
        self.assertEqual(version_tuple("ollama version 0.12.7"), (0, 12, 7))


if __name__ == "__main__":
    unittest.main()
