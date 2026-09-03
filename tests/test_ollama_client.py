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
    def test_ensure_model_skips_download_when_model_is_installed(self):
        client = SimpleNamespace(
            list=Mock(
                return_value=SimpleNamespace(
                    models=[SimpleNamespace(model=DEFAULT_MODEL, digest="sha256:abc")]
                )
            ),
            pull=Mock(),
            close=Mock(),
        )
        fake_ollama = SimpleNamespace(Client=Mock(return_value=client))

        with (
            patch.dict(sys.modules, {"ollama": fake_ollama}),
            patch(
                "uma_bct.adapters.ollama_client.fetch_ollama_version",
                return_value="0.12.8",
            ),
        ):
            metadata = OllamaVisionModel().ensure_model(progress_stream=io.StringIO())

        self.assertEqual(metadata["model_digest"], "sha256:abc")
        client.pull.assert_not_called()

    def test_ensure_model_downloads_only_a_missing_model_with_stderr_progress(self):
        missing = SimpleNamespace(models=[])
        installed = SimpleNamespace(
            models=[SimpleNamespace(model=DEFAULT_MODEL, digest="sha256:new")]
        )
        initial_inventory = SimpleNamespace(
            list=Mock(return_value=missing), close=Mock()
        )
        pull_client = SimpleNamespace(
            pull=Mock(
                return_value=iter(
                    [
                        {
                            "status": "\x1b[31m\npulling manifest",
                            "completed": None,
                            "total": None,
                        },
                        {"status": "downloading", "completed": 50, "total": 100},
                    ]
                )
            ),
            close=Mock(),
        )
        final_inventory = SimpleNamespace(
            list=Mock(return_value=installed),
            close=Mock(),
        )
        fake_ollama = SimpleNamespace(
            Client=Mock(side_effect=[initial_inventory, pull_client, final_inventory])
        )
        progress = io.StringIO()

        with (
            patch.dict(sys.modules, {"ollama": fake_ollama}),
            patch(
                "uma_bct.adapters.ollama_client.fetch_ollama_version",
                return_value="0.12.8",
            ),
        ):
            metadata = OllamaVisionModel().ensure_model(progress_stream=progress)

        pull_client.pull.assert_called_once_with(model=DEFAULT_MODEL, stream=True)
        self.assertEqual(metadata["model_digest"], "sha256:new")
        self.assertIn("pulling manifest", progress.getvalue())
        self.assertIn("50%", progress.getvalue())
        self.assertNotIn("\x1b", progress.getvalue())
        initial_inventory.close.assert_called_once_with()
        pull_client.close.assert_called_once_with()
        final_inventory.close.assert_called_once_with()

    def test_model_download_failure_is_actionable_and_closes_clients(self):
        def failing_updates():
            yield {"status": "starting"}
            raise OSError("connection lost")

        inventory_client = SimpleNamespace(
            list=Mock(return_value=SimpleNamespace(models=[])),
            close=Mock(),
        )
        pull_client = SimpleNamespace(
            pull=Mock(return_value=failing_updates()),
            close=Mock(),
        )
        fake_ollama = SimpleNamespace(
            Client=Mock(side_effect=[inventory_client, pull_client])
        )

        with (
            patch.dict(sys.modules, {"ollama": fake_ollama}),
            patch(
                "uma_bct.adapters.ollama_client.fetch_ollama_version",
                return_value="0.12.8",
            ),
            self.assertRaisesRegex(RuntimeError, f"ollama pull {DEFAULT_MODEL}"),
        ):
            OllamaVisionModel().ensure_model(progress_stream=io.StringIO())

        inventory_client.close.assert_called_once_with()
        pull_client.close.assert_called_once_with()

    def test_model_must_be_visible_after_a_reported_download(self):
        first_inventory = SimpleNamespace(
            list=Mock(return_value=SimpleNamespace(models=[])),
            close=Mock(),
        )
        pull_client = SimpleNamespace(
            pull=Mock(return_value=iter([])),
            close=Mock(),
        )
        final_inventory = SimpleNamespace(
            list=Mock(return_value=SimpleNamespace(models=[])),
            close=Mock(),
        )
        fake_ollama = SimpleNamespace(
            Client=Mock(side_effect=[first_inventory, pull_client, final_inventory])
        )

        with (
            patch.dict(sys.modules, {"ollama": fake_ollama}),
            patch(
                "uma_bct.adapters.ollama_client.fetch_ollama_version",
                return_value="0.12.8",
            ),
            self.assertRaisesRegex(RuntimeError, "still unavailable"),
        ):
            OllamaVisionModel().ensure_model(progress_stream=io.StringIO())

        first_inventory.close.assert_called_once_with()
        pull_client.close.assert_called_once_with()
        final_inventory.close.assert_called_once_with()

    def test_model_download_bounds_terminal_output_without_aborting(self):
        initial_inventory = SimpleNamespace(
            list=Mock(return_value=SimpleNamespace(models=[])),
            close=Mock(),
        )
        pull_client = SimpleNamespace(
            pull=Mock(
                return_value=iter(
                    [
                        SimpleNamespace(status="first"),
                        SimpleNamespace(status="second"),
                    ]
                )
            ),
            close=Mock(),
        )
        final_inventory = SimpleNamespace(
            list=Mock(
                return_value=SimpleNamespace(
                    models=[SimpleNamespace(model=DEFAULT_MODEL, digest="ready")]
                )
            ),
            close=Mock(),
        )
        fake_ollama = SimpleNamespace(
            Client=Mock(side_effect=[initial_inventory, pull_client, final_inventory])
        )
        progress = io.StringIO()

        with (
            patch.dict(sys.modules, {"ollama": fake_ollama}),
            patch(
                "uma_bct.adapters.ollama_client.fetch_ollama_version",
                return_value="0.12.8",
            ),
            patch("uma_bct.adapters.ollama_client.MAX_MODEL_PULL_PROGRESS_LINES", 1),
        ):
            metadata = OllamaVisionModel().ensure_model(progress_stream=progress)

        self.assertEqual(metadata["model_digest"], "ready")
        self.assertIn("first", progress.getvalue())
        self.assertNotIn("second", progress.getvalue())
        self.assertIn("suppressed", progress.getvalue())

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

    def test_stream_uses_valid_thinking_when_content_is_not_json(self):
        chunks = [
            SimpleNamespace(
                message=SimpleNamespace(
                    content="not JSON",
                    thinking='analysis before {"boxes": []} analysis after',
                )
            )
        ]
        client = SimpleNamespace(chat=Mock(return_value=iter(chunks)), close=Mock())
        fake_ollama = SimpleNamespace(Client=Mock(return_value=client))

        with patch.dict(sys.modules, {"ollama": fake_ollama}):
            value = OllamaVisionModel().query(
                Path("box.png"), "prompt", {"type": "object"}
            )

        self.assertEqual(value, {"boxes": []})

    def test_stream_prefers_valid_content_over_valid_thinking(self):
        chunk = SimpleNamespace(
            message={
                "content": '{"source": "content"}',
                "thinking": '{"source": "thinking"}',
            }
        )
        client = SimpleNamespace(chat=Mock(return_value=iter([chunk])), close=Mock())
        fake_ollama = SimpleNamespace(Client=Mock(return_value=client))

        with patch.dict(sys.modules, {"ollama": fake_ollama}):
            value = OllamaVisionModel().query(
                Path("box.png"), "prompt", {"type": "object"}
            )

        self.assertEqual(value, {"source": "content"})

    def test_stream_reports_both_invalid_response_sources(self):
        chunk = SimpleNamespace(
            message={"content": "no object", "thinking": "still no object"}
        )
        client = SimpleNamespace(chat=Mock(return_value=iter([chunk])), close=Mock())
        fake_ollama = SimpleNamespace(Client=Mock(return_value=client))

        with (
            patch.dict(sys.modules, {"ollama": fake_ollama}),
            self.assertRaisesRegex(
                ValueError,
                "content: .*thinking:",
            ),
        ):
            OllamaVisionModel().query(Path("box.png"), "prompt", {"type": "object"})

    def test_stream_rejects_a_response_without_content_or_thinking(self):
        client = SimpleNamespace(
            chat=Mock(return_value=iter([SimpleNamespace(message={})])),
            close=Mock(),
        )
        fake_ollama = SimpleNamespace(Client=Mock(return_value=client))

        with (
            patch.dict(sys.modules, {"ollama": fake_ollama}),
            self.assertRaisesRegex(ValueError, "neither content nor thinking"),
        ):
            OllamaVisionModel().query(Path("box.png"), "prompt", {"type": "object"})


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

    def test_ollama_host_rejects_unsupported_url_shapes(self):
        cases = (
            ("ftp://127.0.0.1:11434", r"HTTP\(S\) URL"),
            ("http://127.0.0.1:11434/api", "must not contain a path"),
            ("http://127.0.0.1:11434?model=qwen", "must not contain a path"),
            ("http://127.0.0.1:11434#fragment", "must not contain a path"),
        )
        with patch.dict("os.environ", {}, clear=True):
            for value, message in cases:
                with (
                    self.subTest(value=value),
                    self.assertRaisesRegex(ValueError, message),
                ):
                    normalize_ollama_host(value)

    def test_version_parser_accepts_prefixed_versions(self):
        self.assertEqual(version_tuple("ollama version 0.12.7"), (0, 12, 7))


if __name__ == "__main__":
    unittest.main()
