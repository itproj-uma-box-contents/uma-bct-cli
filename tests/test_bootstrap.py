import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from uma_bct.bootstrap import build_pipeline


class PipelineBootstrapTest(unittest.TestCase):
    def test_preflights_ollama_once_and_attaches_model_metadata(self):
        client = Mock()
        client.preflight.return_value = {
            "model": "qwen-test",
            "model_digest": "sha256:test",
            "ollama_version": "1.2.3",
            "ollama_host": "http://localhost:11434",
        }

        with patch("uma_bct.bootstrap.OllamaVisionModel", return_value=client) as model:
            pipeline = build_pipeline(
                detector_name="qwen",
                transcriber_name="qwen",
                model_name="qwen-test",
                work_directory=Path("crops"),
                minimum_detector_confidence=0.6,
                request_timeout_seconds=42.0,
            )

        model.assert_called_once_with(model="qwen-test", timeout_seconds=42.0)
        client.preflight.assert_called_once_with()
        self.assertEqual(pipeline.model_metadata["model_digest"], "sha256:test")
        self.assertEqual(pipeline.fallback_detector.name, "opencv")


if __name__ == "__main__":
    unittest.main()
