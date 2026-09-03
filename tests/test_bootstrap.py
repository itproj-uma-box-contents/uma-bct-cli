import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from uma_bct.bootstrap import build_pipeline


class PipelineBootstrapTest(unittest.TestCase):
    def test_builds_the_supported_opencv_detector_without_a_fallback(self):
        client = Mock()
        client.preflight.return_value = {"model": "qwen-test"}

        with patch("uma_bct.bootstrap.OllamaVisionModel", return_value=client):
            pipeline = build_pipeline(
                detector_name="opencv",
                transcriber_name="qwen",
                model_name="qwen-test",
                work_directory=Path("crops"),
                minimum_detector_confidence=0.6,
            )

        self.assertEqual(pipeline.detector.name, "opencv")
        self.assertIsNone(pipeline.fallback_detector)
        self.assertEqual(pipeline.transcriber.name, "qwen")

    def test_rejects_unknown_adapters_before_constructing_the_model(self):
        cases = (
            ({"detector_name": "unknown", "transcriber_name": "qwen"}, "detector"),
            ({"detector_name": "qwen", "transcriber_name": "unknown"}, "transcriber"),
        )
        for adapter_names, message in cases:
            with (
                self.subTest(adapter_names=adapter_names),
                patch("uma_bct.bootstrap.OllamaVisionModel") as model,
                self.assertRaisesRegex(ValueError, message),
            ):
                build_pipeline(
                    **adapter_names,
                    model_name="qwen-test",
                    work_directory=Path("crops"),
                    minimum_detector_confidence=0.6,
                )

            model.assert_not_called()

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
