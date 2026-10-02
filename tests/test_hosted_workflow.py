"""Hosted selection shares the local pipeline without crossing run identities."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image

from uma_bct.bootstrap import build_pipeline
from uma_bct.cli import parse_args
from uma_bct.managed_batch import (
    DETAILS_FILE_NAME,
    ManagedBatchError,
    initial_progress,
    input_snapshot,
    select_batch,
    snapshot_signature,
    validate_model_resume,
)
from uma_bct.model_config import HostedModelConfig
from uma_bct.setup_command import run_setup


class HostedWorkflowTest(unittest.TestCase):
    def setUp(self):
        self.config = HostedModelConfig(model="anthropic/claude-test", api_key="secret-test-key")
        self.metadata = {
            "model": self.config.model,
            "backend": "hosted",
            "provider": "anthropic",
            "api_base": "https://api.anthropic.com",
        }

    def test_one_pipeline_client_serves_detector_and_transcriber(self):
        model = Mock()
        model.preflight.return_value = self.metadata
        with patch("uma_bct.bootstrap.HostedVisionModel", return_value=model) as hosted, patch(
            "uma_bct.bootstrap.OllamaVisionModel"
        ) as ollama:
            pipeline = build_pipeline(
                detector_name="qwen", transcriber_name="qwen", model_name="qwen-local",
                work_directory=Path("unused"), minimum_detector_confidence=0.6,
                hosted_model=self.config,
            )
        hosted.assert_called_once()
        ollama.assert_not_called()
        self.assertIs(pipeline.detector.client, model)
        self.assertIs(pipeline.transcriber.client, model)
        self.assertEqual(pipeline.model_metadata, self.metadata)

    def test_synthetic_photo_runs_detection_and_transcription_through_hosted_client(self):
        def reply(value):
            return SimpleNamespace(choices=[SimpleNamespace(
                finish_reason="stop", message=SimpleNamespace(content=json.dumps(value)), logprobs=None,
            )])

        sdk = SimpleNamespace(
            get_supported_openai_params=Mock(return_value=("response_format",)),
            supports_response_schema=Mock(return_value=False),
            completion=Mock(side_effect=(
                reply({"boxes": [{"bbox_2d": [0, 0, 1000, 1000]}]}),
                reply({"boxes": [{
                    "position": "centre", "record_identifiers": ["1960.0009"], "unit": "4",
                    "typed": ["1960.0009", "Unit 4"], "handwritten": [], "auxiliary_annotations": [],
                }]}),
            )),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            photo = root / "synthetic.png"
            Image.new("RGB", (200, 200), "white").save(photo)
            with patch("uma_bct.adapters.hosted_client._load_litellm", return_value=sdk), contextlib.redirect_stderr(
                io.StringIO()
            ):
                pipeline = build_pipeline(
                    detector_name="qwen", transcriber_name="qwen", model_name="qwen-local",
                    work_directory=root / "crops", minimum_detector_confidence=0.6,
                    hosted_model=self.config,
                )
                result = pipeline.process(photo)
        self.assertEqual(sdk.completion.call_count, 2)
        self.assertEqual(result.model_metadata, self.metadata)
        self.assertEqual(len(result.records), 1)
        self.assertEqual(result.records[0].transcription.confidence, 0)
        self.assertTrue(result.records[0].transcription.needs_review)
        for call in sdk.completion.call_args_list:
            parts = call.kwargs["messages"][0]["content"]
            self.assertTrue(parts[1]["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_hosted_setup_checks_dependency_without_pulling_ollama(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "uma_bct.setup_command.HostedVisionModel"
        ) as hosted, patch("uma_bct.setup_command.OllamaVisionModel") as ollama, contextlib.redirect_stdout(
            io.StringIO()
        ) as stdout, contextlib.redirect_stderr(io.StringIO()):
            status = run_setup(root=Path(directory), hosted_model=self.config)
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(stdout.getvalue())["model"], self.config.model)
        hosted.return_value.preflight.assert_called_once_with()
        ollama.assert_not_called()

    def test_hosted_resume_requires_backend_model_and_origin(self):
        args = parse_args([])
        args.hosted_model = self.config
        progress = initial_progress(args, Path("batch"), [], 10.0)
        self.assertEqual(progress["configuration"]["backend"], "hosted")
        self.assertEqual(progress["configuration"]["model"], self.config.model)
        self.assertNotIn(self.config.api_key, json.dumps(progress))
        validate_model_resume(progress, self.metadata, Path("batch"))
        for update, message in (
            ({"backend": "ollama"}, "backend changed"),
            ({"model": "anthropic/other"}, "model identity changed"),
            ({"api_base": "https://other.example"}, "endpoint changed"),
        ):
            with self.subTest(update=update), self.assertRaisesRegex(ManagedBatchError, message):
                validate_model_resume(progress, {**self.metadata, **update}, Path("batch"))

    def test_completed_local_batch_is_not_reused_for_hosted_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / "inputs"
            results = root / "Results"
            inputs.mkdir()
            results.mkdir()
            image = inputs / "shelf.jpg"
            image.write_bytes(b"test bytes")
            snapshot = input_snapshot([image], inputs)
            old_batch = results / "Batch 2026-01-01 00-00-00"
            old_batch.mkdir()
            (old_batch / DETAILS_FILE_NAME).write_text(json.dumps({
                "created_at": "2026-01-01T00:00:00+00:00",
                "input_snapshot": snapshot_signature(snapshot),
                "configuration": {"model": "qwen-local"},
            }), encoding="utf-8")
            args = parse_args([])
            args.path_to_inputs = inputs
            args.model = "qwen-local"
            args.hosted_model = self.config
            args.reprocess = False
            _new_batch, progress, resumed, completed = select_batch(args, results, snapshot, 10.0)
        self.assertIsNone(completed)
        self.assertFalse(resumed)
        self.assertEqual(progress["model"], self.metadata)


if __name__ == "__main__":
    unittest.main()
