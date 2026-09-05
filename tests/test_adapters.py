import contextlib
import io
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from helpers import make_box, make_image

from uma_bct.adapters.ollama_client import (
    DEFAULT_OLLAMA_HOST,
    OllamaVisionModel,
    normalize_ollama_host,
)
from uma_bct.adapters.qwen_detector import (
    QwenBoxDetector,
    validate_grounding_proposals,
)
from uma_bct.adapters.qwen_transcriber import (
    QwenTextTranscriber,
    crop_question,
    validate_transcription_result,
)
from uma_bct.bootstrap import build_pipeline


class FakeVision:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def query(self, image_path, prompt, schema):
        self.calls.append((image_path, prompt, schema))
        return self.response


def record(**changes):
    value = {
        "position": "middle-left",
        "record_identifiers": ["1960.0009"],
        "unit": "4",
        "typed": ["1960.0009", "Unit 4"],
        "handwritten": [],
        "auxiliary_annotations": [],
        "transcription_confidence": 95,
    }
    value.update(changes)
    return value


class QwenAdapterTest(unittest.TestCase):
    def test_detector_keeps_valid_boxes_and_reports_malformed_proposals(self):
        response = {
            "boxes": [
                {"bbox_2d": [100, 200, 500, 800]},
                {"bbox_2d": [5, 5, 4, 20]},
            ]
        }
        client = FakeVision(response)
        detector = QwenBoxDetector(client)
        with patch(
            "uma_bct.adapters.qwen_detector.sanitized_detection_image",
            return_value=contextlib.nullcontext(Path("clean.png")),
        ):
            regions = detector.detect(make_image())
        self.assertEqual(regions[0].bounding_box.as_xywh(), (20, 60, 80, 180))
        self.assertEqual(len(detector.dropped_proposal_warnings), 1)

    def test_detector_rejects_an_entirely_invalid_response(self):
        for response in (
            {"boxes": []},
            {"boxes": [{"bbox_2d": [1, 2, 3]}]},
            {"boxes": [{"bbox_2d": [1, 2, float("nan"), 4]}]},
        ):
            with (
                self.subTest(response=response),
                self.assertRaises((TypeError, ValueError)),
            ):
                validate_grounding_proposals(response)

    def test_transcriber_maps_multiple_records_and_marks_uncertainty(self):
        box = make_box()
        client = FakeVision(
            {"boxes": [record(), record(unit="5", typed=["1960.0009", "Unit 5?"])]}
        )
        results = QwenTextTranscriber(client).transcribe(box)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].unit_number, "4")
        self.assertTrue(results[1].needs_review)
        self.assertIn("x=10, y=20, width=80, height=160", client.calls[0][1])

    def test_transcription_validator_rejects_bad_shape_or_confidence(self):
        for response in (
            {"boxes": [{}]},
            {"boxes": [record(transcription_confidence=True)]},
            {"boxes": [record(transcription_confidence=101)]},
        ):
            with (
                self.subTest(response=response),
                self.assertRaises((TypeError, ValueError)),
            ):
                validate_transcription_result(response)

    def test_crop_prompt_contains_the_client_identity_fields(self):
        prompt = crop_question(make_box())
        for field in ("record_identifiers", "unit", "typed", "handwritten"):
            self.assertIn(field, prompt)


class OllamaClientTest(unittest.TestCase):
    def test_endpoint_policy_defaults_to_numeric_loopback(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(normalize_ollama_host("localhost"), DEFAULT_OLLAMA_HOST)
            self.assertEqual(
                normalize_ollama_host("http://127.0.0.1:12000"),
                "http://127.0.0.1:12000",
            )
            with self.assertRaisesRegex(ValueError, "numeric loopback"):
                normalize_ollama_host("http://model.example:11434")

    def test_remote_and_compose_endpoints_require_explicit_opt_in(self):
        with patch.dict("os.environ", {"UMA_BCT_ALLOW_REMOTE_OLLAMA": "1"}, clear=True):
            self.assertEqual(
                normalize_ollama_host("https://model.example"),
                "https://model.example:11434",
            )
            with self.assertRaisesRegex(ValueError, "HTTPS"):
                normalize_ollama_host("http://model.example")
        with patch.dict(
            "os.environ", {"UMA_BCT_ALLOW_CONTAINER_OLLAMA": "1"}, clear=True
        ):
            self.assertEqual(
                normalize_ollama_host("http://ollama:11434"),
                "http://ollama:11434",
            )

    def test_query_collects_streamed_json_with_safe_client_settings(self):
        chunks = [
            {"message": {"content": '{"boxes":'}},
            {"message": {"content": " []}"}},
        ]
        client = SimpleNamespace(chat=Mock(return_value=iter(chunks)), close=Mock())
        factory = Mock(return_value=client)
        with (
            patch.dict(sys.modules, {"ollama": SimpleNamespace(Client=factory)}),
            patch.dict("os.environ", {}, clear=True),
        ):
            result = OllamaVisionModel().query(
                Path("box.png"), "prompt", {"type": "object"}
            )
        self.assertEqual(result, {"boxes": []})
        factory.assert_called_once_with(
            host=DEFAULT_OLLAMA_HOST,
            timeout=600.0,
            follow_redirects=False,
            trust_env=False,
        )
        self.assertEqual(client.chat.call_args.kwargs["stream"], True)

    def test_setup_pull_rechecks_and_returns_model_identity(self):
        missing = SimpleNamespace(
            list=Mock(return_value={"models": []}),
            close=Mock(),
        )
        pull = SimpleNamespace(
            pull=Mock(return_value=iter([{"status": "downloading"}])),
            close=Mock(),
        )
        ready = SimpleNamespace(
            list=Mock(
                return_value={
                    "models": [{"model": "qwen-test", "digest": "sha256:test"}]
                }
            ),
            close=Mock(),
        )
        model = OllamaVisionModel(model="qwen-test")
        with patch.object(
            OllamaVisionModel, "_client", side_effect=(missing, pull, ready)
        ):
            metadata = model.ensure_model(progress_stream=io.StringIO())
        self.assertEqual(metadata["model_digest"], "sha256:test")
        pull.pull.assert_called_once_with(model="qwen-test", stream=True)

    def test_bootstrap_builds_qwen_with_opencv_fallback(self):
        client = Mock()
        client.preflight.return_value = {"model": "qwen-test"}
        with patch("uma_bct.bootstrap.OllamaVisionModel", return_value=client):
            pipeline = build_pipeline(
                detector_name="qwen",
                transcriber_name="qwen",
                model_name="qwen-test",
                work_directory=Path("crops"),
                minimum_detector_confidence=0.6,
            )
        self.assertEqual(pipeline.detector.name, "qwen")
        self.assertEqual(pipeline.fallback_detector.name, "opencv")
        client.preflight.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
