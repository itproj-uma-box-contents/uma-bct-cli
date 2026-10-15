"""Malformed model metadata must not discard otherwise usable box readings."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import test_pipeline
from helpers import make_box

from uma_bct.adapters.ollama_client import OllamaVisionModel
from uma_bct.adapters.qwen_response import box_token_confidences
from uma_bct.adapters.qwen_transcriber import QwenTextTranscriber


class ModelResponseSafetyTest(unittest.TestCase):
    def transcribe(self, raw, probability):
        chunk = {"message": {"content": raw}, "logprobs": [{"token": raw, "logprob": probability}]}
        client = SimpleNamespace(chat=Mock(return_value=iter([chunk])), close=Mock())
        with patch.object(OllamaVisionModel, "client", return_value=client):
            return QwenTextTranscriber(OllamaVisionModel(model="qwen-test")).transcribe(make_box())

    def test_invalid_log_probabilities_preserve_reading_with_unavailable_confidence(self):
        raw = json.dumps({"boxes": [test_pipeline.qwen_record(handwritten=["Correspondence"])]})
        for probability in (True, 0.5, float("nan"), float("inf"), float("-inf")):
            with self.subTest(probability=probability):
                readings = self.transcribe(raw, probability)
                self.assertEqual(readings[0].handwritten_lines, ("Correspondence",))
                self.assertEqual(readings[0].confidence, 0)
                self.assertTrue(readings[0].needs_review)
                self.assertEqual(box_token_confidences(((raw, probability),), 1), [None])

    def test_log_probabilities_must_describe_the_actual_returned_text(self):
        raw = json.dumps({"boxes": [test_pipeline.qwen_record(handwritten=["Actual text"])]})
        other = json.dumps({"boxes": [test_pipeline.qwen_record(handwritten=["Different text"])]})
        client = SimpleNamespace(
            chat=Mock(return_value=iter([{"message": {"content": raw}, "logprobs": [{"token": other, "logprob": 0.0}]}])),
            close=Mock(),
        )
        with patch.object(OllamaVisionModel, "client", return_value=client):
            readings = QwenTextTranscriber(OllamaVisionModel(model="qwen-test")).transcribe(make_box())
        self.assertEqual(readings[0].handwritten_lines, ("Actual text",))
        self.assertEqual(readings[0].confidence, 0)

    def test_invalid_unicode_fails_one_crop_without_losing_a_successful_box(self):
        fixture = test_pipeline.PipelineTest()
        fixture.setUp()
        good = json.dumps({"boxes": [test_pipeline.qwen_record(handwritten=["Valid 王 📦"])]})
        bad = json.dumps({"boxes": [test_pipeline.qwen_record(unit="5", handwritten=["invalid \ud800"])]})
        clients = [
            SimpleNamespace(chat=Mock(return_value=iter([{"message": {"content": raw}}])), close=Mock())
            for raw in (good, bad)
        ]
        transcriber = QwenTextTranscriber(OllamaVisionModel(model="qwen-test"))
        detector = test_pipeline.FakeDetector("qwen", [fixture.region, fixture.region])
        with patch.object(OllamaVisionModel, "client", side_effect=clients) as client_factory:
            result = fixture.process(fixture.pipeline(detector, transcriber))
        self.assertEqual(client_factory.call_count, 2)
        self.assertEqual(result.status, "partial")
        self.assertEqual(len(result.records), 1)
        self.assertEqual(result.records[0].transcription.handwritten_lines, ("Valid 王 📦",))
