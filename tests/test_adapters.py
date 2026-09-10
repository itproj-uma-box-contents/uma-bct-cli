import contextlib
import io
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from helpers import make_box, make_image

from uma_bct.adapters.ollama_client import (
    DEFAULT_OLLAMA_HOST,
    MAX_MODEL_RESPONSE_CHARACTERS,
    OllamaVisionModel,
    json_from_text,
    normalize_ollama_host,
)
from uma_bct.adapters.qwen_detector import (
    GROUNDING_QUESTION,
    QwenBoxDetector,
    validate_grounding_proposals,
)
from uma_bct.adapters.qwen_response import normalize_boxes_envelope
from uma_bct.adapters.qwen_transcriber import (
    MAX_TEXT_CHARACTERS,
    MAX_TRANSCRIPTION_RECORDS,
    QUESTION,
    QwenTextTranscriber,
    crop_question,
    validate_transcription_result,
)
from uma_bct.bootstrap import build_pipeline


class FakeVision:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def query(self, image_path, prompt, schema, *, validator=None):
        self.calls.append((image_path, prompt, schema))
        if validator is not None:
            validator(self.response)
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
    def test_normalizer_leaves_the_canonical_envelope_unchanged(self):
        response = {"boxes": [record()]}
        self.assertIs(normalize_boxes_envelope(response), response)

    def test_transcription_accepts_complete_equivalent_response_envelopes(self):
        first = record(
            handwritten=["Café 📦; O'Brien"],
            auxiliary_annotations=["Fragile; retain [brackets]"],
        )
        second = record(
            position="middle-right",
            unit="5",
            typed=["1960.0009", "Unit 5"],
        )
        envelope = json.dumps({"boxes": [first, second]}, ensure_ascii=False)
        cases = (
            (envelope, [first, second]),
            (f"```json\n{envelope}\n```", [first, second]),
            (f"Structured result follows:\n{envelope}\nEnd result.", [first, second]),
            (f"Structured [validated] result:\n{envelope}", [first, second]),
            (
                "Structured [validated] result:\n"
                f"{json.dumps([first, second], ensure_ascii=False)}",
                [first, second],
            ),
            (json.dumps(first, ensure_ascii=False), [first]),
            (json.dumps([first, second], ensure_ascii=False), [first, second]),
            (
                f"{json.dumps(first, ensure_ascii=False)}\n"
                f"{json.dumps(second, ensure_ascii=False)}",
                [first, second],
            ),
            (
                f"{json.dumps(first, ensure_ascii=False)},\n"
                f"{json.dumps(second, ensure_ascii=False)}",
                [first, second],
            ),
        )

        for raw, expected in cases:
            with self.subTest(raw=raw[:80]):
                parsed = json_from_text(raw)
                self.assertEqual(validate_transcription_result(parsed), expected)

    def test_detector_accepts_complete_equivalent_response_envelopes(self):
        proposal = {"bbox_2d": [100, 200, 500, 800]}
        for response in (proposal, [proposal], {"boxes": [proposal]}):
            with self.subTest(response=response):
                boxes, warnings = validate_grounding_proposals(response)
                self.assertEqual(boxes, [(100.0, 200.0, 500.0, 800.0)])
                self.assertEqual(warnings, ())

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
            {"boxes": [{"bbox_2d": [1, 2, True, 4]}]},
            {"boxes": [{"bbox_2d": [-1, 2, 3, 4]}]},
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
            {"boxes": [record(typed="not an array")]},
            {"boxes": [record(position="x" * (MAX_TEXT_CHARACTERS + 1))]},
            {"boxes": [record(typed=["x" * (MAX_TEXT_CHARACTERS + 1)])]},
        ):
            with (
                self.subTest(response=response),
                self.assertRaises((TypeError, ValueError)),
            ):
                validate_transcription_result(response)

    def test_transcription_rejects_excessive_records_after_rewrapping(self):
        response = [record(unit=str(index)) for index in range(101)]
        self.assertEqual(MAX_TRANSCRIPTION_RECORDS, 100)
        with self.assertRaisesRegex(ValueError, "at most 100"):
            validate_transcription_result(response)

    def test_response_parser_rejects_incomplete_or_ambiguous_json(self):
        valid_record = json.dumps(record())
        invalid_responses = (
            "",
            '{"boxes": [',
            "42",
            f"42 {valid_record}",
            f'{valid_record} "unrelated"',
            f"{valid_record}{valid_record}",
            f"{valid_record};{valid_record}",
            f"{valid_record}\n{{",
            f"{valid_record} tru",
            f'{valid_record} "unterminated',
            f"{valid_record} -",
            f"42 output follows {valid_record}",
        )
        for raw in invalid_responses:
            with self.subTest(raw=raw[:80]), self.assertRaises(ValueError):
                json_from_text(raw)

    def test_response_parser_preserves_the_four_mib_safety_limit(self):
        with self.assertRaisesRegex(ValueError, "4 MiB"):
            json_from_text("x" * (MAX_MODEL_RESPONSE_CHARACTERS + 1))

    def test_transcription_rejects_unrelated_and_mixed_json_values(self):
        envelope = json.dumps({"boxes": [record()]})
        bare = json.dumps(record(unit="5"))
        unrelated = json.dumps({"status": "ok"})
        for raw in (
            unrelated,
            f"{bare}\n{unrelated}",
            f"{envelope}\n{bare}",
            f"{envelope}\n{json.dumps({'boxes': [record(unit='6')]})}",
        ):
            with self.subTest(raw=raw[:80]), self.assertRaises((TypeError, ValueError)):
                validate_transcription_result(json_from_text(raw))

    def test_crop_prompt_contains_the_client_identity_fields(self):
        prompt = crop_question(make_box())
        for field in ("record_identifiers", "unit", "typed", "handwritten"):
            self.assertIn(field, prompt)

    def test_qwen_prompts_require_one_boxes_envelope(self):
        for prompt in (GROUNDING_QUESTION, QUESTION, crop_question(make_box())):
            with self.subTest(prompt=prompt[:40]):
                self.assertIn("exactly one JSON object", prompt)
                self.assertIn('top-level "boxes" array', prompt)


class OllamaClientTest(unittest.TestCase):
    def query_response(self, content, *, thinking="", validator=None):
        chunks = [{"message": {"content": content, "thinking": thinking}}]
        client = SimpleNamespace(chat=Mock(return_value=iter(chunks)), close=Mock())
        factory = Mock(return_value=client)
        kwargs = {"validator": validator} if validator is not None else {}
        with (
            patch.dict(sys.modules, {"ollama": SimpleNamespace(Client=factory)}),
            patch.dict("os.environ", {}, clear=True),
        ):
            result = OllamaVisionModel().query(
                Path("box.png"), "prompt", {"type": "object"}, **kwargs
            )
        return result

    def test_endpoint_policy_defaults_to_numeric_loopback(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(normalize_ollama_host("localhost"), DEFAULT_OLLAMA_HOST)
            self.assertEqual(
                normalize_ollama_host("http://127.0.0.1:12000"),
                "http://127.0.0.1:12000",
            )
            with self.assertRaisesRegex(ValueError, "numeric loopback"):
                normalize_ollama_host("http://model.example:11434")

    def test_endpoint_policy_rejects_empty_user_information_and_port_zero(self):
        with patch.dict("os.environ", {}, clear=True):
            for endpoint in ("http://@127.0.0.1", "http://127.0.0.1:0"):
                with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                    normalize_ollama_host(endpoint)

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

    def test_query_uses_valid_thinking_when_content_fails_the_validator(self):
        expected = record(unit="8", typed=["1960.0009", "Unit 8"])
        result = self.query_response(
            json.dumps({"status": "not the requested schema"}),
            thinking=json.dumps({"boxes": [expected]}),
            validator=validate_transcription_result,
        )
        self.assertEqual(validate_transcription_result(result), [expected])

    def test_query_rejects_incomplete_noncanonical_detector_content(self):
        first = {"bbox_2d": [1, 2, 3, 4]}
        expected = {"bbox_2d": [10, 20, 30, 40]}
        thinking = json.dumps({"boxes": [expected]})
        incomplete_content = (
            json.dumps({"status": "ok"}) + "\n" + json.dumps(first),
            json.dumps([first, {"status": "ok"}]),
            json.dumps([first]) + "\n" + json.dumps(expected),
        )

        for content in incomplete_content:
            with self.subTest(content=content):
                result = self.query_response(
                    content,
                    thinking=thinking,
                    validator=validate_grounding_proposals,
                )
                boxes, warnings = validate_grounding_proposals(result)
                self.assertEqual(boxes, [(10.0, 20.0, 30.0, 40.0)])
                self.assertEqual(warnings, ())

    def test_query_uses_thinking_when_detector_content_overflows(self):
        huge_integer = "9" * 309
        content = f'{{"boxes":[{{"bbox_2d":[0,0,{huge_integer},1]}}]}}'
        expected = {"bbox_2d": [10, 20, 30, 40]}
        result = self.query_response(
            content,
            thinking=json.dumps({"boxes": [expected]}),
            validator=validate_grounding_proposals,
        )
        boxes, warnings = validate_grounding_proposals(result)
        self.assertEqual(boxes, [(10.0, 20.0, 30.0, 40.0)])
        self.assertEqual(warnings, ())

    def test_query_prefers_valid_content_over_valid_thinking(self):
        expected = record(unit="8", typed=["1960.0009", "Unit 8"])
        ignored = record(unit="9", typed=["1960.0009", "Unit 9"])
        result = self.query_response(
            json.dumps(expected),
            thinking=json.dumps({"boxes": [ignored]}),
            validator=validate_transcription_result,
        )
        self.assertEqual(validate_transcription_result(result), [expected])

    def test_query_reports_bounded_channel_errors_without_response_text(self):
        private_text = "private archive text " * 300
        with self.assertRaises(ValueError) as raised:
            self.query_response(
                json.dumps({"status": private_text}),
                thinking='{\n  "boxes": [',
                validator=validate_transcription_result,
            )
        message = str(raised.exception)
        self.assertIn("content", message)
        self.assertIn("thinking", message)
        self.assertLessEqual(len(message), 1024)
        self.assertNotIn(private_text, message)

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
