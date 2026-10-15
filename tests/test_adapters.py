import contextlib
import io
import json
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from helpers import make_box, make_image

from uma_bct.adapters import ollama_client
from uma_bct.adapters.ollama_client import (
    DEFAULT_OLLAMA_HOST,
    MAX_MODEL_RESPONSE_CHARACTERS,
    OllamaVisionModel,
    json_from_text,
    normalize_ollama_host,
)
from uma_bct.adapters.qwen_date_detector import QwenDateDetector, get_result_schema, parse_qwen_result
from uma_bct.adapters.qwen_detector import (
    GEMINI_GROUNDING_SCHEMA,
    GROUNDING_QUESTION,
    GROUNDING_SCHEMA,
    HOSTED_GROUNDING_QUESTION,
    HOSTED_GROUNDING_SCHEMA,
    QwenBoxDetector,
    validate_grounding_proposals,
)
from uma_bct.adapters.qwen_response import box_object_spans, box_token_confidences, wrap_boxes
from uma_bct.adapters.qwen_transcriber import (
    MAX_TEXT_CHARACTERS,
    MAX_TRANSCRIPTION_RECORDS,
    QUESTION,
    QwenTextTranscriber,
    crop_question,
    validate_transcription_result,
)
from uma_bct.adapters.regex_date_detector import RegexDateDetector, fill_prefix
from uma_bct.bootstrap import build_pipeline
from uma_bct.ports import ModelResponse
from uma_bct.util.date_util import Date, DateRange


class FakeVision:
    def __init__(self, response, token_logprobs=()):
        self.response = response
        self.token_logprobs = token_logprobs
        self.calls = []

    def query(self, image_path, prompt, schema, *, validator=None, logprobs=False):
        self.calls.append((image_path, prompt, schema, logprobs))
        if validator is not None:
            validator(self.response)
        return ModelResponse(value=self.response, token_logprobs=self.token_logprobs if logprobs else ())


def record(**changes):
    value = {
        "position": "middle-left",
        "record_identifiers": ["1960.0009"],
        "unit": "4",
        "typed": ["1960.0009", "Unit 4"],
        "handwritten": [],
        "auxiliary_annotations": [],
    }
    value.update(changes)
    return value


def confident_token_logprobs(response, low_confidence_characters=""):
    """Build a realistic (token, logprob) stream for a JSON response.

    One character per token keeps the offset arithmetic obvious; a real
    Ollama stream instead groups several characters per token. Every
    character is highly confident except the ones listed.
    """

    raw_text = json.dumps(response, separators=(",", ":"))
    return tuple((character, -6.0 if character in low_confidence_characters else -0.0001) for character in raw_text)


class QwenAdapterTest(unittest.TestCase):
    def test_wrap_boxes_keeps_the_normal_reply_unchanged(self):
        response = {"boxes": [record()]}
        self.assertIs(wrap_boxes(response), response)

    def test_transcription_accepts_complete_equivalent_response_envelopes(self):
        first = record(handwritten=["Café 📦; O'Brien"], auxiliary_annotations=["Fragile; retain [brackets]"])
        second = record(position="middle-right", unit="5", typed=["1960.0009", "Unit 5"])
        envelope = json.dumps({"boxes": [first, second]}, ensure_ascii=False)
        cases = (
            (envelope, [first, second]),
            (f"```json\n{envelope}\n```", [first, second]),
            (f"Structured result follows:\n{envelope}\nEnd result.", [first, second]),
            (json.dumps(first, ensure_ascii=False), [first]),
            (json.dumps([first, second], ensure_ascii=False), [first, second]),
            (f"{json.dumps(first, ensure_ascii=False)}\n{json.dumps(second, ensure_ascii=False)}", [first, second]),
            (f"{json.dumps(first, ensure_ascii=False)},\n{json.dumps(second, ensure_ascii=False)}", [first, second]),
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
        response = {"boxes": [{"bbox_2d": [100, 200, 500, 800]}, {"bbox_2d": [5, 5, 4, 20]}]}
        client = FakeVision(response)
        detector = QwenBoxDetector(client)
        with patch(
            "uma_bct.adapters.qwen_detector.sanitized_detection_image",
            return_value=contextlib.nullcontext(Path("clean.png")),
        ):
            regions = detector.detect(make_image())
        self.assertEqual(regions[0].bounding_box.as_xywh(), (20, 60, 80, 180))
        self.assertEqual(len(detector.dropped_proposal_warnings), 1)
        self.assertIs(client.calls[0][2], GROUNDING_SCHEMA)
        self.assertEqual(client.calls[0][1], GROUNDING_QUESTION)
        self.assertNotIn("at least about half of its face width", client.calls[0][1])

    def test_hosted_x_first_detector_uses_stricter_prompt_without_gemini_coordinates(self):
        client = FakeVision({"boxes": [{"bbox_2d": [100, 200, 500, 800]}]})
        detector = QwenBoxDetector(client, prompt_profile="hosted")
        with patch(
            "uma_bct.adapters.qwen_detector.sanitized_detection_image",
            return_value=contextlib.nullcontext(Path("clean.png")),
        ):
            detector.detect(make_image())
        prompt = client.calls[0][1]
        self.assertEqual(prompt, HOSTED_GROUNDING_QUESTION)
        self.assertIn("at least about half of its face width", prompt)
        self.assertIn("[x1, y1, x2, y2]", prompt)
        self.assertNotIn("[ymin, xmin, ymax, xmax]", prompt)
        self.assertIs(client.calls[0][2], HOSTED_GROUNDING_SCHEMA)

    def test_gemini_detector_converts_y_first_coordinates_on_a_non_square_image(self):
        client = FakeVision({"boxes": [{"bbox_2d": [100, 200, 500, 800]}, {"bbox_2d": [5, 5, 4, 20]}]})
        detector = QwenBoxDetector(client, coordinate_order="yxyx", prompt_profile="hosted")
        with patch(
            "uma_bct.adapters.qwen_detector.sanitized_detection_image",
            return_value=contextlib.nullcontext(Path("clean.png")),
        ):
            regions = detector.detect(make_image())
        self.assertEqual(regions[0].bounding_box.as_xywh(), (40, 30, 120, 120))
        self.assertEqual(len(detector.dropped_proposal_warnings), 1)
        self.assertIn("[ymin, xmin, ymax, xmax]", client.calls[0][1])
        self.assertNotIn("[x1, y1, x2, y2]", client.calls[0][1])
        self.assertIn("at least about half of its face width", client.calls[0][1])
        self.assertIs(client.calls[0][2], GEMINI_GROUNDING_SCHEMA)
        self.assertNotIn("description", GROUNDING_SCHEMA["properties"]["boxes"]["items"]["properties"]["bbox_2d"])

    def test_gemini_detector_keeps_side_by_side_boxes_as_vertical_crops(self):
        client = FakeVision({"boxes": [
            {"bbox_2d": [0, 0, 1000, 330]},
            {"bbox_2d": [0, 330, 1000, 660]},
            {"bbox_2d": [0, 660, 1000, 1000]},
        ]})
        detector = QwenBoxDetector(client, coordinate_order="yxyx", prompt_profile="hosted")
        with patch(
            "uma_bct.adapters.qwen_detector.sanitized_detection_image",
            return_value=contextlib.nullcontext(Path("clean.png")),
        ):
            regions = detector.detect(make_image())
        self.assertEqual([region.bounding_box.as_xywh() for region in regions], [
            (0, 0, 66, 300), (66, 0, 66, 300), (132, 0, 68, 300),
        ])

    def test_detector_rejects_an_entirely_invalid_response(self):
        for response in (
            {"boxes": []},
            {"boxes": [{"bbox_2d": [1, 2, 3]}]},
            {"boxes": [{"bbox_2d": [1, 2, float("nan"), 4]}]},
            {"boxes": [{"bbox_2d": [1, 2, True, 4]}]},
            {"boxes": [{"bbox_2d": [-1, 2, 3, 4]}]},
        ):
            with self.subTest(response=response), self.assertRaises((TypeError, ValueError)):
                validate_grounding_proposals(response)

    def test_transcriber_maps_multiple_records_and_marks_uncertainty(self):
        box = make_box()
        response = {"boxes": [record(), record(unit="5", typed=["1960.0009", "Unit 5?"])]}
        client = FakeVision(response, token_logprobs=confident_token_logprobs(response))
        results = QwenTextTranscriber(client).transcribe(box)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].unit_number, "4")
        self.assertGreater(results[0].confidence, 90)
        self.assertFalse(results[0].needs_review)
        self.assertTrue(results[1].needs_review, "a '?' hedge must still force review")
        self.assertIn("x=10, y=20, width=80, height=160", client.calls[0][1])
        self.assertTrue(client.calls[0][3], "transcription requests must ask for logprobs")

    def test_transcriber_recovers_one_unambiguous_raw_unit_for_run_details(self):
        box = make_box()
        # Cover printed labels, preserved padding, and matching handwriting.
        cases = (
            ("typed", record(unit="", typed=["1960.0009", "Unit 7"]), "7"),
            ("padded typed", record(unit="", typed=["1960.0009", "Unit 0007"]), "0007"),
            (
                "matching typed and handwritten",
                record(unit="", typed=["1960.0009", "Box No: 55"], handwritten=["Box 55"]),
                "55",
            ),
        )

        # Run each raw response through the same production transcription path.
        for source, model_record, expected in cases:
            with self.subTest(source=source):
                response = {"boxes": [model_record]}
                client = FakeVision(response, token_logprobs=confident_token_logprobs(response))

                transcription = QwenTextTranscriber(client).transcribe(box)[0]

                # Keep the visible digits but require a human to review the recovery.
                self.assertEqual(transcription.unit_number, expected)
                self.assertEqual(transcription.to_dict()["unit"], expected)
                self.assertTrue(transcription.needs_review)

    def test_transcriber_leaves_unsafe_raw_unit_evidence_blank(self):
        box = make_box()
        cases = (
            ("handwritten only", record(unit="", typed=["1960.0009"], handwritten=["7"])),
            ("repeated handwriting", record(unit="", typed=["1960.0009"], handwritten=["39", "39"])),
            ("bare typed", record(unit="", typed=["1960.0009", "7"])),
            ("ambiguous", record(unit="", typed=["1960.0009", "Unit 7 or 8"])),
            (
                "conflicting",
                record(unit="", typed=["1960.0009", "Unit 7"], handwritten=["8"]),
            ),
            ("uncertain", record(unit="", typed=["1960.0009", "Unit 7?"])),
            ("crossed out", record(unit="", typed=["1960.0009", "Unit 7 crossed out"])),
        )

        for reason, model_record in cases:
            with self.subTest(reason=reason):
                response = {"boxes": [model_record]}
                client = FakeVision(response, token_logprobs=confident_token_logprobs(response))

                transcription = QwenTextTranscriber(client).transcribe(box)[0]

                self.assertEqual(transcription.unit_number, "")
                self.assertEqual(transcription.to_dict()["unit"], "")
                self.assertTrue(transcription.needs_review)

    def test_transcriber_scores_confidence_from_the_least_confident_character(self):
        box = make_box()
        response = {"boxes": [record()]}
        client = FakeVision(response, token_logprobs=confident_token_logprobs(response, low_confidence_characters="4"))
        results = QwenTextTranscriber(client).transcribe(box)
        self.assertEqual(results[0].confidence, round(100 * math.exp(-6.0)))
        # A low score is caught later by the --min-transcription-confidence
        # export threshold, not needs_review, which stays reserved for
        # hedged/empty text and a totally unavailable score.
        self.assertFalse(results[0].needs_review)

    def test_transcriber_treats_unavailable_token_confidence_as_zero_and_flags_review(self):
        box = make_box()
        client = FakeVision({"boxes": [record()]})  # no logprobs supplied
        results = QwenTextTranscriber(client).transcribe(box)
        self.assertEqual(results[0].confidence, 0)
        self.assertTrue(results[0].needs_review)

    def test_transcription_validator_rejects_bad_shape(self):
        for response in (
            {"boxes": [{}]},
            {"boxes": [record(typed="not an array")]},
            {"boxes": [record(position="x" * (MAX_TEXT_CHARACTERS + 1))]},
            {"boxes": [record(typed=["x" * (MAX_TEXT_CHARACTERS + 1)])]},
        ):
            with self.subTest(response=response), self.assertRaises((TypeError, ValueError)):
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
            f"Structured [validated] result:\n{valid_record}",
            f"[bad]\n{valid_record}",
            f"[true,]\n{valid_record}",
            f"[[bad]]\n{valid_record}",
            f"{valid_record}\nDone [validated].",
        )
        for raw in invalid_responses:
            with self.subTest(raw=raw[:80]), self.assertRaises(ValueError):
                json_from_text(raw)

    def test_response_parser_bounds_ambiguous_container_scanning(self):
        with (
            patch.object(ollama_client, "prefix_is_safe", wraps=ollama_client.prefix_is_safe) as balance_check,
            self.assertRaises(ValueError),
        ):
            json_from_text("[" * 4096)
        self.assertLessEqual(balance_check.call_count, 1)

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
        self.assertNotIn("Physical-box checks:", prompt)

    def test_qwen_prompts_require_one_boxes_envelope(self):
        for prompt in (GROUNDING_QUESTION, QUESTION, crop_question(make_box())):
            with self.subTest(prompt=prompt[:40]):
                self.assertIn("exactly one JSON object", prompt)
                self.assertIn('top-level "boxes" array', prompt)

    def test_box_object_spans_ignores_braces_inside_string_content(self):
        raw_text = '{"boxes":[{"typed":["a { b"]},{"unit":"9"}]}'
        spans = box_object_spans(raw_text)
        self.assertEqual(len(spans), 2)
        self.assertEqual(raw_text[spans[0][0] : spans[0][1]], '{"typed":["a { b"]}')
        self.assertEqual(raw_text[spans[1][0] : spans[1][1]], '{"unit":"9"}')

    def test_box_object_spans_finds_nothing_in_a_noncanonical_reply(self):
        # A bare array or adjacent objects (no {"boxes": [...]} wrapper) is
        # tolerated by json_from_text/wrap_boxes but is not scored: there is
        # no reliable brace depth to key spans off of.
        self.assertEqual(box_object_spans('[{"unit":"1"},{"unit":"2"}]'), [])
        self.assertEqual(box_object_spans('{"unit":"1"}\n{"unit":"2"}'), [])

    def test_box_token_confidences_falls_back_to_none_on_span_mismatch(self):
        self.assertEqual(box_token_confidences((), 2), [None, None])
        mismatched = (("{", -0.1), ("}", -0.1))
        self.assertEqual(box_token_confidences(mismatched, 2), [None, None])


class OllamaClientTest(unittest.TestCase):
    def query_response(self, content, *, thinking="", validator=None, content_logprobs=None, thinking_logprobs=None):
        # Real streams never carry content and thinking in the same chunk;
        # split into up to two chunks so attached logprobs land unambiguously.
        chunks = []
        if thinking:
            chunk = {"message": {"content": "", "thinking": thinking}}
            if thinking_logprobs is not None:
                chunk["logprobs"] = thinking_logprobs
            chunks.append(chunk)
        chunk = {"message": {"content": content, "thinking": ""}}
        if content_logprobs is not None:
            chunk["logprobs"] = content_logprobs
        chunks.append(chunk)

        client = SimpleNamespace(chat=Mock(return_value=iter(chunks)), close=Mock())
        factory = Mock(return_value=client)
        kwargs = {"validator": validator} if validator is not None else {}
        with patch.dict(sys.modules, {"ollama": SimpleNamespace(Client=factory)}), patch.dict("os.environ", {}, clear=True):
            return OllamaVisionModel().query(Path("box.png"), "prompt", {"type": "object"}, logprobs=True, **kwargs)

    def test_endpoint_policy_defaults_to_numeric_loopback(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(normalize_ollama_host("localhost"), DEFAULT_OLLAMA_HOST)
            self.assertEqual(normalize_ollama_host("http://127.0.0.1:12000"), "http://127.0.0.1:12000")
            with self.assertRaisesRegex(ValueError, "numeric loopback"):
                normalize_ollama_host("http://model.example:11434")

    def test_endpoint_policy_rejects_empty_user_information_and_port_zero(self):
        with patch.dict("os.environ", {}, clear=True):
            for endpoint in ("http://@127.0.0.1", "http://127.0.0.1:0"):
                with self.subTest(endpoint=endpoint), self.assertRaises(ValueError):
                    normalize_ollama_host(endpoint)

    def test_remote_and_compose_endpoints_require_explicit_opt_in(self):
        with patch.dict("os.environ", {"UMA_BCT_ALLOW_REMOTE_OLLAMA": "1"}, clear=True):
            self.assertEqual(normalize_ollama_host("https://model.example"), "https://model.example:11434")
            with self.assertRaisesRegex(ValueError, "HTTPS"):
                normalize_ollama_host("http://model.example")
        with patch.dict("os.environ", {"UMA_BCT_ALLOW_CONTAINER_OLLAMA": "1"}, clear=True):
            self.assertEqual(normalize_ollama_host("http://ollama:11434"), "http://ollama:11434")

    def test_streamed_response_does_not_print_dots(self):
        chunks = [{"message": {"content": '{"boxes": []}'}}] + [{"message": {"content": ""}}] * 60
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            content, _thinking, _content_logprobs, _thinking_logprobs = ollama_client.collect_stream_text(chunks)
        self.assertEqual(content, '{"boxes": []}')
        self.assertEqual(stderr.getvalue(), "")

    def test_query_collects_streamed_json_with_safe_client_settings(self):
        chunks = [{"message": {"content": '{"boxes":'}}, {"message": {"content": " []}"}}]
        client = SimpleNamespace(chat=Mock(return_value=iter(chunks)), close=Mock())
        factory = Mock(return_value=client)
        with patch.dict(sys.modules, {"ollama": SimpleNamespace(Client=factory)}), patch.dict("os.environ", {}, clear=True):
            result = OllamaVisionModel().query(Path("box.png"), "prompt", {"type": "object"})
        self.assertEqual(result.value, {"boxes": []})
        self.assertEqual(result.token_logprobs, ())
        factory.assert_called_once_with(host=DEFAULT_OLLAMA_HOST, timeout=600.0, follow_redirects=False, trust_env=False)
        self.assertEqual(client.chat.call_args.kwargs["stream"], True)
        self.assertEqual(client.chat.call_args.kwargs["logprobs"], False)

    def test_query_requests_and_returns_content_logprobs(self):
        chunks = [
            {"message": {"content": '{"unit"'}, "logprobs": [{"token": '{"unit"', "logprob": -12.0}]},
            {"message": {"content": ':"4"}'}, "logprobs": [{"token": ':"4"}', "logprob": -0.01}]},
        ]
        client = SimpleNamespace(chat=Mock(return_value=iter(chunks)), close=Mock())
        factory = Mock(return_value=client)
        with patch.dict(sys.modules, {"ollama": SimpleNamespace(Client=factory)}), patch.dict("os.environ", {}, clear=True):
            result = OllamaVisionModel().query(Path("box.png"), "prompt", {"type": "object"}, logprobs=True)
        self.assertEqual(result.value, {"unit": "4"})
        self.assertEqual(result.token_logprobs, (('{"unit"', -12.0), (':"4"}', -0.01)))
        self.assertEqual(client.chat.call_args.kwargs["logprobs"], True)

    def test_query_returns_thinking_logprobs_when_thinking_wins(self):
        expected = record(unit="8", typed=["1960.0009", "Unit 8"])
        thinking_text = json.dumps({"boxes": [expected]})
        result = self.query_response(
            json.dumps({"status": "not the requested schema"}),
            thinking=thinking_text,
            validator=validate_transcription_result,
            thinking_logprobs=[{"token": thinking_text, "logprob": -3.0}],
        )
        self.assertEqual(validate_transcription_result(result.value), [expected])
        self.assertEqual(result.token_logprobs, ((thinking_text, -3.0),))

    def test_query_uses_valid_thinking_when_content_fails_the_validator(self):
        expected = record(unit="8", typed=["1960.0009", "Unit 8"])
        result = self.query_response(
            json.dumps({"status": "not the requested schema"}),
            thinking=json.dumps({"boxes": [expected]}),
            validator=validate_transcription_result,
        )
        self.assertEqual(validate_transcription_result(result.value), [expected])

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
                result = self.query_response(content, thinking=thinking, validator=validate_grounding_proposals)
                boxes, warnings = validate_grounding_proposals(result.value)
                self.assertEqual(boxes, [(10.0, 20.0, 30.0, 40.0)])
                self.assertEqual(warnings, ())

    def test_query_uses_thinking_when_detector_content_overflows(self):
        huge_integer = "9" * 309
        content = f'{{"boxes":[{{"bbox_2d":[0,0,{huge_integer},1]}}]}}'
        expected = {"bbox_2d": [10, 20, 30, 40]}
        result = self.query_response(
            content, thinking=json.dumps({"boxes": [expected]}), validator=validate_grounding_proposals
        )
        boxes, warnings = validate_grounding_proposals(result.value)
        self.assertEqual(boxes, [(10.0, 20.0, 30.0, 40.0)])
        self.assertEqual(warnings, ())

    def test_query_prefers_valid_content_over_valid_thinking(self):
        expected = record(unit="8", typed=["1960.0009", "Unit 8"])
        ignored = record(unit="9", typed=["1960.0009", "Unit 9"])
        result = self.query_response(
            json.dumps(expected), thinking=json.dumps({"boxes": [ignored]}), validator=validate_transcription_result
        )
        self.assertEqual(validate_transcription_result(result.value), [expected])

    def test_query_reports_bounded_channel_errors_without_response_text(self):
        private_text = "private archive text " * 300
        with self.assertRaises(ValueError) as raised:
            self.query_response(
                json.dumps({"status": private_text}), thinking='{\n  "boxes": [', validator=validate_transcription_result
            )
        message = str(raised.exception)
        self.assertIn("content", message)
        self.assertIn("thinking", message)
        self.assertLessEqual(len(message), 1024)
        self.assertNotIn(private_text, message)

    def test_setup_pull_rechecks_and_returns_model_identity(self):
        missing = SimpleNamespace(list=Mock(return_value={"models": []}), close=Mock())
        pull = SimpleNamespace(pull=Mock(return_value=iter([{"status": "downloading"}])), close=Mock())
        ready = SimpleNamespace(
            list=Mock(return_value={"models": [{"model": "qwen-test", "digest": "sha256:test"}]}), close=Mock()
        )
        probe = SimpleNamespace(chat=Mock(return_value={"message": {"content": "ignored"}}), close=Mock())
        model = OllamaVisionModel(model="qwen-test")
        with patch.object(OllamaVisionModel, "client", side_effect=(missing, pull, ready, probe)):
            metadata = model.ensure_model(progress_stream=io.StringIO())
        self.assertEqual(metadata["model_digest"], "sha256:test")
        pull.pull.assert_called_once_with(model="qwen-test", stream=True)
        probe.chat.assert_called_once_with(
            model="qwen-test",
            messages=[{"role": "user", "content": "Reply briefly."}],
            think=False,
            options={"temperature": 0, "num_ctx": 16384, "num_predict": 1},
            stream=False,
        )

    def test_preflight_proves_the_model_can_load_without_exposing_its_reply(self):
        inventory = SimpleNamespace(
            list=Mock(return_value={"models": [{"model": "qwen-test", "digest": "sha256:test"}]}), close=Mock()
        )
        probe = SimpleNamespace(chat=Mock(return_value={"message": {"content": "private reply"}}), close=Mock())
        stdout = io.StringIO()
        model = OllamaVisionModel(model="qwen-test")

        with patch.object(OllamaVisionModel, "client", side_effect=(inventory, probe)), contextlib.redirect_stdout(stdout):
            metadata = model.preflight()

        self.assertEqual(metadata["ollama_host"], DEFAULT_OLLAMA_HOST)
        self.assertEqual(stdout.getvalue(), "")
        probe.chat.assert_called_once_with(
            model="qwen-test",
            messages=[{"role": "user", "content": "Reply briefly."}],
            think=False,
            options={"temperature": 0, "num_ctx": 16384, "num_predict": 1},
            stream=False,
        )

    def test_inference_probe_reports_model_load_failures_once(self):
        client = SimpleNamespace(chat=Mock(side_effect=RuntimeError("not enough memory")), close=Mock())
        with patch.object(OllamaVisionModel, "client", return_value=client):
            with self.assertRaisesRegex(RuntimeError, "inference preflight.*not enough memory"):
                OllamaVisionModel(model="qwen-test").probe()

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
        self.assertEqual(pipeline.date_detector.name, "regex")
        client.preflight.assert_called_once_with()


class FillPrefixTest(unittest.TestCase):
    """fill_prefix reconstructs a truncated end-year (e.g. 1942-9 -> 1949)."""

    def test_deduces_the_end_value_from_the_start_values_prefix(self):
        self.assertEqual(fill_prefix("1942", "9"), "1949")
        self.assertEqual(fill_prefix("1942", "49"), "1949")

    def test_falls_back_to_the_end_value_when_the_suffix_does_not_increase(self):
        # 41 is not a "greater suffix" than 45, so it is treated as a
        # complete (smaller) year rather than an abbreviation of 1945.
        self.assertEqual(fill_prefix("1945", "41"), "41")
        self.assertEqual(fill_prefix("1945", "1"), "1")

    def test_returns_the_other_value_when_one_side_is_empty(self):
        self.assertEqual(fill_prefix("", "1990"), "1990")
        self.assertEqual(fill_prefix("1990", ""), "1990")


class RegexDateDetectorTest(unittest.TestCase):
    def setUp(self):
        self.detector = RegexDateDetector()

    def test_extracts_numeric_dates_in_supported_formats(self):
        cases = (
            ("12/05/1990", Date(day=12, month=5, year=1990)),
            ("5/1990", Date(month=5, year=1990)),
            ("5/1890", Date(month=5, year=1890)),
            ("12/5", Date(day=12, month=5)),
            ("2001", Date(year=2001))
        )
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(self.detector.extract_dates(text), (expected,))

    def test_extracts_month_name_dates_case_insensitively(self):
        cases = (
            ("12 May 1990", Date(day=12, month=5, year=1990)),
            ("12th May 1990", Date(day=12, month=5, year=1990)),
            ("12 MAY 1990", Date(day=12, month=5, year=1990)),
            ("may 1990", Date(month=5, year=1990)),
            ("May 1990", Date(month=5, year=1990)),
        )
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(self.detector.extract_dates(text), (expected,))

    def test_fill_short_year(self):
        cases = (
            ("12 May 25", Date(day=12, month=5, year=2025)),
            ("12th May 2025", Date(day=12, month=5, year=2025)),
            ("may 21", None),
            ("21", None),
        )
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(self.detector.extract_dates(text), (expected,) if expected else ())

    def test_ordinal_formats_recognise_month_and_year_but_not_the_literal_day(self):
        # "1st"/"2nd" are baked into these two formats as literal text
        # rather than a captured day placeholder, so today's implementation
        # records month/year but leaves day empty for exactly these forms
        # (contrast with "12th", which does use the dd placeholder).
        for text in ("1st May 1990", "2nd May 1990"):
            with self.subTest(text=text):
                self.assertEqual(self.detector.extract_dates(text), (Date(month=5, year=1990),))

    def test_configured_year_range_is_enforced(self):
        self.assertEqual(self.detector.extract_dates("1499"), ())
        self.assertEqual(self.detector.extract_dates("2040"), ())
        self.assertEqual(self.detector.extract_dates("1500"), (Date(year=1500),))
        self.assertEqual(self.detector.extract_dates("2039"), (Date(year=2039),))

    def test_prefers_the_longest_match_and_drops_contained_submatches(self):
        # "1990" and "05/1990" are both matches contained inside the full
        # dd/mm/yyyy match, and must not be reported alongside it.
        self.assertEqual(
            self.detector.extract_dates("12/05/1990"),
            (Date(day=12, month=5, year=1990),),
        )

    def test_extracts_several_dates_in_left_to_right_order(self):
        text = "Received 12/05/1990, filed 01/1991"
        self.assertEqual(
            self.detector.extract_dates(text),
            (
                Date(day=12, month=5, year=1990),
                Date(month=1, year=1991),
            ),
        )

    def test_returns_empty_tuple_when_no_date_is_present(self):
        self.assertEqual(self.detector.extract_dates("undated item"), ())

    def test_custom_range_formats_build_a_date_range(self):
        cases = (
            ("12-15/05/1990", DateRange(Date(12, 5, 1990), Date(15, 5, 1990))),
            ("12-15/05", DateRange(Date(12, 5, None), Date(15, 5, None))),
            ("12-15/May/1990", DateRange(Date(12, 5, 1990), Date(15, 5, 1990))),
            ("05-06/1990", DateRange(Date(None, 5, 1990), Date(None, 6, 1990))),
            ("May-Jun/1990", DateRange(Date(None, 5, 1990), Date(None, 6, 1990))),
        )
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(self.detector.extract_dates(text), (expected,))

    def test_year_range_shorthand_fills_the_missing_prefix_from_the_start_year(self):
        self.assertEqual(
            self.detector.extract_dates("1942-9"),
            (DateRange(Date(year=1942), Date(year=1949)),),
        )
        self.assertEqual(
            self.detector.extract_dates("1942-59"),
            (DateRange(Date(year=1942), Date(year=1959)),),
        )

    def test_generic_ranges_combine_any_two_configured_date_formats(self):
        for delimiter in ("-", " to "):
            text = f"12/05/1990{delimiter}15/06/1991"
            with self.subTest(text=text):
                self.assertEqual(
                    self.detector.extract_dates(text),
                    (DateRange(Date(12, 5, 1990), Date(15, 6, 1991)),),
                )


# Build a single-date result entry with useful defaults.
def date_entry(**changes):
    value = {
        "raw_text": "1950",
        "starting_index": 0,
        "start_date": {"year": 1950},
        "end_date": {},
    }
    value.update(changes)
    return value


# Build a date-range result entry with useful defaults.
def range_entry(**changes):
    value = {
        "raw_text": "1930-1933",
        "starting_index": 0,
        "start_date": {"year": 1930},
        "end_date": {"year": 1933},
    }
    value.update(changes)
    return value


class QwenDateDetectorTest(unittest.TestCase):
    # Check that an entry with an empty end_date becomes a plain Date.
    def test_entry_with_empty_end_date_becomes_a_date(self):
        result = {"dates": [date_entry(start_date={"day": 10, "month": 12, "year": 1950})]}
        self.assertEqual(parse_qwen_result(result), (Date(day=10, month=12, year=1950),))

    # Check that an entry with any end_date component becomes a DateRange.
    def test_entry_with_a_populated_end_date_becomes_a_date_range(self):
        result = {"dates": [range_entry(start_date={"month": 3, "year": 1930}, end_date={"month": 5, "year": 1933})]}
        self.assertEqual(
            parse_qwen_result(result),
            (DateRange(start=Date(month=3, year=1930), end=Date(month=5, year=1933)),),
        )

    # Check that missing start/end fields are left as None rather than guessed.
    def test_missing_date_components_are_left_as_none(self):
        result = {"dates": [date_entry(start_date={"year": 1850})]}
        self.assertEqual(parse_qwen_result(result), (Date(day=None, month=None, year=1850),))

    # Check that an end_date is only treated as present if at least one of its
    # three components is non-None (an empty dict must not become DateRange()).
    def test_end_date_with_all_components_absent_is_treated_as_no_range(self):
        result = {"dates": [date_entry(end_date={})]}
        parsed = parse_qwen_result(result)
        self.assertEqual(parsed, (Date(year=1950),))
        self.assertIsInstance(parsed[0], Date)

    # Check that results are reordered into ascending starting_index order,
    # regardless of the order the model returned them in.
    def test_results_are_sorted_by_starting_index(self):
        result = {
            "dates": [
                date_entry(starting_index=62, start_date={"year": 2021}),
                date_entry(starting_index=0, start_date={"year": 1950}),
                range_entry(starting_index=39, start_date={"month": 3, "year": 1930}, end_date={"month": 5, "year": 1933}),
            ]
        }
        self.assertEqual(
            parse_qwen_result(result),
            (
                Date(year=1950),
                DateRange(start=Date(month=3, year=1930), end=Date(month=5, year=1933)),
                Date(year=2021),
            ),
        )

    # Check that an empty "dates" list produces an empty tuple.
    def test_empty_dates_list_produces_empty_tuple(self):
        self.assertEqual(parse_qwen_result({"dates": []}), ())

    # Check that a mix of individual dates and ranges from the prompt's own
    # worked example round-trips as expected.
    def test_handles_a_realistic_mixed_response(self):
        result = {
            "dates": [
                date_entry(starting_index=0, start_date={"day": 10, "month": 12, "year": 1950}),
                date_entry(starting_index=39 - 100, start_date={"month": 5, "year": 1921}),  # out of order on purpose
                range_entry(starting_index=39, start_date={"month": 3, "year": 1930}, end_date={"month": 5, "year": 1933}),
                range_entry(starting_index=62, start_date={"year": 2021}, end_date={"year": 2023}),
            ]
        }
        parsed = parse_qwen_result(result)
        self.assertEqual(len(parsed), 4)
        # Sorted by starting_index: the -61 entry comes first.
        self.assertEqual(parsed[0], Date(month=5, year=1921))
        self.assertEqual(parsed[1], Date(day=10, month=12, year=1950))
        self.assertEqual(parsed[2], DateRange(start=Date(month=3, year=1930), end=Date(month=5, year=1933)))
        self.assertEqual(parsed[3], DateRange(start=Date(year=2021), end=Date(year=2023)))

    # Check that the schema's starting_index upper bound is derived from the
    # length of the input string, so the model can't point past the text.
    def test_starting_index_max_is_derived_from_string_length(self):
        schema = get_result_schema("12345")
        date_schema = schema["properties"]["dates"]["items"]["properties"]
        self.assertEqual(date_schema["starting_index"]["max"], 4)

    # Check the schema always requires the four fields the parser depends on.
    def test_required_fields_match_what_the_parser_reads(self):
        schema = get_result_schema("some text")
        self.assertEqual(
            set(schema["properties"]["dates"]["items"]["required"]),
            {"raw_text", "starting_index", "start_date", "end_date"},
        )

    # Check that the detector defaults to the "qwen" name used elsewhere
    # (e.g. by build_pipeline) to select it.
    def test_name_defaults_to_qwen(self):
        self.assertEqual(QwenDateDetector(FakeVision({"dates": []})).name, "qwen")

    # Check that extract_dates returns an empty tuple for text with no dates.
    def test_extract_dates_returns_empty_tuple_when_no_dates_found(self):
        detector = QwenDateDetector(FakeVision({"dates": []}))
        self.assertEqual(detector.extract_dates("undated item"), ())

    # Check that extract_dates parses a single complete date from the client's response.
    def test_extract_dates_parses_a_single_date(self):
        response = {"dates": [date_entry(start_date={"day": 10, "month": 12, "year": 1950})]}
        detector = QwenDateDetector(FakeVision(response))
        self.assertEqual(detector.extract_dates("10/12/1950"), (Date(day=10, month=12, year=1950),))

    # Check that extract_dates parses a date range from the client's response.
    def test_extract_dates_parses_a_date_range(self):
        response = {"dates": [range_entry(start_date={"year": 2021}, end_date={"year": 2023})]}
        detector = QwenDateDetector(FakeVision(response))
        self.assertEqual(detector.extract_dates("2021-23"), (DateRange(start=Date(year=2021), end=Date(year=2023)),))

    # Check that extract_dates preserves left-to-right order across several
    # dates and ranges even when the model lists them out of order.
    def test_extract_dates_orders_multiple_results_by_position_in_the_text(self):
        text = "10/12/1950. In 5/1921, Confrence 3.1930 to 5.1933"
        response = {
            "dates": [
                range_entry(starting_index=34, start_date={"month": 3, "year": 1930}, end_date={"month": 5, "year": 1933}),
                date_entry(starting_index=0, start_date={"day": 10, "month": 12, "year": 1950}),
                date_entry(starting_index=15, start_date={"month": 5, "year": 1921}),
            ]
        }
        detector = QwenDateDetector(FakeVision(response))
        self.assertEqual(
            detector.extract_dates(text),
            (
                Date(day=10, month=12, year=1950),
                Date(month=5, year=1921),
                DateRange(start=Date(month=3, year=1930), end=Date(month=5, year=1933)),
            ),
        )

    # Check that the detector's own name can still be overridden if needed,
    # since QwenDateDetector is a plain dataclass field rather than a constant.
    def test_name_can_be_overridden(self):
        detector = QwenDateDetector(FakeVision({"dates": []}), name="qwen-custom")
        self.assertEqual(detector.name, "qwen-custom")


if __name__ == "__main__":
    unittest.main()
