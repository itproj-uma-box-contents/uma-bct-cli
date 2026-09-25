import contextlib
import io
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from helpers import make_box, make_image, make_transcription

from uma_bct.adapters.empty_date_detector import EmptyDateDetector
from uma_bct.adapters.ollama_client import OllamaVisionModel
from uma_bct.adapters.qwen_transcriber import QwenTextTranscriber
from uma_bct.domain import BoundingBox, Box, DetectedRegion
from uma_bct.pipeline import BoxTranscriptionPipeline


# Provide a controllable detector test double.
class FakeDetector:
    # Store the state this object needs.
    def __init__(self, name, outcome, warnings=()):
        self.name = name
        self.outcome = outcome
        self.dropped_proposal_warnings = warnings

    # Detect boxes in the current image.
    def detect(self, image):
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


# Provide a controllable cropper test double.
class FakeCropper:
    # Create the boxes.
    def create_boxes(self, image, regions, directory):
        return [
            Box(
                f"shelf:{index:03d}", image, region.bounding_box, Path(f"crop-{index}.png"), "middle-left", region.confidence
            )
            for index, region in enumerate(regions, start=1)
        ]


# Provide a controllable scripted transcriber test double.
class ScriptedTranscriber:
    name = "qwen"

    # Store the state this object needs.
    def __init__(self, outcomes, full=()):
        self.outcomes = list(outcomes)
        self.full = full
        self.full_calls = 0

    # Transcribe the current box.
    def transcribe(self, box):
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        if outcome == "record":
            return [make_transcription(box)]
        return outcome

    # Transcribe the full image when crops are unavailable.
    def transcribe_full_image(self, box):
        self.full_calls += 1
        if isinstance(self.full, Exception):
            raise self.full
        if self.full == "record":
            return [make_transcription(box)]
        return self.full


# Keep the crop only transcriber behavior together.
class CropOnlyTranscriber:
    name = "qwen"

    # Transcribe the current box.
    def transcribe(self, box):
        raise RuntimeError("unreadable")


# Handle the Qwen record step.
def qwen_record(unit="4", **changes):
    value = {
        "position": "middle-left",
        "record_identifiers": ["1960.0009"],
        "unit": unit,
        "typed": ["1960.0009", f"Unit {unit}"],
        "handwritten": [],
        "auxiliary_annotations": [],
    }
    value.update(changes)
    return value


# Group the pipeline checks.
class PipelineTest(unittest.TestCase):
    # Prepare a clean fixture for each check.
    def setUp(self):
        self.image = make_image()
        self.region = DetectedRegion(BoundingBox((10, 20), (90, 180)), 0.9)
        self.full_box = make_box(
            image=self.image, box_id="shelf:full", bbox=BoundingBox((0, 0), (200, 300)), crop_path=Path("full.png")
        )

    # Handle the pipeline step.
    def pipeline(self, detector, transcriber, fallback=None):
        return BoxTranscriptionPipeline(
            detector=detector,
            fallback_detector=fallback,
            cropper=FakeCropper(),
            transcriber=transcriber,
            date_detector=EmptyDateDetector(),
            work_directory=Path("crops"),
            model_metadata={"model": "qwen-test"},
        )

    # Process the current input.
    def process(self, pipeline):
        with (
            patch("uma_bct.pipeline.prepare_input_image", return_value=contextlib.nullcontext(self.image)),
            patch("uma_bct.pipeline.create_full_image_box", return_value=self.full_box),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            return pipeline.process(self.image.file_path)

    # Process the Qwen responses.
    def process_qwen_responses(self, *responses):
        clients = [
            SimpleNamespace(chat=Mock(return_value=iter([{"message": {"content": response, "thinking": ""}}])), close=Mock())
            for response in responses
        ]
        transcriber = QwenTextTranscriber(OllamaVisionModel(model="qwen-test"))
        with patch.object(OllamaVisionModel, "client", side_effect=clients) as client_factory:
            result = self.process(self.pipeline(FakeDetector("qwen", [self.region]), transcriber))
        return result, client_factory.call_count

    # Check that the code processes a normal Qwen crop.
    def test_processes_a_normal_qwen_crop(self):
        result = self.process(self.pipeline(FakeDetector("qwen", [self.region]), ScriptedTranscriber(["record"])))
        self.assertEqual(result.status, "complete")
        self.assertEqual(len(result.records), 1)
        self.assertEqual(result.fallback_path, ("qwen",))

    # Check that the code falls back from Qwen to OpenCV.
    def test_falls_back_from_qwen_to_opencv(self):
        result = self.process(
            self.pipeline(
                FakeDetector("qwen", RuntimeError("no boxes")),
                ScriptedTranscriber(["record"]),
                FakeDetector("opencv", [self.region]),
            )
        )
        self.assertEqual(result.actual_detector, "opencv")
        self.assertEqual(result.fallback_path, ("qwen", "opencv"))
        self.assertEqual(result.status, "complete")

    # Check that the code uses full image when both detectors are empty.
    def test_uses_full_image_when_both_detectors_are_empty(self):
        transcriber = ScriptedTranscriber([], full="record")
        result = self.process(self.pipeline(FakeDetector("qwen", []), transcriber, FakeDetector("opencv", [])))
        self.assertEqual(result.actual_detector, "full-image")
        self.assertEqual(result.fallback_path, ("qwen", "opencv", "full-image"))
        self.assertEqual(result.status, "complete")
        self.assertEqual(transcriber.full_calls, 1)

    # Check that the code retries full image when every crop fails.
    def test_retries_full_image_when_every_crop_fails(self):
        transcriber = ScriptedTranscriber([RuntimeError("bad crop")], full="record")
        result = self.process(self.pipeline(FakeDetector("qwen", [self.region]), transcriber))
        self.assertEqual(result.actual_detector, "full-image")
        self.assertEqual(result.status, "complete")
        self.assertEqual(transcriber.full_calls, 1)

    # Check that equivalent bare crop response does not trigger full image retry.
    def test_equivalent_bare_crop_response_does_not_trigger_full_image_retry(self):
        expected = qwen_record(handwritten=["Café 📦; retain verbatim"], auxiliary_annotations=["Fragile [contents]"])
        result, request_count = self.process_qwen_responses(json.dumps(expected, ensure_ascii=False))

        self.assertEqual(request_count, 1)
        self.assertEqual(result.status, "complete")
        self.assertEqual(result.actual_detector, "qwen")
        self.assertEqual(result.fallback_path, ("qwen",))
        self.assertEqual(len(result.records), 1)
        self.assertEqual(result.records[0].transcription.unit_number, "4")
        self.assertEqual(result.records[0].transcription.handwritten_lines, ("Café 📦; retain verbatim",))

    # Check that adjacent full image records remain separate and ordered.
    def test_adjacent_full_image_records_remain_separate_and_ordered(self):
        invalid_crop = json.dumps({"boxes": [qwen_record(typed="not an array")]})
        first = qwen_record("1", position="left")
        second = qwen_record("2", position="right")
        adjacent_full_image = f"{json.dumps(first)}\n{json.dumps(second)}"

        result, request_count = self.process_qwen_responses(invalid_crop, adjacent_full_image)

        self.assertEqual(request_count, 2)
        self.assertEqual(result.status, "partial")
        self.assertEqual(result.actual_detector, "full-image")
        self.assertEqual(result.fallback_path, ("qwen", "full-image"))
        self.assertEqual([item.transcription.unit_number for item in result.records], ["1", "2"])
        self.assertNotEqual(result.records[0].box.box_id, result.records[1].box.box_id)

    # Check that invalid crop response still uses full image recovery.
    def test_invalid_crop_response_still_uses_full_image_recovery(self):
        invalid_crop = json.dumps({"boxes": [qwen_record(typed="not an array")]})
        recovered = qwen_record("7", position="centre")

        result, request_count = self.process_qwen_responses(invalid_crop, json.dumps({"boxes": [recovered]}))

        self.assertEqual(request_count, 2)
        self.assertEqual(result.status, "complete")
        self.assertEqual(result.actual_detector, "full-image")
        self.assertEqual(result.records[0].transcription.unit_number, "7")

    # Check that truncated primitive tail still uses full image recovery.
    def test_truncated_primitive_tail_still_uses_full_image_recovery(self):
        invalid_crop = json.dumps(qwen_record()) + " tru"
        recovered = qwen_record("8", position="centre")

        result, request_count = self.process_qwen_responses(invalid_crop, json.dumps({"boxes": [recovered]}))

        self.assertEqual(request_count, 2)
        self.assertEqual(result.status, "complete")
        self.assertEqual(result.actual_detector, "full-image")
        self.assertEqual(result.records[0].transcription.unit_number, "8")

    # Check that failed full image recovery does not invent catalogue rows.
    def test_failed_full_image_recovery_does_not_invent_catalogue_rows(self):
        invalid_crop = json.dumps({"boxes": [qwen_record(typed=None)]})
        invalid_full_image = json.dumps({"boxes": [qwen_record(unit=None)]})

        result, request_count = self.process_qwen_responses(invalid_crop, invalid_full_image)

        self.assertEqual(request_count, 2)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.records, ())

    # Check that partial crop failure keeps successful rows.
    def test_partial_crop_failure_keeps_successful_rows(self):
        second = DetectedRegion(BoundingBox((100, 20), (180, 180)), 0.9)
        result = self.process(
            self.pipeline(
                FakeDetector("qwen", [self.region, second]), ScriptedTranscriber([RuntimeError("bad crop"), "record"])
            )
        )
        self.assertEqual(result.status, "partial")
        self.assertEqual(len(result.records), 1)

    # Check that the code preserves multiple physical boxes from one response.
    def test_preserves_multiple_physical_boxes_from_one_response(self):
        box = make_box(image=self.image)
        response = [
            make_transcription(box, unit="1", typed=("1960.0009", "Unit 1")),
            make_transcription(box, unit="2", typed=("1960.0009", "Unit 2")),
        ]
        result = self.process(self.pipeline(FakeDetector("qwen", [self.region]), ScriptedTranscriber([response])))
        self.assertEqual(len(result.records), 2)
        self.assertNotEqual(result.records[0].box.box_id, result.records[1].box.box_id)

    # Check that crop only transcriber fails without inventing a result.
    def test_crop_only_transcriber_fails_without_inventing_a_result(self):
        result = self.process(self.pipeline(FakeDetector("opencv", [self.region]), CropOnlyTranscriber()))
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.records, ())
        self.assertIn("unavailable", result.warnings[-1])

    # Check that manifest shape is compact.
    def test_manifest_shape_is_compact(self):
        result = self.process(self.pipeline(FakeDetector("qwen", [self.region]), ScriptedTranscriber(["record"])))
        value = result.to_dict()
        self.assertIn("boxes", value)
        self.assertIn("fallback_path", value)
        self.assertNotIn("attempts", value)
        self.assertNotIn("duration_seconds", str(value))
        self.assertNotIn("proposals", str(value))


if __name__ == "__main__":
    unittest.main()
