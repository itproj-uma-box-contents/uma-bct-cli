import tempfile
import unittest
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from uma_bct.adapters.qwen_detector import (
    GROUNDING_SCHEMA,
    MAX_DETECTOR_PROPOSALS,
    QwenBoxDetector,
    validate_grounding_proposals,
)
from uma_bct.adapters.qwen_transcriber import (
    MAX_TEXT_CHARACTERS,
    MAX_TEXT_LINES_PER_FIELD,
    MAX_TRANSCRIPTION_RECORDS,
    QUESTION,
    SCHEMA,
    QwenTextTranscriber,
    crop_question,
    validate_transcription_result,
)
from uma_bct.domain import BoundingBox, Box, InputImage


class FakeVisionModel:
    def __init__(self, result: Any) -> None:
        self.result = result
        self.calls: list[tuple[Path, str, dict[str, Any]]] = []

    def query(
        self,
        image_path: Path,
        prompt: str,
        schema: dict[str, Any],
    ) -> Any:
        self.calls.append((image_path, prompt, schema))
        return self.result


class QwenBoxDetectorTest(unittest.TestCase):
    def test_schema_and_validator_bound_detector_proposals(self):
        self.assertEqual(
            GROUNDING_SCHEMA["properties"]["boxes"]["maxItems"],
            MAX_DETECTOR_PROPOSALS,
        )
        oversized = {
            "boxes": [
                {"bbox_2d": [0, 0, 1, 1]} for _ in range(MAX_DETECTOR_PROPOSALS + 1)
            ]
        }
        with self.assertRaisesRegex(ValueError, "at most 100"):
            validate_grounding_proposals(oversized)

    def test_converts_normalized_qwen_coordinates_to_shared_pixel_regions(self):
        model = FakeVisionModel({"boxes": [{"bbox_2d": [100, 200, 500, 800]}]})
        detector = QwenBoxDetector(model)
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        image_path = Path(temporary_directory.name) / "shelf.jpg"
        self.assertTrue(
            cv2.imwrite(str(image_path), np.zeros((100, 200, 3), dtype=np.uint8))
        )
        image = InputImage(image_path, width=200, height=100)

        regions = detector.detect(image)

        self.assertEqual(len(regions), 1)
        self.assertEqual(
            regions[0].bounding_box,
            BoundingBox((20, 20), (100, 80)),
        )
        sanitized_path = model.calls[0][0]
        self.assertNotEqual(sanitized_path, image.file_path)
        self.assertEqual(sanitized_path.suffix, ".png")
        self.assertFalse(sanitized_path.exists())

    def test_rejects_an_empty_grounding_result_like_the_reference(self):
        detector = QwenBoxDetector(FakeVisionModel({"boxes": []}))
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        image_path = Path(temporary_directory.name) / "shelf.jpg"
        self.assertTrue(
            cv2.imwrite(str(image_path), np.zeros((100, 200, 3), dtype=np.uint8))
        )
        image = InputImage(image_path, width=200, height=100)

        with self.assertRaisesRegex(ValueError, "contained no boxes"):
            detector.detect(image)

    def test_retains_valid_qwen_proposals_and_warns_about_malformed_ones(self):
        model = FakeVisionModel(
            {
                "boxes": [
                    {"bbox_2d": [100, 200, 50, 800]},
                    {"bbox_2d": [100, 200, 500, 800]},
                ]
            }
        )
        detector = QwenBoxDetector(model)
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        image_path = Path(temporary_directory.name) / "shelf.jpg"
        self.assertTrue(
            cv2.imwrite(str(image_path), np.zeros((100, 200, 3), dtype=np.uint8))
        )
        image = InputImage(image_path, width=200, height=100)

        regions = detector.detect(image)

        self.assertEqual(len(regions), 1)
        self.assertEqual(len(detector.dropped_proposal_warnings), 1)
        self.assertIn("invalid corners", detector.dropped_proposal_warnings[0])


class QwenTextTranscriberTest(unittest.TestCase):
    @staticmethod
    def _record(**updates: Any) -> dict[str, Any]:
        record: dict[str, Any] = {
            "position": "top-left",
            "record_identifiers": ["1974.0056"],
            "unit": "7",
            "typed": ["1974.0056", "Unit 7"],
            "handwritten": [],
            "auxiliary_annotations": [],
            "transcription_confidence": 95,
        }
        record.update(updates)
        return record

    def test_maps_qwen_output_to_detector_independent_domain_values(self):
        model = FakeVisionModel(
            {
                "boxes": [
                    {
                        "position": "top-left",
                        "record_identifiers": ["UMA-AC-000000330"],
                        "unit": "0007",
                        "typed": ["UMA-AC-000000330 Unit 0007"],
                        "handwritten": ["Minutes 1968-70"],
                        "auxiliary_annotations": ["Fragile"],
                        "transcription_confidence": 88,
                    }
                ]
            }
        )
        image = InputImage(Path("shelf.jpg"), width=200, height=100)
        box = Box(
            box_id="shelf:001",
            source_image=image,
            bounding_box=BoundingBox((20, 20), (100, 80)),
            image_file_path=Path("box_001.png"),
            position="top-left",
            detector_confidence=0.9,
        )

        transcriptions = QwenTextTranscriber(model).transcribe(box)
        transcription = transcriptions[0]

        self.assertEqual(transcription.box_id, box.box_id)
        self.assertEqual(transcription.label.unit_number, "0007")
        self.assertEqual(
            transcription.label.record_identifiers,
            ("UMA-AC-000000330",),
        )
        self.assertEqual(
            transcription.typed_lines,
            ("UMA-AC-000000330 Unit 0007",),
        )
        self.assertEqual(transcription.handwritten_lines, ("Minutes 1968-70",))
        self.assertEqual(transcription.auxiliary_annotations, ("Fragile",))
        self.assertEqual(transcription.confidence, 88)
        self.assertFalse(transcription.needs_review)
        self.assertIn(
            "The crop centre is approximately top-left",
            model.calls[0][1],
        )

    def test_preserves_every_box_record_returned_for_a_crop(self):
        records = [
            {
                "position": "top-left",
                "record_identifiers": ["1974.0056"],
                "unit": unit,
                "typed": [f"Unit {unit}"],
                "handwritten": [],
                "auxiliary_annotations": [],
                "transcription_confidence": 90,
            }
            for unit in ("0001", "0002")
        ]
        model = FakeVisionModel({"boxes": records})
        image = InputImage(Path("shelf.jpg"), width=200, height=100)
        box = Box(
            "shelf:001",
            image,
            BoundingBox((20, 20), (100, 80)),
            Path("box_001.png"),
            "top-left",
            0.9,
        )

        transcriptions = QwenTextTranscriber(model).transcribe(box)

        self.assertEqual(
            [item.label.unit_number for item in transcriptions],
            ["0001", "0002"],
        )

    def test_flags_identity_fields_missing_from_raw_typed_text(self):
        record = {
            "position": "top-left",
            "record_identifiers": ["1990.0003"],
            "unit": "30",
            "typed": ["Melbourne College Advanced Education"],
            "handwritten": [],
            "auxiliary_annotations": [],
            "transcription_confidence": 95,
        }
        model = FakeVisionModel({"boxes": [record]})
        image = InputImage(Path("shelf.jpg"), width=200, height=100)
        box = Box(
            "shelf:001",
            image,
            BoundingBox((20, 20), (100, 80)),
            Path("box_001.png"),
            "top-left",
            0.9,
        )

        transcription = QwenTextTranscriber(model).transcribe(box)[0]

        self.assertTrue(transcription.needs_review)
        self.assertIn("neighbouring boxes", model.calls[0][1])

    def test_flags_explicit_uncertainty_and_model_meta_comments(self):
        markers = (
            "",
            "?",
            "[illegible]",
            "(handwritten annotation)",
            "(手写签名)",
            "possibly Jones",
            "John ? Smith",
            "[illegible: surname]",
            "signature: Smith",
            "??? maybe",
            "Staff 1970e",
        )
        for marker in markers:
            with self.subTest(marker=marker):
                record = {
                    "position": "top-left",
                    "record_identifiers": ["1974.0056"],
                    "unit": "7",
                    "typed": ["1974.0056", "Unit 7"],
                    "handwritten": [marker],
                    "auxiliary_annotations": [],
                    "transcription_confidence": 95,
                }
                model = FakeVisionModel({"boxes": [record]})

                transcription = QwenTextTranscriber(model).transcribe(
                    self._box_for_prompt()
                )[0]

                self.assertTrue(transcription.needs_review)

    def test_arbitrary_prose_does_not_corroborate_a_structured_unit(self):
        model = FakeVisionModel(
            {
                "boxes": [
                    self._record(
                        typed=["1974.0056"], handwritten=["Staff 30 years"], unit="30"
                    )
                ]
            }
        )

        transcription = QwenTextTranscriber(model).transcribe(self._box_for_prompt())[0]

        self.assertTrue(transcription.needs_review)

    def test_bare_typed_unit_requires_identifier_line_adjacency(self):
        accepted = FakeVisionModel(
            {"boxes": [self._record(typed=["1974.0056", "30"], unit="30")]}
        )
        rejected = FakeVisionModel(
            {
                "boxes": [
                    self._record(typed=["1974.0056", "Staff records", "30"], unit="30")
                ]
            }
        )

        accepted_result = QwenTextTranscriber(accepted).transcribe(
            self._box_for_prompt()
        )[0]
        rejected_result = QwenTextTranscriber(rejected).transcribe(
            self._box_for_prompt()
        )[0]

        self.assertFalse(accepted_result.needs_review)
        self.assertTrue(rejected_result.needs_review)

    def test_schema_and_validator_bound_transcription_output(self):
        boxes_schema = SCHEMA["properties"]["boxes"]
        self.assertEqual(boxes_schema["maxItems"], MAX_TRANSCRIPTION_RECORDS)
        self.assertEqual(
            boxes_schema["items"]["properties"]["typed"]["maxItems"],
            MAX_TEXT_LINES_PER_FIELD,
        )

        with self.assertRaisesRegex(ValueError, "at most 100 boxes"):
            validate_transcription_result(
                {
                    "boxes": [
                        self._record() for _ in range(MAX_TRANSCRIPTION_RECORDS + 1)
                    ]
                }
            )
        with self.assertRaisesRegex(ValueError, "at most 100 lines"):
            validate_transcription_result(
                {
                    "boxes": [
                        self._record(typed=["line"] * (MAX_TEXT_LINES_PER_FIELD + 1))
                    ]
                }
            )
        with self.assertRaisesRegex(ValueError, "at most 4096 characters"):
            validate_transcription_result(
                {"boxes": [self._record(position="x" * (MAX_TEXT_CHARACTERS + 1))]}
            )

    def test_reference_schema_accepts_multiple_box_records(self):
        record = {
            "position": "top-left",
            "record_identifiers": ["1974.0056"],
            "unit": "0007",
            "typed": ["Unit 0007"],
            "handwritten": [],
            "auxiliary_annotations": [],
            "transcription_confidence": 88,
        }

        result = validate_transcription_result({"boxes": [record, record]})

        self.assertEqual(len(result), 2)

    def test_prompt_accepts_handwritten_unpadded_units(self):
        crop_prompt = crop_question(self._box_for_prompt())
        self.assertIn("handwritten on", QUESTION)
        self.assertIn("crossed out, or ambiguous", QUESTION)
        self.assertIn("handwritten on", crop_prompt)
        self.assertIn("crossed out, or ambiguous", crop_prompt)
        self.assertIn("complete printed Unit line", QUESTION)
        self.assertIn("complete printed Unit line", crop_prompt)
        self.assertIn("handwritten box/unit number", crop_prompt)
        self.assertIn("Never translate", QUESTION)
        self.assertIn("score below 80", crop_prompt)

    def test_accepts_old_cached_records_without_new_optional_lists(self):
        record = {
            "position": "top-left",
            "unit": "7",
            "typed": ["1974.0056", "Unit 7"],
            "handwritten": [],
            "transcription_confidence": 70,
        }

        result = validate_transcription_result({"boxes": [record]})

        self.assertEqual(result[0]["record_identifiers"], [])
        self.assertEqual(result[0]["auxiliary_annotations"], [])

    def test_rejects_malformed_model_output(self):
        model = FakeVisionModel({"boxes": [{"unit": "0007"}]})
        image = InputImage(Path("shelf.jpg"), width=200, height=100)
        box = Box(
            "shelf:001",
            image,
            BoundingBox((20, 20), (100, 80)),
            Path("box_001.png"),
            "top-left",
            0.9,
        )

        with self.assertRaises(TypeError):
            QwenTextTranscriber(model).transcribe(box)

    def test_rejects_missing_required_typed_lines_with_a_clear_error(self):
        record = {
            "position": "top-left",
            "record_identifiers": [],
            "unit": "0007",
            "handwritten": [],
            "auxiliary_annotations": [],
            "transcription_confidence": 88,
        }

        with self.assertRaisesRegex(ValueError, r"box 1\.typed"):
            validate_transcription_result({"boxes": [record]})

    @staticmethod
    def _box_for_prompt() -> Box:
        image = InputImage(Path("shelf.jpg"), width=200, height=100)
        return Box(
            "shelf:001",
            image,
            BoundingBox((20, 20), (100, 80)),
            Path("box_001.png"),
            "top-left",
            0.9,
        )


if __name__ == "__main__":
    unittest.main()
