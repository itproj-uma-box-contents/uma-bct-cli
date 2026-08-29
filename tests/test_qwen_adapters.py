import unittest
from pathlib import Path
from typing import Any

from src.adapters.qwen_detector import QwenBoxDetector
from src.adapters.qwen_transcriber import QwenTextTranscriber
from src.domain import BoundingBox, Box, InputImage


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
    def test_converts_normalized_qwen_coordinates_to_shared_pixel_regions(self):
        model = FakeVisionModel({"boxes": [{"bbox_2d": [100, 200, 500, 800]}]})
        detector = QwenBoxDetector(model)
        image = InputImage(Path("shelf.jpg"), width=200, height=100)

        regions = detector.detect(image)

        self.assertEqual(len(regions), 1)
        self.assertEqual(
            regions[0].bounding_box,
            BoundingBox((20, 20), (100, 80)),
        )
        self.assertEqual(model.calls[0][0], image.file_path)


class QwenTextTranscriberTest(unittest.TestCase):
    def test_maps_qwen_output_to_detector_independent_domain_values(self):
        model = FakeVisionModel(
            {
                "boxes": [
                    {
                        "position": "top-left",
                        "unit": "0007",
                        "typed": ["UMA-AC-123 Unit 0007"],
                        "handwritten": ["Minutes 1968-70"],
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

        transcription = QwenTextTranscriber(model).transcribe(box)

        self.assertEqual(transcription.box_id, box.box_id)
        self.assertEqual(transcription.label.unit_number, "0007")
        self.assertEqual(transcription.typed_lines, ("UMA-AC-123 Unit 0007",))
        self.assertEqual(transcription.handwritten_lines, ("Minutes 1968-70",))
        self.assertEqual(transcription.confidence, 88)
        self.assertIn("Approximate position: top-left", model.calls[0][1])

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


if __name__ == "__main__":
    unittest.main()
