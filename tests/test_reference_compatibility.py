import hashlib
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.adapters.ollama_client import (
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_THINK,
    OllamaVisionModel,
)
from src.adapters.opencv_detector import (
    DETECTION_MAX_DIMENSION,
    INTERNAL_CONFIDENCE_FLOOR,
)
from src.adapters.qwen_detector import GROUNDING_QUESTION, GROUNDING_SCHEMA
from src.adapters.qwen_transcriber import QUESTION, SCHEMA, crop_question
from src.domain import BoundingBox, Box, InputImage
from src.imaging import (
    CONTAINMENT_THRESHOLD,
    MAX_AREA_FRACTION,
    MAX_ASPECT_RATIO,
    MIN_AREA_FRACTION,
    MIN_ASPECT_RATIO,
    MIN_HEIGHT_FRACTION,
    MIN_WIDTH_FRACTION,
    NMS_IOU_THRESHOLD,
    PADDING_FRACTION,
)


def text_digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def schema_digest(value: dict[str, object]) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return text_digest(canonical)


class ReferencePromptCompatibilityTest(unittest.TestCase):
    def test_qwen_detection_prompt_and_schema_match_reference_hashes(self):
        self.assertEqual(
            text_digest(GROUNDING_QUESTION),
            "d6cac9d883c082203050a7805f5e6340096e3fa83659c173255ccd83f2e93699",
        )
        self.assertEqual(
            schema_digest(GROUNDING_SCHEMA),
            "ae20cc51b3684880aa51ad61fdf81f02236bfdbb03c29520c5d2f0761fd60e4f",
        )

    def test_qwen_transcription_prompt_and_schema_match_reference_hashes(self):
        self.assertEqual(
            text_digest(QUESTION),
            "fcd7bea4ce16bf8a08c6605b6423715aabee805b9d739e859fb7cff093a0b612",
        )
        self.assertEqual(
            schema_digest(SCHEMA),
            "33209214f8724b0ecf6294460bc7a4519a1193e88ab5eb1827bdbeb24b94da35",
        )

    def test_generated_crop_prompt_matches_reference_hash(self):
        image = InputImage(Path("shelf.jpg"), width=200, height=300)
        box = Box(
            "shelf:001",
            image,
            BoundingBox((10, 20), (90, 180)),
            Path("box.png"),
            "top-left",
            0.9,
        )

        self.assertEqual(
            text_digest(crop_question(box)),
            "b242623ebf57447096cbb0df54fd2b1c0d2737b71b469c61b31dffa1d9614436",
        )


class ReferenceOpenCVCompatibilityTest(unittest.TestCase):
    def test_detection_and_crop_constants_match_reference_values(self):
        self.assertEqual(DETECTION_MAX_DIMENSION, 1400)
        self.assertEqual(INTERNAL_CONFIDENCE_FLOOR, 0.45)
        self.assertEqual(MIN_AREA_FRACTION, 0.025)
        self.assertEqual(MAX_AREA_FRACTION, 0.45)
        self.assertEqual(MIN_WIDTH_FRACTION, 0.08)
        self.assertEqual(MIN_HEIGHT_FRACTION, 0.18)
        self.assertEqual(MIN_ASPECT_RATIO, 0.30)
        self.assertEqual(MAX_ASPECT_RATIO, 1.35)
        self.assertEqual(CONTAINMENT_THRESHOLD, 0.90)
        self.assertEqual(NMS_IOU_THRESHOLD, 0.45)
        self.assertEqual(PADDING_FRACTION, 0.05)


class ReferenceOllamaSettingsTest(unittest.TestCase):
    def test_streamed_request_uses_reference_model_and_settings(self):
        chunks = [
            SimpleNamespace(message={"content": '{"boxes":'}),
            SimpleNamespace(message={"content": " []}"}),
        ]
        chat = Mock(return_value=iter(chunks))
        fake_ollama = SimpleNamespace(chat=chat)

        with patch.dict(sys.modules, {"ollama": fake_ollama}):
            result = OllamaVisionModel().query(
                Path("box.png"),
                "prompt",
                {"type": "object"},
            )

        self.assertEqual(result, {"boxes": []})
        chat.assert_called_once_with(
            model="qwen3-vl",
            messages=[
                {
                    "role": "user",
                    "content": "prompt",
                    "images": ["box.png"],
                }
            ],
            format={"type": "object"},
            think=False,
            options={"temperature": 0, "num_ctx": 16384},
            stream=True,
        )
        self.assertEqual(DEFAULT_MODEL, "qwen3-vl")
        self.assertEqual(DEFAULT_CONTEXT_LENGTH, 16384)
        self.assertEqual(DEFAULT_TEMPERATURE, 0)
        self.assertIs(DEFAULT_THINK, False)


if __name__ == "__main__":
    unittest.main()
