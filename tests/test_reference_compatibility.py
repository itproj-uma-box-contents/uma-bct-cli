import hashlib
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import uma_bct
import uma_bct.cli as cli_module
from uma_bct import pipeline as pipeline_module
from uma_bct.adapters.ollama_client import (
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_HOST,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_TEMPERATURE,
    DEFAULT_THINK,
    OllamaVisionModel,
)
from uma_bct.adapters.opencv_detector import (
    DETECTION_MAX_DIMENSION,
    INTERNAL_CONFIDENCE_FLOOR,
    detect_regions,
)
from uma_bct.adapters.qwen_detector import (
    GROUNDING_QUESTION,
    GROUNDING_SCHEMA,
    validate_grounding_result,
)
from uma_bct.adapters.qwen_transcriber import (
    QUESTION,
    SCHEMA,
    TRANSCRIPTION_SCHEMA,
    crop_question,
)
from uma_bct.cli import parse_args
from uma_bct.constants import (
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
from uma_bct.domain import BoundingBox, Box, DetectedRegion, InputImage
from uma_bct.run_artifacts import manifest_path, write_json_atomic


def text_digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def schema_digest(value: dict[str, object]) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return text_digest(canonical)


class ReferencePromptCompatibilityTest(unittest.TestCase):
    def test_legacy_public_symbols_remain_compatible(self):
        arguments = parse_args(
            [
                "--path-to-output",
                "legacy.csv",
                "--work-dir",
                "legacy-crops",
            ]
        )

        self.assertEqual(arguments.path_to_outputs, Path("legacy.csv"))
        self.assertEqual(arguments.path_to_box_images, Path("legacy-crops"))
        self.assertEqual(
            validate_grounding_result({"boxes": [{"bbox_2d": [1, 2, 3, 4]}]}),
            [(1.0, 2.0, 3.0, 4.0)],
        )
        self.assertIs(TRANSCRIPTION_SCHEMA, SCHEMA)
        self.assertIs(
            uma_bct.BoxTranscriptionPipeline,
            pipeline_module.BoxTranscriptionPipeline,
        )
        self.assertIs(uma_bct.ImageResult, pipeline_module.ImageResult)
        self.assertIs(cli_module.manifest_path, manifest_path)
        self.assertIs(cli_module.write_json_atomic, write_json_atomic)

    def test_qwen_detection_prompt_and_schema_match_reference_hashes(self):
        self.assertEqual(
            text_digest(GROUNDING_QUESTION),
            "d6cac9d883c082203050a7805f5e6340096e3fa83659c173255ccd83f2e93699",
        )
        self.assertEqual(
            schema_digest(GROUNDING_SCHEMA),
            "442cb89bb94d83b0d796170eb1d5f7b6ce8a8891454f3f7a5d9560666dff8b75",
        )

    def test_qwen_transcription_contract_extends_reference_for_catalogue_ids(self):
        required = SCHEMA["properties"]["boxes"]["items"]["required"]

        self.assertIn("record_identifiers", required)
        self.assertIn("auxiliary_annotations", required)
        self.assertIn("1960.0009", QUESTION)
        self.assertIn("UMA-SRE-20160038", QUESTION)

    def test_generated_crop_prompt_includes_original_geometry_and_new_fields(self):
        image = InputImage(Path("shelf.jpg"), width=200, height=300)
        box = Box(
            "shelf:001",
            image,
            BoundingBox((10, 20), (90, 180)),
            Path("box.png"),
            "top-left",
            0.9,
        )

        prompt = crop_question(box)
        self.assertIn("x=10, y=20, width=80, height=160", prompt)
        self.assertIn("record_identifiers", prompt)
        self.assertIn("auxiliary_annotations", prompt)


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

    def test_resized_regions_use_exact_per_axis_reference_mapping(self):
        original = np.zeros((2003, 3337, 3), dtype=np.uint8)
        candidate = DetectedRegion(
            BoundingBox((101, 207), (777, 799)),
            confidence=0.9,
        )

        with patch(
            "uma_bct.adapters.opencv_detector.extract_candidates",
            return_value=[candidate],
        ):
            regions = detect_regions(original)

        self.assertEqual(
            regions[0].bounding_box,
            BoundingBox((240, 493), (1853, 1906)),
        )


class ReferenceOllamaSettingsTest(unittest.TestCase):
    def test_streamed_request_uses_reference_model_and_settings(self):
        chunks = [
            SimpleNamespace(message={"content": '{"boxes":'}),
            SimpleNamespace(message={"content": " []}"}),
        ]
        chat = Mock(return_value=iter(chunks))
        client = SimpleNamespace(chat=chat, close=Mock())
        client_factory = Mock(return_value=client)
        fake_ollama = SimpleNamespace(Client=client_factory)

        with (
            patch.dict(sys.modules, {"ollama": fake_ollama}),
            patch.dict("os.environ", {}, clear=True),
        ):
            result = OllamaVisionModel().query(
                Path("box.png"),
                "prompt",
                {"type": "object"},
            )

        self.assertEqual(result, {"boxes": []})
        client_factory.assert_called_once_with(
            host=DEFAULT_OLLAMA_HOST,
            timeout=DEFAULT_REQUEST_TIMEOUT_SECONDS,
            follow_redirects=False,
            trust_env=False,
        )
        chat.assert_called_once_with(
            model="qwen3-vl:8b-instruct-q4_K_M",
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
        client.close.assert_called_once_with()
        self.assertEqual(DEFAULT_MODEL, "qwen3-vl:8b-instruct-q4_K_M")
        self.assertEqual(DEFAULT_CONTEXT_LENGTH, 16384)
        self.assertEqual(DEFAULT_TEMPERATURE, 0)
        self.assertIs(DEFAULT_THINK, False)


if __name__ == "__main__":
    unittest.main()
