"""Qwen box localization translated into detector-neutral pixel regions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ..domain import BoundingBox, DetectedRegion, InputImage
from ..ports import VisionModelClient

GROUNDING_QUESTION = """Locate every cardboard archive box whose front face is fully or
substantially visible. Return one tight bounding box per physical box as bbox_2d
[x1, y1, x2, y2], using coordinates normalized from 0 to 1000. Exclude thin box
fragments, shelf parts, and labels or signs as separate boxes. Do not transcribe
any text. Return only the requested JSON."""

GROUNDING_SCHEMA = {
    "type": "object",
    "properties": {
        "boxes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "bbox_2d": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                    }
                },
                "required": ["bbox_2d"],
            },
        }
    },
    "required": ["boxes"],
}


def validate_grounding_result(value: Any) -> list[tuple[float, float, float, float]]:
    if not isinstance(value, dict) or not isinstance(value.get("boxes"), list):
        raise TypeError("the response must be an object containing a boxes array")

    boxes: list[tuple[float, float, float, float]] = []
    for index, item in enumerate(value["boxes"], start=1):
        bbox = item.get("bbox_2d") if isinstance(item, dict) else None
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise TypeError(f"box {index}.bbox_2d must contain four numbers")
        if not all(
            isinstance(coordinate, (int, float))
            and not isinstance(coordinate, bool)
            and math.isfinite(coordinate)
            for coordinate in bbox
        ):
            raise TypeError(f"box {index}.bbox_2d must contain finite numbers")
        x1, y1, x2, y2 = (float(coordinate) for coordinate in bbox)
        if not all(0.0 <= coordinate <= 1000.0 for coordinate in (x1, y1, x2, y2)):
            raise ValueError(f"box {index}.bbox_2d must use coordinates from 0 to 1000")
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"box {index}.bbox_2d has invalid corners")
        boxes.append((x1, y1, x2, y2))
    if not boxes:
        raise ValueError("the grounding response contained no boxes")
    return boxes


@dataclass(frozen=True)
class QwenBoxDetector:
    """Use a local Qwen vision model only to identify archive-box locations."""

    client: VisionModelClient
    name: str = "qwen"

    def detect(self, image: InputImage) -> list[DetectedRegion]:
        result = self.client.query(image.file_path, GROUNDING_QUESTION, GROUNDING_SCHEMA)
        normalized_boxes = validate_grounding_result(result)
        regions: list[DetectedRegion] = []
        for x1, y1, x2, y2 in normalized_boxes:
            left = math.floor(x1 * image.width / 1000.0)
            top = math.floor(y1 * image.height / 1000.0)
            right = math.ceil(x2 * image.width / 1000.0)
            bottom = math.ceil(y2 * image.height / 1000.0)
            regions.append(
                DetectedRegion(
                    BoundingBox((left, top), (right, bottom)),
                    confidence=1.0,
                )
            )
        return regions
