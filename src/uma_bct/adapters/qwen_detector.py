"""Qwen box localization translated into detector-neutral pixel regions."""

from __future__ import annotations

import math
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2

from ..domain import BoundingBox, DetectedRegion, InputImage
from ..imaging import read_image
from ..ports import VisionModelClient
from .qwen_response import response_boxes

MAX_DETECTOR_PROPOSALS = 100

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
            "maxItems": MAX_DETECTOR_PROPOSALS,
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


def validate_grounding_proposals(value: Any) -> tuple[list[tuple[float, float, float, float]], tuple[str, ...]]:
    """Validate proposals independently and report malformed entries."""

    proposals = response_boxes(
        value,
        maximum=MAX_DETECTOR_PROPOSALS,
        limit_subject="the grounding response",
    )

    boxes: list[tuple[float, float, float, float]] = []
    dropped: list[str] = []
    for index, item in enumerate(proposals, start=1):
        try:
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
                raise ValueError(
                    f"box {index}.bbox_2d must use coordinates from 0 to 1000"
                )
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"box {index}.bbox_2d has invalid corners")
        except (TypeError, ValueError) as error:
            dropped.append(str(error))
            continue
        boxes.append((x1, y1, x2, y2))
    if not boxes:
        if not proposals:
            raise ValueError("the grounding response contained no boxes")
        detail = f": {'; '.join(dropped[:3])}" if dropped else ""
        raise ValueError(f"the grounding response contained no valid boxes{detail}")
    return boxes, tuple(dropped)


def validate_grounding_result(value: Any) -> list[tuple[float, float, float, float]]:
    """Return valid normalized boxes for callers that do not need warnings."""

    boxes, _dropped = validate_grounding_proposals(value)
    return boxes


@contextmanager
def sanitized_detection_image(image: InputImage):
    """Yield an EXIF-free PNG so even opted-in remote inference omits metadata."""

    with tempfile.TemporaryDirectory(prefix="uma-bct-detection-") as directory:
        image_path = Path(directory) / "source.png"
        if not cv2.imwrite(str(image_path), read_image(image.file_path)):
            raise OSError("Could not create a sanitized detection image.")
        image_path.chmod(0o600)
        yield image_path


@dataclass
class QwenBoxDetector:
    """Use a local Qwen vision model only to identify archive-box locations."""

    client: VisionModelClient
    name: str = "qwen"
    dropped_proposal_warnings: tuple[str, ...] = field(default=(), init=False)

    def detect(self, image: InputImage) -> list[DetectedRegion]:
        self.dropped_proposal_warnings = ()
        with sanitized_detection_image(image) as image_path:
            result = self.client.query(image_path, GROUNDING_QUESTION, GROUNDING_SCHEMA)
        normalized_boxes, dropped = validate_grounding_proposals(result)
        regions: list[DetectedRegion] = []
        conversion_warnings: list[str] = []
        for index, (x1, y1, x2, y2) in enumerate(normalized_boxes, start=1):
            left = math.floor(x1 * image.width / 1000.0)
            top = math.floor(y1 * image.height / 1000.0)
            right = math.ceil(x2 * image.width / 1000.0)
            bottom = math.ceil(y2 * image.height / 1000.0)
            try:
                region = DetectedRegion(
                    BoundingBox((left, top), (right, bottom)), confidence=1.0
                )
            except ValueError as error:
                conversion_warnings.append(
                    f"valid proposal {index} became unusable in image pixels: {error}"
                )
                continue
            regions.append(region)
        self.dropped_proposal_warnings = tuple(
            f"Qwen dropped a malformed proposal: {warning}"
            for warning in (*dropped, *conversion_warnings)
        )
        if not regions:
            raise ValueError("the grounding response contained no usable pixel boxes")
        return regions
