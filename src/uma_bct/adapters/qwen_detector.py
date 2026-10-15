"""Find archive boxes with Qwen or a hosted model and return pixel coordinates.

Qwen uses X-first coordinates; Gemini's native convention is Y-first. Both
use a 0-1000 grid. Requests use a fresh PNG with no EXIF data.
"""

from __future__ import annotations

import math
import tempfile
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Literal

from ..domain import BoundingBox, DetectedRegion, InputImage
from ..imaging import read_input_image, write_png
from ..ports import VisionModelClient
from .qwen_response import response_boxes

MAX_DETECTOR_PROPOSALS = 100

CoordinateOrder = Literal["xyxy", "yxyx"]
PromptProfile = Literal["qwen", "hosted"]

GROUNDING_OUTPUT_RULES = """Return exactly one JSON object containing one top-level "boxes" array;
never return separate JSON objects. Return only the requested JSON."""

_QWEN_GROUNDING_INSTRUCTIONS = """Locate every cardboard archive box whose front face is fully or
substantially visible. Return one tight bounding box per physical box as bbox_2d
{coordinates}, using coordinates normalized from 0 to 1000. Exclude thin box
fragments, shelf parts, and labels or signs as separate boxes. Do not transcribe
any text."""

GROUNDING_QUESTION = (
    _QWEN_GROUNDING_INSTRUCTIONS.format(coordinates="[x1, y1, x2, y2]") + " " + GROUNDING_OUTPUT_RULES
)

_HOSTED_GROUNDING_INSTRUCTIONS = """Detect physical cardboard archive-box FRONT FACES in this image.

A qualifying object is one substantially visible front face of one physical
archive box. Labels and handwriting belong to that face, not to separate boxes.
A side panel, lid, thin sliver of a neighbouring box, shelf board, shelf upright,
gap, shadow, or loose label is not a qualifying front face. A label is not
required, but a box with no readable label qualifies ONLY when its whole face
is clearly visible.

First count the qualifying front faces visually, then return exactly one tight
bounding rectangle for each face in left-to-right, top-row-first order. Do not
split one face into multiple rectangles or add boxes to fill gaps in a sequence.
Enclose the whole visible front face, including its printed label and handwriting;
do not crop just the label, an edge, or a narrow side strip. Base edges on
visible boundaries and do not extrapolate hidden boxes.

Boxes cut off by the image edge: include one ONLY if BOTH are true:
(1) at least about half of its face width is inside the image, and
(2) part of its label or handwriting is clearly visible.
If either is false, omit it. Never return a box that shows only a narrow strip
of cardboard at the image edge, even if you can tell a box continues beyond it.
When unsure whether something is a full face or a fragment, omit it.

Each box contains only "bbox_2d": {coordinates}, normalized to 0-1000 for the
supplied image. Y means vertical distance from the TOP; X means horizontal
distance from the LEFT. 0 is the top/left boundary and 1000 is the bottom/right
boundary. Use {{"boxes": []}} when no qualifying front face is visible.
Do not transcribe text or report the count separately."""

HOSTED_GROUNDING_QUESTION = (
    _HOSTED_GROUNDING_INSTRUCTIONS.format(coordinates="[x1, y1, x2, y2]") + "\n" + GROUNDING_OUTPUT_RULES
)
GEMINI_GROUNDING_QUESTION = (
    _HOSTED_GROUNDING_INSTRUCTIONS.format(coordinates="[ymin, xmin, ymax, xmax]") + "\n" + GROUNDING_OUTPUT_RULES
)

GROUNDING_SCHEMA = {
    "type": "object",
    "properties": {
        "boxes": {
            "type": "array",
            "maxItems": MAX_DETECTOR_PROPOSALS,
            "items": {
                "type": "object",
                "properties": {"bbox_2d": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4}},
                "required": ["bbox_2d"],
            },
        }
    },
    "required": ["boxes"],
}


def hosted_grounding_schema(coordinates: str) -> dict[str, Any]:
    schema = deepcopy(GROUNDING_SCHEMA)
    schema["properties"]["boxes"]["items"]["properties"]["bbox_2d"]["description"] = (
        f"One qualifying archive-box front face as {coordinates}, normalized to 0-1000. "
        "Do not return a label, side panel, shelf part, or thin fragment as a separate box. "
        "Omit boxes cut off by the image edge unless at least half the face is visible and text is visible."
    )
    return schema


HOSTED_GROUNDING_SCHEMA = hosted_grounding_schema("[x1, y1, x2, y2]")
GEMINI_GROUNDING_SCHEMA = hosted_grounding_schema("[ymin, xmin, ymax, xmax]")


def validate_grounding_proposals(
    value: Any, *, coordinate_order: CoordinateOrder = "xyxy"
) -> tuple[list[tuple[float, float, float, float]], tuple[str, ...]]:
    """Validate proposals and normalize either convention to X-first corners."""

    if coordinate_order not in {"xyxy", "yxyx"}:
        raise ValueError("Grounding coordinate order must be xyxy or yxyx.")

    normal_reply = isinstance(value, dict) and "boxes" in value
    proposals = response_boxes(value, maximum=MAX_DETECTOR_PROPOSALS, limit_subject="the grounding response")

    boxes: list[tuple[float, float, float, float]] = []
    dropped: list[str] = []
    # A normal boxes envelope may keep good proposals and report the bad ones.
    for index, item in enumerate(proposals, start=1):
        try:
            bbox = item.get("bbox_2d") if isinstance(item, dict) else None
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise TypeError(f"box {index}.bbox_2d must contain four numbers")
            if not all(
                isinstance(coordinate, (int, float)) and not isinstance(coordinate, bool) and math.isfinite(coordinate)
                for coordinate in bbox
            ):
                raise TypeError(f"box {index}.bbox_2d must contain finite numbers")
            first, second, third, fourth = (float(coordinate) for coordinate in bbox)
            x1, y1, x2, y2 = (
                (second, first, fourth, third) if coordinate_order == "yxyx" else (first, second, third, fourth)
            )
            if not all(0.0 <= coordinate <= 1000.0 for coordinate in (x1, y1, x2, y2)):
                raise ValueError(f"box {index}.bbox_2d must use coordinates from 0 to 1000")
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"box {index}.bbox_2d has invalid corners")
        except (OverflowError, TypeError, ValueError) as error:
            dropped.append(str(error))
            continue
        boxes.append((x1, y1, x2, y2))
    if not boxes:
        if not proposals:
            raise ValueError("the grounding response contained no boxes")
        detail = f": {'; '.join(dropped[:3])}" if dropped else ""
        raise ValueError(f"the grounding response contained no valid boxes{detail}")
    if dropped and not normal_reply:
        raise ValueError("a response without a boxes wrapper must contain only valid box records")
    return boxes, tuple(dropped)


@contextmanager
def sanitized_detection_image(image: InputImage):
    """Make a temporary PNG with no EXIF data."""

    with tempfile.TemporaryDirectory(prefix="uma-bct-detection-") as directory:
        image_path = Path(directory) / "source.png"
        if not write_png(image_path, read_input_image(image)):
            raise OSError("Could not create a sanitized detection image.")
        image_path.chmod(0o600)
        yield image_path


@dataclass
class QwenBoxDetector:
    """Use the shared detector with the selected model's coordinate convention."""

    client: VisionModelClient
    name: str = "qwen"
    coordinate_order: CoordinateOrder = field(default="xyxy", kw_only=True)
    prompt_profile: PromptProfile = field(default="qwen", kw_only=True)
    dropped_proposal_warnings: tuple[str, ...] = field(default=(), init=False)

    def detect(self, image: InputImage) -> list[DetectedRegion]:
        self.dropped_proposal_warnings = ()
        validator = partial(validate_grounding_proposals, coordinate_order=self.coordinate_order)
        if self.prompt_profile == "hosted":
            prompt, schema = (
                (GEMINI_GROUNDING_QUESTION, GEMINI_GROUNDING_SCHEMA)
                if self.coordinate_order == "yxyx" else (HOSTED_GROUNDING_QUESTION, HOSTED_GROUNDING_SCHEMA)
            )
        else:
            prompt, schema = GROUNDING_QUESTION, GROUNDING_SCHEMA
        # Qwen needs only pixels for grounding. The temporary PNG prevents
        # source EXIF from entering the model request.
        with sanitized_detection_image(image) as image_path:
            response = self.client.query(
                image_path, prompt, schema, validator=validator
            )
        normalized_boxes, dropped = validator(response.value)
        regions: list[DetectedRegion] = []
        conversion_warnings: list[str] = []
        # Grounding coordinates use a 0-1000 scale; round outward in source
        # pixels so the eventual crop does not cut the detected box short.
        for index, (x1, y1, x2, y2) in enumerate(normalized_boxes, start=1):
            left = math.floor(x1 * image.width / 1000.0)
            top = math.floor(y1 * image.height / 1000.0)
            right = math.ceil(x2 * image.width / 1000.0)
            bottom = math.ceil(y2 * image.height / 1000.0)
            try:
                region = DetectedRegion(BoundingBox((left, top), (right, bottom)), confidence=1.0)
            except ValueError as error:
                conversion_warnings.append(f"valid proposal {index} became unusable in image pixels: {error}")
                continue
            regions.append(region)
        self.dropped_proposal_warnings = tuple(
            f"Qwen dropped a malformed proposal: {warning}" for warning in (*dropped, *conversion_warnings)
        )
        if not regions:
            raise ValueError("the grounding response contained no usable pixel boxes")
        return regions
