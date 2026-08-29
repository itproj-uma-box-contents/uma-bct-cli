"""Qwen transcription of detector-independent archive-box crops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict

from ..domain import Box, BoxText, BoxTranscription, Label
from ..ports import VisionModelClient

QUESTION = """This photo shows several cardboard archive boxes on a shelf.

For EVERY box fully visible in the photo, report:
- "position": where it sits, e.g. "top-left", "top-middle", "bottom-right".
- "unit": the 4-digit Unit number from the printed white label (e.g. "Unit 0004" -> "0004").
  Use "" if you cannot read it.
- "typed": the printed/typed text on the box, mainly the white label. Transcribe it in full.
- "handwritten": the handwritten text, one entry per visual line, top to bottom.
- "transcription_confidence": an integer from 0 to 100, without a percent sign.
  Estimate how confident you are that the unit, typed text, and handwritten text
  together are entirely accurate and include all relevant visible text. Use 100
  for complete confidence and 0 for no confidence.

Rules:
- Keep handwriting and typed text strictly separate. The white label is typed, not handwritten.
- Transcribe handwriting verbatim, including crossed-out or partly legible words.
  Do not expand abbreviations. Use "" for a line you truly cannot make out.
- Lower transcription_confidence if any visible text is uncertain or may be missing.
  An empty transcription may score highly only when you are confident there is no
  relevant text; score it low when text appears to be present but is unreadable.
- Base transcription_confidence only on the transcription, not the box position,
  crop boundaries, localization, or image-region detection quality.
- Work left to right, top row first. Never merge two boxes together."""

SCHEMA = {
    "type": "object",
    "properties": {
        "boxes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "position": {"type": "string"},
                    "unit": {"type": "string"},
                    "typed": {"type": "array", "items": {"type": "string"}},
                    "handwritten": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "transcription_confidence": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 100,
                    },
                },
                "required": [
                    "position",
                    "unit",
                    "typed",
                    "handwritten",
                    "transcription_confidence",
                ],
            },
        }
    },
    "required": ["boxes"],
}

# Backward-compatible name used by this modular adapter's earlier tests/callers.
TRANSCRIPTION_SCHEMA = SCHEMA


class BoxRecord(TypedDict):
    position: str
    unit: str
    typed: list[str]
    handwritten: list[str]
    transcription_confidence: int


def validate_transcription_result(value: Any) -> list[BoxRecord]:
    """Validate the same multi-box response accepted by the reference pipeline."""

    if not isinstance(value, dict) or not isinstance(value.get("boxes"), list):
        raise TypeError("the response must be an object containing a boxes array")

    validated: list[BoxRecord] = []
    for index, item in enumerate(value["boxes"], start=1):
        if not isinstance(item, dict):
            raise TypeError(f"box {index} must be an object")
        for field in ("position", "unit"):
            if not isinstance(item.get(field), str):
                raise TypeError(f"box {index}.{field} must be a string")
        for field in ("typed", "handwritten"):
            lines = item.get(field)
            if not isinstance(lines, list) or not all(
                isinstance(line, str) for line in lines
            ):
                raise ValueError(f"box {index}.{field} must be an array of strings")
        confidence = item.get("transcription_confidence")
        if not isinstance(confidence, int) or isinstance(confidence, bool):
            raise TypeError(f"box {index}.transcription_confidence must be an integer")
        if not 0 <= confidence <= 100:
            raise ValueError(
                f"box {index}.transcription_confidence must be between 0 and 100"
            )
        validated.append(
            {
                "position": item["position"],
                "unit": item["unit"],
                "typed": list(item["typed"]),
                "handwritten": list(item["handwritten"]),
                "transcription_confidence": confidence,
            }
        )
    return validated


def crop_question(box: Box) -> str:
    """Build the reference crop prompt from this project's generic Box object."""

    x, y, width, height = box.bounding_box.as_xywh()
    return f"""This clean, unannotated image is a crop from a photo of archive boxes.

Original photo size: {box.source_image.width} x {box.source_image.height} pixels.
Crop bounding box in the original: x={x}, y={y}, width={width}, height={height}.
The crop centre is approximately {box.position} in the original photo.

For EVERY archive box fully or substantially visible in this crop, report:
- "position": its position in the ORIGINAL photo, e.g. "top-left" or "bottom-right".
- "unit": the 4-digit Unit number from the printed white label (e.g. "Unit 0004" -> "0004").
  Use "" if you cannot read it.
- "typed": the printed/typed text on the box, mainly the white label. Transcribe it in full.
- "handwritten": the handwritten text, one entry per visual line, top to bottom.
- "transcription_confidence": an integer from 0 to 100, without a percent sign.
  Estimate how confident you are that the unit, typed text, and handwritten text
  together are entirely accurate and include all relevant visible text. Use 100
  for complete confidence and 0 for no confidence.

A substantially visible box must show most of its front face plus its label or meaningful
handwriting. Omit thin edge fragments and boxes whose useful text is outside the crop.

Rules:
- Keep handwriting and typed text strictly separate. The white label is typed, not handwritten.
- Transcribe handwriting verbatim, including crossed-out or partly legible words.
  Do not expand abbreviations. Use "" for a line you truly cannot make out.
- Lower transcription_confidence if any visible text is uncertain or may be missing.
  An empty transcription may score highly only when you are confident there is no
  relevant text; score it low when text appears to be present but is unreadable.
- Base transcription_confidence only on the transcription, not the box position,
  crop boundaries, localization, or image-region detection quality.
- Work left to right, top row first. Never merge two boxes together.
- Return only information visible in this clean crop; do not infer missing text from context."""


def _to_domain(box: Box, item: BoxRecord) -> BoxTranscription:
    confidence = item["transcription_confidence"]
    typed = tuple(item["typed"])
    label = Label(box.box_id, item["unit"], typed) if item["unit"] or typed else None
    lines: list[BoxText] = []
    for text_type in ("typed", "handwritten"):
        for position, text in enumerate(item[text_type], start=1):
            lines.append(BoxText(position, text, text_type, confidence))
    return BoxTranscription(
        box_id=box.box_id,
        position=item["position"] or box.position,
        label=label,
        text_lines=tuple(lines),
        confidence=confidence,
    )


@dataclass(frozen=True)
class QwenTextTranscriber:
    """Transcribe one generic box crop using the reference Qwen request contract."""

    client: VisionModelClient
    name: str = "qwen"

    def transcribe(self, box: Box) -> BoxTranscription:
        result = self.client.query(box.image_file_path, crop_question(box), SCHEMA)
        records = validate_transcription_result(result)
        if not records:
            raise ValueError("the transcription response contained no boxes")

        # BoxDetector creates one primary crop per box. The reference schema remains
        # multi-box so the model request is unchanged; the modular domain currently
        # stores the first record until overlap-aware merge support is ported.
        return _to_domain(box, records[0])
