"""Qwen transcription of detector-independent archive-box crops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..domain import Box, BoxText, BoxTranscription, Label
from ..ports import VisionModelClient

TRANSCRIPTION_PROMPT = """This clean image is one crop from a shelf photo. Focus on the
primary archive box whose front face fills the crop and return exactly one box.

Report:
- "position": the supplied approximate position in the original photo.
- "unit": the 4-digit Unit number from the printed white label, or "".
- "typed": all printed or typed lines, including the white label.
- "handwritten": handwritten text, one visual line per array item, top to bottom.
- "transcription_confidence": an integer from 0 to 100 for the completeness and
  accuracy of unit, typed, and handwritten text together.

Keep typed and handwritten text separate. Transcribe verbatim, do not expand
abbreviations, and use an empty string for text that is visible but unreadable.
Do not infer missing text from context. Return only the requested JSON."""

TRANSCRIPTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "boxes": {
            "type": "array",
            "minItems": 1,
            "maxItems": 1,
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


def validate_transcription_result(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or not isinstance(value.get("boxes"), list):
        raise TypeError("The response must contain a boxes array.")
    if len(value["boxes"]) != 1 or not isinstance(value["boxes"][0], dict):
        raise ValueError("The response must contain exactly one box object.")

    item = value["boxes"][0]
    for field in ("position", "unit"):
        if not isinstance(item.get(field), str):
            raise TypeError(f"box.{field} must be a string.")
    for field in ("typed", "handwritten"):
        lines = item.get(field)
        if not isinstance(lines, list) or not all(
            isinstance(line, str) for line in lines
        ):
            raise TypeError(f"box.{field} must be an array of strings.")
    confidence = item.get("transcription_confidence")
    if not isinstance(confidence, int) or isinstance(confidence, bool):
        raise TypeError("box.transcription_confidence must be an integer.")
    if not 0 <= confidence <= 100:
        raise ValueError("box.transcription_confidence must be between 0 and 100.")
    return item


@dataclass(frozen=True)
class QwenTextTranscriber:
    """Transcribe one generic box crop using a local Qwen-compatible client."""

    client: VisionModelClient
    name: str = "qwen"

    def transcribe(self, box: Box) -> BoxTranscription:
        prompt = f"{TRANSCRIPTION_PROMPT}\n\nApproximate position: {box.position}."
        result = self.client.query(
            box.image_file_path,
            prompt,
            TRANSCRIPTION_SCHEMA,
        )
        item = validate_transcription_result(result)
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
