"""Qwen transcription of detector-independent archive-box crops.

``QUESTION``/``RULES`` encode the actual data-entry policy in the prompt
itself: identifiers must be copied character-by-character rather than
guessed, printed and handwritten text must stay in separate fields, and
confidence must reflect transcription accuracy rather than crop quality.
``UNCERTAINTY`` is a second, code-side check on top of the model's own
confidence score -- it flags a transcription for review when any field is
empty or contains hedging language ("possibly", "illegible", a literal "?")
or CJK characters, which in this all-English archive context are a strong
signal of a misrecognition rather than genuine content. ``transcribe`` and
``transcribe_full_image`` share the same validation and prompt fields
(``FIELDS``/``RULES``); only the framing question differs between a single
crop and a whole photo.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, TypedDict

from ..domain import Box, BoxTranscription
from ..ports import VisionModelClient
from .qwen_response import bounded_string, bounded_string_list, response_boxes

MAX_TRANSCRIPTION_RECORDS = 100
MAX_TEXT_LINES_PER_FIELD = 100
MAX_TEXT_CHARACTERS = 4096

FIELDS = """For each reported physical box, provide:
- "position": where it sits in the original photo, such as "top-left".
- "record_identifiers": every visible record-set identifier in reading order.
  Supported examples include "1960.0009", "96/9", "UMA-AC-000000330", and
  "UMA-SRE-20160038". Use [] when none is legible.
- "unit": the visible printed or handwritten box number as digits only. Older
  box numbers may be unpadded. Use "" when it is missing, crossed out, or
  ambiguous.
- "typed": every printed visual line. Include printed identifiers and complete
  Unit lines verbatim even though they also appear in structured fields.
- "handwritten": every handwritten visual line, top to bottom. Include a
  handwritten unit number in its complete visual line.
- "auxiliary_annotations": explicit condition/content labels that are not the
  record origin, identifier, or Unit number. Use [] when absent.
- "transcription_confidence": an integer from 0 to 100 for the accuracy and
  completeness of the unit, printed text, and handwriting together."""

RULES = """Rules:
- Copy identifiers character by character. A YYYY.NNNN identifier has exactly
  four digits after the dot; never append the separate Unit number.
- Omit an uncertain identifier and lower confidence instead of guessing.
- Keep printed and handwritten text strictly separate. The white label is typed.
- Transcribe handwriting verbatim, including crossed-out or partly legible
  words. Do not expand abbreviations.
- Never translate, summarize, describe, or classify handwriting. Do not insert
  model comments such as "handwritten annotation" or "signature".
- Use confidence below 80 whenever a visible character may be wrong or missing.
  An empty transcription may score highly only when no relevant text is visible.
- Base confidence on transcription, not position, crop boundaries, or detection.
- Work left to right and top row first. Never merge two physical boxes.
- Return exactly one JSON object containing one top-level "boxes" array; never
  return separate JSON objects.
- Return only the requested JSON."""

QUESTION = f"""This photo shows several cardboard archive boxes on a shelf.

Report every substantially visible box, working left to right and top row first.
{FIELDS}

{RULES}"""

SCHEMA = {
    "type": "object",
    "properties": {
        "boxes": {
            "type": "array",
            "maxItems": MAX_TRANSCRIPTION_RECORDS,
            "items": {
                "type": "object",
                "properties": {
                    "position": {"type": "string", "maxLength": MAX_TEXT_CHARACTERS},
                    "record_identifiers": {
                        "type": "array",
                        "maxItems": MAX_TEXT_LINES_PER_FIELD,
                        "items": {
                            "type": "string",
                            "maxLength": MAX_TEXT_CHARACTERS,
                        },
                    },
                    "unit": {"type": "string", "maxLength": MAX_TEXT_CHARACTERS},
                    "typed": {
                        "type": "array",
                        "maxItems": MAX_TEXT_LINES_PER_FIELD,
                        "items": {
                            "type": "string",
                            "maxLength": MAX_TEXT_CHARACTERS,
                        },
                    },
                    "handwritten": {
                        "type": "array",
                        "maxItems": MAX_TEXT_LINES_PER_FIELD,
                        "items": {
                            "type": "string",
                            "maxLength": MAX_TEXT_CHARACTERS,
                        },
                    },
                    "auxiliary_annotations": {
                        "type": "array",
                        "maxItems": MAX_TEXT_LINES_PER_FIELD,
                        "items": {
                            "type": "string",
                            "maxLength": MAX_TEXT_CHARACTERS,
                        },
                    },
                    "transcription_confidence": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 100,
                    },
                },
                "required": [
                    "position",
                    "record_identifiers",
                    "unit",
                    "typed",
                    "handwritten",
                    "auxiliary_annotations",
                    "transcription_confidence",
                ],
            },
        }
    },
    "required": ["boxes"],
}

UNCERTAINTY = re.compile(
    r"\?|\b(?:possibly|probably|maybe|uncertain|illegible|unreadable|"
    r"unclear|unknown|ambiguous|signature)\b|\ufffd|[\u3400-\u4dbf\u4e00-\u9fff]",
    re.IGNORECASE,
)


class BoxRecord(TypedDict):
    position: str
    record_identifiers: list[str]
    unit: str
    typed: list[str]
    handwritten: list[str]
    auxiliary_annotations: list[str]
    transcription_confidence: int


def _lines(item: dict[str, Any], index: int, field: str) -> list[str]:
    return bounded_string_list(
        item,
        box_number=index,
        field=field,
        maximum_items=MAX_TEXT_LINES_PER_FIELD,
        maximum_characters=MAX_TEXT_CHARACTERS,
    )


def _validate_record(item: Any, index: int) -> BoxRecord:
    if not isinstance(item, dict):
        raise TypeError(f"box {index} must be an object")
    record: BoxRecord = {
        "position": bounded_string(
            item,
            box_number=index,
            field="position",
            maximum_characters=MAX_TEXT_CHARACTERS,
        ),
        "record_identifiers": _lines(item, index, "record_identifiers"),
        "unit": bounded_string(
            item,
            box_number=index,
            field="unit",
            maximum_characters=MAX_TEXT_CHARACTERS,
        ),
        "typed": _lines(item, index, "typed"),
        "handwritten": _lines(item, index, "handwritten"),
        "auxiliary_annotations": _lines(item, index, "auxiliary_annotations"),
        "transcription_confidence": 0,
    }
    confidence = item.get("transcription_confidence")
    if not isinstance(confidence, int) or isinstance(confidence, bool):
        raise TypeError(f"box {index}.transcription_confidence must be an integer")
    if not 0 <= confidence <= 100:
        raise ValueError(
            f"box {index}.transcription_confidence must be between 0 and 100"
        )
    record["transcription_confidence"] = confidence
    return record


def validate_transcription_result(value: Any) -> list[BoxRecord]:
    records = response_boxes(
        value,
        maximum=MAX_TRANSCRIPTION_RECORDS,
        limit_subject="the response",
    )
    return [_validate_record(item, index) for index, item in enumerate(records, 1)]


def crop_question(box: Box) -> str:
    x, y, width, height = box.bounding_box.as_xywh()
    return f"""This is a clean crop from a photo of archive boxes.

Original photo: {box.source_image.width} x {box.source_image.height} pixels.
Crop: x={x}, y={y}, width={width}, height={height}; centre={box.position}.

Report the central archive box. Report multiple boxes only when multiple complete
front faces occupy comparable central areas; ignore fragments at crop edges.
{FIELDS}

{RULES}"""


def _to_domain(box: Box, item: BoxRecord) -> BoxTranscription:
    values = (
        *item["typed"],
        *item["handwritten"],
        *item["auxiliary_annotations"],
    )
    return BoxTranscription(
        box_id=box.box_id,
        position=item["position"] or box.position,
        confidence=item["transcription_confidence"],
        record_identifiers=tuple(item["record_identifiers"]),
        unit_number=item["unit"],
        typed_lines=tuple(item["typed"]),
        handwritten_lines=tuple(item["handwritten"]),
        auxiliary_annotations=tuple(item["auxiliary_annotations"]),
        needs_review=any(
            not value.strip() or UNCERTAINTY.search(value) for value in values
        ),
    )


@dataclass(frozen=True)
class QwenTextTranscriber:
    client: VisionModelClient
    name: str = "qwen"

    def _request(self, box: Box, prompt: str) -> list[BoxTranscription]:
        records = validate_transcription_result(
            self.client.query(
                box.image_file_path,
                prompt,
                SCHEMA,
                validator=validate_transcription_result,
            )
        )
        return [_to_domain(box, record) for record in records]

    def transcribe(self, box: Box) -> list[BoxTranscription]:
        return self._request(box, crop_question(box))

    def transcribe_full_image(self, box: Box) -> list[BoxTranscription]:
        return self._request(box, QUESTION)
