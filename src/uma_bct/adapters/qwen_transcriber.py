"""Qwen transcription of detector-independent archive-box crops."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, TypedDict

from ..domain import Box, BoxText, BoxTranscription, Label
from ..identity import (
    RECORD_IDENTIFIER,
    identifier_evidence_details,
    normalise_identifier,
    unit_claim_is_corroborated,
)
from ..ports import VisionModelClient
from .qwen_response import bounded_string, bounded_string_list, response_boxes

MAX_TRANSCRIPTION_RECORDS = 100
MAX_TEXT_LINES_PER_FIELD = 100
MAX_TEXT_CHARACTERS = 4096

QUESTION = """This photo shows several cardboard archive boxes on a shelf.

For EVERY box fully visible in the photo, report:
- "position": where it sits, e.g. "top-left", "top-middle", "bottom-right".
- "record_identifiers": every record-set identifier visible on the primary label,
  in reading order. Supported examples include "1960.0009", "96/9",
  "UMA-AC-000000330", and "UMA-SRE-20160038". Use [] if none is legible.
- "unit": the box number, whether printed on the primary label or handwritten on
  an older box (e.g. "Unit 0004" -> "0004"). Older numbers may be unpadded;
  return only the visible digits. Use "" when missing, crossed out, or ambiguous.
- "typed": every printed/typed visual line on the box, mainly the white label.
  Include the printed record identifier and the complete printed Unit line here
  verbatim even though they are also returned in structured fields.
- "handwritten": the handwritten text, one entry per visual line, top to bottom.
  Include a handwritten box/unit number in its complete visual line here even
  though its digits are also returned in "unit".
- "auxiliary_annotations": text from additional condition/content labels that is not
  the record origin, record-set identifier, or Unit number. Use [] when absent.
- "transcription_confidence": an integer from 0 to 100, without a percent sign.
  Estimate how confident you are that the unit, typed text, and handwritten text
  together are entirely accurate and include all relevant visible text. Use 100
  for complete confidence and 0 for no confidence.

Rules:
- Copy identifiers character by character. A YYYY.NNNN identifier has exactly four
  digits after the dot; never append the separate Unit number to it. For example,
  a label showing "1990.0001" and Unit 30 means identifier "1990.0001", unit "30".
- If any identifier character is uncertain, omit that identifier and lower confidence
  instead of guessing.
- Keep handwriting and typed text strictly separate. The white label is typed, not handwritten.
- Transcribe handwriting verbatim, including crossed-out or partly legible words.
  Do not expand abbreviations. Use "" for a line you truly cannot make out.
- Never translate, summarize, describe, or classify handwriting. Do not insert
  parenthetical meta-comments such as "handwritten annotation" or "signature";
  return only characters that are actually visible on the box.
- Lower transcription_confidence if any visible text is uncertain or may be missing.
  If any transcribed character is uncertain, use a score below 80 so the record is
  flagged for review.
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
            "maxItems": MAX_TRANSCRIPTION_RECORDS,
            "items": {
                "type": "object",
                "properties": {
                    "position": {
                        "type": "string",
                        "maxLength": MAX_TEXT_CHARACTERS,
                    },
                    "record_identifiers": {
                        "type": "array",
                        "maxItems": MAX_TEXT_LINES_PER_FIELD,
                        "items": {
                            "type": "string",
                            "maxLength": MAX_TEXT_CHARACTERS,
                        },
                        "description": (
                            "Ordered visible identifiers; repeat a printed value in "
                            "typed or a handwritten value in handwritten."
                        ),
                    },
                    "unit": {
                        "type": "string",
                        "maxLength": MAX_TEXT_CHARACTERS,
                        "description": (
                            "Visible box-number digits, also repeated verbatim in "
                            "the source-appropriate raw text array."
                        ),
                    },
                    "typed": {
                        "type": "array",
                        "maxItems": MAX_TEXT_LINES_PER_FIELD,
                        "items": {
                            "type": "string",
                            "maxLength": MAX_TEXT_CHARACTERS,
                        },
                        "description": (
                            "Every printed visual line, including printed record "
                            "identifiers and complete Unit lines."
                        ),
                    },
                    "handwritten": {
                        "type": "array",
                        "maxItems": MAX_TEXT_LINES_PER_FIELD,
                        "items": {
                            "type": "string",
                            "maxLength": MAX_TEXT_CHARACTERS,
                        },
                        "description": (
                            "Every handwritten visual line, including handwritten "
                            "box or unit numbers."
                        ),
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

# Backward-compatible name used by this modular adapter's earlier tests/callers.
TRANSCRIPTION_SCHEMA = SCHEMA

UNCERTAINTY_MARKERS = {
    "?",
    "??",
    "???",
    "illegible",
    "unreadable",
    "unclear",
    "unknown",
    "ambiguous",
    "handwritten",
    "handwritten annotation",
    "handwritten note",
    "signature",
}
BRACKETED_META_COMMENT = re.compile(
    r"^[\[(]\s*(?:illegible|unreadable|unclear|unknown|ambiguous|"
    r"handwritten(?:\s+(?:annotation|note))?|signature)\s*[\])]$",
    re.IGNORECASE,
)
CJK_CHARACTER = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
UNCERTAINTY_EVIDENCE = re.compile(
    r"\?|\b(?:possibly|probably|maybe|uncertain|illegible|unreadable|"
    r"unclear|unknown|ambiguous)\b|\b(?:handwritten(?:\s+(?:annotation|note))?|"
    r"signature)\s*:",
    re.IGNORECASE,
)
SUSPICIOUS_YEAR_SUFFIX = re.compile(r"\b(?:18|19|20)\d{2}[A-RT-Z]\b", re.I)


class BoxRecord(TypedDict):
    position: str
    record_identifiers: list[str]
    unit: str
    typed: list[str]
    handwritten: list[str]
    auxiliary_annotations: list[str]
    transcription_confidence: int


def validate_transcription_result(value: Any) -> list[BoxRecord]:
    """Validate the same multi-box response accepted by the reference pipeline."""

    records = response_boxes(
        value,
        maximum=MAX_TRANSCRIPTION_RECORDS,
        limit_subject="the response",
    )

    validated: list[BoxRecord] = []
    for index, item in enumerate(records, start=1):
        if not isinstance(item, dict):
            raise TypeError(f"box {index} must be an object")
        position = bounded_string(
            item,
            box_number=index,
            field="position",
            maximum_characters=MAX_TEXT_CHARACTERS,
        )
        unit = bounded_string(
            item,
            box_number=index,
            field="unit",
            maximum_characters=MAX_TEXT_CHARACTERS,
        )
        typed = bounded_string_list(
            item,
            box_number=index,
            field="typed",
            maximum_items=MAX_TEXT_LINES_PER_FIELD,
            maximum_characters=MAX_TEXT_CHARACTERS,
        )
        handwritten = bounded_string_list(
            item,
            box_number=index,
            field="handwritten",
            maximum_items=MAX_TEXT_LINES_PER_FIELD,
            maximum_characters=MAX_TEXT_CHARACTERS,
        )
        # Additive fields remain optional so older cached responses stay readable.
        record_identifiers = bounded_string_list(
            item,
            box_number=index,
            field="record_identifiers",
            maximum_items=MAX_TEXT_LINES_PER_FIELD,
            maximum_characters=MAX_TEXT_CHARACTERS,
            default=[],
        )
        auxiliary_annotations = bounded_string_list(
            item,
            box_number=index,
            field="auxiliary_annotations",
            maximum_items=MAX_TEXT_LINES_PER_FIELD,
            maximum_characters=MAX_TEXT_CHARACTERS,
            default=[],
        )
        confidence = item.get("transcription_confidence")
        if not isinstance(confidence, int) or isinstance(confidence, bool):
            raise TypeError(f"box {index}.transcription_confidence must be an integer")
        if not 0 <= confidence <= 100:
            raise ValueError(
                f"box {index}.transcription_confidence must be between 0 and 100"
            )
        validated.append(
            {
                "position": position,
                "record_identifiers": record_identifiers,
                "unit": unit,
                "typed": typed,
                "handwritten": handwritten,
                "auxiliary_annotations": auxiliary_annotations,
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

Report the detector-targeted archive box whose front face occupies the centre and
largest area of this crop. Ignore neighbouring boxes that are only cut-off fragments
at a crop edge. Only report multiple boxes when multiple complete front faces occupy
comparable central areas in the crop. For each reported box, provide:
- "position": its position in the ORIGINAL photo, e.g. "top-left" or "bottom-right".
- "record_identifiers": every record-set identifier visible on the primary label,
  in reading order. Examples: "1960.0009", "96/9", "UMA-AC-000000330".
- "unit": the box number, whether printed on the primary label or handwritten on
  an older box (e.g. "Unit 0004" -> "0004"). Older numbers may be unpadded;
  return only the visible digits. Use "" when missing, crossed out, or ambiguous.
- "typed": every printed/typed visual line on the box, mainly the white label.
  Include the printed record identifier and the complete printed Unit line here
  verbatim even though they are also returned in structured fields.
- "handwritten": the handwritten text, one entry per visual line, top to bottom.
  Include a handwritten box/unit number in its complete visual line here even
  though its digits are also returned in "unit".
- "auxiliary_annotations": text from additional condition/content labels that is not
  the record origin, record-set identifier, or Unit number. Use [] when absent.
- "transcription_confidence": an integer from 0 to 100, without a percent sign.
  Estimate how confident you are that the unit, typed text, and handwritten text
  together are entirely accurate and include all relevant visible text. Use 100
  for complete confidence and 0 for no confidence.

A substantially visible box must show most of its front face plus its label or meaningful
handwriting. Omit thin edge fragments and boxes whose useful text is outside the crop.

Rules:
- Copy identifiers character by character. A YYYY.NNNN identifier has exactly four
  digits after the dot; never append the separate Unit number to it. For example,
  a label showing "1990.0001" and Unit 30 means identifier "1990.0001", unit "30".
- If any identifier character is uncertain, omit that identifier and lower confidence
  instead of guessing.
- Keep handwriting and typed text strictly separate. The white label is typed, not handwritten.
- Transcribe handwriting verbatim, including crossed-out or partly legible words.
  Do not expand abbreviations. Use "" for a line you truly cannot make out.
- Never translate, summarize, describe, or classify handwriting. Do not insert
  parenthetical meta-comments such as "handwritten annotation" or "signature";
  return only characters that are actually visible on the box.
- Lower transcription_confidence if any visible text is uncertain or may be missing.
  If any transcribed character is uncertain, use a score below 80 so the record is
  flagged for review.
  An empty transcription may score highly only when you are confident there is no
  relevant text; score it low when text appears to be present but is unreadable.
- Base transcription_confidence only on the transcription, not the box position,
  crop boundaries, localization, or image-region detection quality.
- Work left to right, top row first. Never merge two boxes together.
- Do not create a separate record for text from a neighbouring box fragment at an edge.
- Return only information visible in this clean crop; do not infer missing text from context."""


def identity_fields_need_review(item: BoxRecord) -> bool:
    """Flag structured identity fields not corroborated by transcribed text."""

    observed_values, malformed_identifiers = identifier_evidence_details(
        (*item["typed"], *item["handwritten"])
    )
    observed_identifiers = {identifier.casefold() for identifier in observed_values}
    structured_identifiers: set[str] = set()
    identifier_mismatch = False
    for candidate in item["record_identifiers"]:
        match = RECORD_IDENTIFIER.fullmatch(candidate.strip())
        if match is None:
            identifier_mismatch = True
            continue
        structured_identifiers.add(normalise_identifier(match.group(0)).casefold())
    identifier_mismatch = identifier_mismatch or (
        structured_identifiers != observed_identifiers
    )

    unit = item["unit"].strip()
    unit_mismatch = bool(unit) and not unit_claim_is_corroborated(
        unit, item["typed"], item["handwritten"]
    )
    return malformed_identifiers or identifier_mismatch or unit_mismatch


def text_fields_need_review(item: BoxRecord) -> bool:
    """Detect explicit uncertainty or model meta-comments in text fields."""

    values = (
        *item["typed"],
        *item["handwritten"],
        *item["auxiliary_annotations"],
    )
    for value in values:
        stripped = value.strip()
        if not stripped:
            return True
        folded = stripped.casefold().strip("[]() ")
        if (
            folded in UNCERTAINTY_MARKERS
            or BRACKETED_META_COMMENT.fullmatch(stripped)
            or UNCERTAINTY_EVIDENCE.search(stripped)
            or "\ufffd" in stripped
            # Qwen has been observed inserting Chinese meta-comment text at high
            # confidence. Retain it verbatim, but conservatively require review.
            or CJK_CHARACTER.search(stripped)
            or SUSPICIOUS_YEAR_SUFFIX.search(stripped)
        ):
            return True
    return False


def _to_domain(box: Box, item: BoxRecord) -> BoxTranscription:
    confidence = item["transcription_confidence"]
    typed = tuple(item["typed"])
    record_identifiers = tuple(item["record_identifiers"])
    label = (
        Label(
            box.box_id,
            item["unit"],
            typed,
            record_identifiers=record_identifiers,
        )
        if item["unit"] or typed or record_identifiers
        else None
    )
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
        needs_review=(
            identity_fields_need_review(item) or text_fields_need_review(item)
        ),
        auxiliary_annotations=tuple(item["auxiliary_annotations"]),
    )


@dataclass(frozen=True)
class QwenTextTranscriber:
    """Transcribe one generic box crop using the reference Qwen request contract."""

    client: VisionModelClient
    name: str = "qwen"

    def _request(self, box: Box, prompt: str) -> list[BoxTranscription]:
        result = self.client.query(box.image_file_path, prompt, SCHEMA)
        records = validate_transcription_result(result)
        return [_to_domain(box, record) for record in records]

    def transcribe(self, box: Box) -> list[BoxTranscription]:
        return self._request(box, crop_question(box))

    def transcribe_full_image(self, box: Box) -> list[BoxTranscription]:
        """Retry the untouched source image using the old full-image prompt."""

        return self._request(box, QUESTION)
