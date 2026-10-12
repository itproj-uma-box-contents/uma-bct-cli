"""Read archive box text with Qwen.

The prompt keeps IDs, printed text, and handwriting separate. The code checks
every field and sends uncertain text for human review.

When Qwen copies one clear printed Unit or Box label but omits the duplicate
structured unit field, the exact visible digits are recovered and flagged for
review. Bare, handwritten, conflicting, uncertain, and crossed-out values
remain blank.

The model is never asked to self-report a confidence score -- that measures
whether the model believes its answer, not whether it is right, and LLMs are
poorly calibrated at that kind of self-assessment. Instead, ``confidence`` is
derived from the model's own token log-probabilities during generation (see
``qwen_response.box_token_confidences``): the probability it assigned to the
single least-confident character it emitted for that box. A box whose score
could not be computed is treated as zero confidence and sent for review
rather than guessing a plausible-looking number.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypedDict

from ..domain import Box, BoxTranscription
from ..identity import resolve_unit_claim
from ..ports import VisionModelClient
from .qwen_response import box_token_confidences, check_text, check_text_lines, response_boxes

MAX_TRANSCRIPTION_RECORDS = 100
MAX_TEXT_LINES_PER_FIELD = 100
MAX_TEXT_CHARACTERS = 4096

FIELDS = """For each reported physical box, provide:
- "position": where it sits in the original photo, such as "top-left".
- "record_identifiers": every visible record-set identifier in reading order.
  Preserve any visible identifier or prefix exactly, including identifiers not
  matching the examples "1960.0009", "96/9", "UMA-AC-000000330", and
  "UMA-SRE-20160038". If a label reads "<prefix> Unit <number>", return the
  prefix in this field. Use [] when none is legible.
- "unit": the visible printed or handwritten box number as digits only. Older
  box numbers may be unpadded. Use "" when it is missing, crossed out, or
  ambiguous.
- "typed": every printed visual line in reading order. This text is used to
  build Unit Type, so include all printed names, headings, identifiers, and
  complete "<prefix> Unit <number>" lines verbatim even though their parts also
  appear in structured fields.
- "handwritten": every handwritten visual line, top to bottom. Include a
  handwritten unit number in its complete visual line.
- "auxiliary_annotations": explicit condition/content labels that are not the
  record origin, identifier, or Unit number. Use [] when absent."""

RULES = """Rules:
- Copy identifiers and prefixes character by character; do not restrict them
  to a known catalogue or organisation prefix. A YYYY.NNNN identifier has
  exactly four digits after the dot; never append the separate Unit number.
- Omit an uncertain identifier and lower confidence instead of guessing.
- Keep printed and handwritten text strictly separate. The white label is typed.
- Transcribe handwriting verbatim, including crossed-out or partly legible
  words. Do not expand abbreviations.
- Never translate, summarize, describe, or classify handwriting. Do not insert
  model comments such as "handwritten annotation" or "signature".
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
                        "items": {"type": "string", "maxLength": MAX_TEXT_CHARACTERS},
                    },
                    "unit": {"type": "string", "maxLength": MAX_TEXT_CHARACTERS},
                    "typed": {
                        "type": "array",
                        "maxItems": MAX_TEXT_LINES_PER_FIELD,
                        "items": {"type": "string", "maxLength": MAX_TEXT_CHARACTERS},
                    },
                    "handwritten": {
                        "type": "array",
                        "maxItems": MAX_TEXT_LINES_PER_FIELD,
                        "items": {"type": "string", "maxLength": MAX_TEXT_CHARACTERS},
                    },
                    "auxiliary_annotations": {
                        "type": "array",
                        "maxItems": MAX_TEXT_LINES_PER_FIELD,
                        "items": {"type": "string", "maxLength": MAX_TEXT_CHARACTERS},
                    },
                },
                "required": ["position", "record_identifiers", "unit", "typed", "handwritten", "auxiliary_annotations"],
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


def validated_lines(item: dict[str, Any], index: int, field: str) -> list[str]:
    return check_text_lines(
        item, box_number=index, field=field, maximum_items=MAX_TEXT_LINES_PER_FIELD, maximum_characters=MAX_TEXT_CHARACTERS
    )


def validate_record(item: Any, index: int) -> BoxRecord:
    if not isinstance(item, dict):
        raise TypeError(f"box {index} must be an object")
    return {
        "position": check_text(item, box_number=index, field="position", maximum_characters=MAX_TEXT_CHARACTERS),
        "record_identifiers": validated_lines(item, index, "record_identifiers"),
        "unit": check_text(item, box_number=index, field="unit", maximum_characters=MAX_TEXT_CHARACTERS),
        "typed": validated_lines(item, index, "typed"),
        "handwritten": validated_lines(item, index, "handwritten"),
        "auxiliary_annotations": validated_lines(item, index, "auxiliary_annotations"),
    }


def validate_transcription_result(value: Any) -> list[BoxRecord]:
    records = response_boxes(value, maximum=MAX_TRANSCRIPTION_RECORDS, limit_subject="the response")
    return [validate_record(item, index) for index, item in enumerate(records, 1)]


def crop_context(box: Box) -> str:
    x, y, width, height = box.bounding_box.as_xywh()
    return f"""This is a clean crop from a photo of archive boxes.

Original photo: {box.source_image.width} x {box.source_image.height} pixels.
Crop: x={x}, y={y}, width={width}, height={height}; centre={box.position}."""


def crop_question(box: Box) -> str:
    return f"""{crop_context(box)}

Report the central archive box. Report multiple boxes only when multiple complete
front faces occupy comparable central areas; ignore fragments at crop edges.
{FIELDS}

{RULES}"""


HOSTED_PHYSICAL_BOX_RULES = """Physical-box checks:
- Report only substantially visible archive-box front faces. A side panel, lid,
  shelf, gap, shadow, or loose label is not a separate box. A narrow edge strip
  with no label area or handwriting is a fragment.
- If no qualifying front face is visible, return {"boxes": []}. Never create an
  empty placeholder record for a non-box region or an edge fragment.
- A blank record (blank fields, [] lists) is allowed ONLY for a complete front
  face that is fully visible and genuinely has no text. A face that is cut off
  by the image or crop edge and shows no label or handwriting is a fragment,
  not a blank box.
- For a face cut off at the edge, keep it only if its label or handwriting is
  at least partly visible; transcribe just what is visible and do not invent
  missing text.
- Never invent missing boxes to complete a sequence. Missing-number suggestions
  are handled after transcription.
- Do not copy text from a neighbouring face into the target box's record."""

HOSTED_FULL_IMAGE_QUESTION = f"{QUESTION}\n\n{HOSTED_PHYSICAL_BOX_RULES}"


def hosted_crop_question(box: Box) -> str:
    return f"""{crop_context(box)}

Report the central archive box IF one is present. Report multiple boxes only
when multiple qualifying front faces occupy comparable central areas.

{FIELDS}

{HOSTED_PHYSICAL_BOX_RULES}

{RULES}"""


def to_domain(box: Box, item: BoxRecord, token_confidence: int | None) -> BoxTranscription:
    values = (*item["typed"], *item["handwritten"], *item["auxiliary_annotations"])
    unit_number, unit_needs_review = resolve_unit_claim(item["unit"], item["typed"], item["handwritten"])
    return BoxTranscription(
        box_id=box.box_id,
        position=item["position"] or box.position,
        # An unavailable score is zero confidence, forced onto review, never
        # a guessed plausible-looking number.
        confidence=token_confidence if token_confidence is not None else 0,
        record_identifiers=tuple(item["record_identifiers"]),
        unit_number=unit_number,
        typed_lines=tuple(item["typed"]),
        handwritten_lines=tuple(item["handwritten"]),
        auxiliary_annotations=tuple(item["auxiliary_annotations"]),
        needs_review=(
            token_confidence is None
            or unit_needs_review
            or any(not value.strip() or UNCERTAINTY.search(value) for value in values)
        ),
    )


@dataclass(frozen=True)
class QwenTextTranscriber:
    client: VisionModelClient
    name: str = "qwen"
    crop_prompt: Callable[[Box], str] = field(default=crop_question, kw_only=True, repr=False)
    full_image_prompt: str = field(default=QUESTION, kw_only=True, repr=False)

    def request(self, box: Box, prompt: str) -> list[BoxTranscription]:
        # Validate the model's structure before mapping it to domain objects.
        # The per-box score comes from returned tokens, not self-rated text.
        response = self.client.query(
            box.image_file_path, prompt, SCHEMA, validator=validate_transcription_result, logprobs=True
        )
        records = validate_transcription_result(response.value)
        confidences = box_token_confidences(response.token_logprobs, len(records))
        return [to_domain(box, record, confidence) for record, confidence in zip(records, confidences, strict=True)]

    def transcribe(self, box: Box) -> list[BoxTranscription]:
        return self.request(box, self.crop_prompt(box))

    def transcribe_full_image(self, box: Box) -> list[BoxTranscription]:
        return self.request(box, self.full_image_prompt)
