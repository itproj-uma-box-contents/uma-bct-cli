"""Shared checks for Qwen JSON replies."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any


def wrap_boxes(value: Any) -> dict[str, Any]:
    """Normalize canonical, bare-array, and single-record replies without merging records."""

    if isinstance(value, dict):
        if "boxes" in value:
            return value
        return {"boxes": [value]}
    if isinstance(value, list):
        if any(isinstance(item, dict) and "boxes" in item for item in value):
            raise TypeError("the response must not mix a boxes object with bare records")
        return {"boxes": value}
    raise TypeError("the response must be an object, a record array, or adjacent records")


def response_boxes(value: Any, *, maximum: int, limit_subject: str) -> list[Any]:
    """Return the boxes list after basic checks."""

    wrapped = wrap_boxes(value)
    if not isinstance(wrapped.get("boxes"), list):
        raise TypeError("the response must be an object containing a boxes array")
    boxes = wrapped["boxes"]
    if len(boxes) > maximum:
        raise ValueError(f"{limit_subject} must contain at most {maximum} boxes")
    return boxes


def check_text(item: dict[str, Any], *, box_number: int, field: str, maximum_characters: int) -> str:
    """Check one required text field and its length."""

    value = item.get(field)
    if not isinstance(value, str):
        raise TypeError(f"box {box_number}.{field} must be a string")
    if len(value) > maximum_characters:
        raise ValueError(f"box {box_number}.{field} must contain at most {maximum_characters} characters")
    return value


def check_text_lines(
    item: dict[str, Any], *, box_number: int, field: str, maximum_items: int, maximum_characters: int
) -> list[str]:
    """Check a required list of text lines."""

    value = item.get(field)
    if not isinstance(value, list) or not all(isinstance(line, str) for line in value):
        raise ValueError(f"box {box_number}.{field} must be an array of strings")
    if len(value) > maximum_items:
        raise ValueError(f"box {box_number}.{field} must contain at most {maximum_items} lines")
    if any(len(line) > maximum_characters for line in value):
        raise ValueError(f"box {box_number}.{field} lines must contain at most {maximum_characters} characters")
    return list(value)


def is_content_token(token: str) -> bool:
    """Check whether a token holds copied characters rather than JSON syntax.

    Structural tokens (braces, quotes, colons) can show very low probability
    purely because the schema forced them, not because the model was unsure;
    they must be excluded from a confidence score.
    """

    return any(character.isalnum() for character in token)


def box_object_spans(raw_text: str) -> list[tuple[int, int]]:
    """Find the character span of each box object in a {"boxes": [...]} reply.

    Scans for brace depth, skipping braces inside string content, so it
    works without a full JSON parse. Only the canonical one-object-one-array
    shape is recognized; a noncanonical reply (bare array, adjacent objects)
    yields no spans, which callers must treat as "unclear", not an error.
    """

    spans: list[tuple[int, int]] = []
    depth = 0
    in_string = False
    escape = False
    start: int | None = None
    for index, character in enumerate(raw_text):
        if in_string:
            if escape:
                escape = False
            elif character == "\\":
                escape = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            if depth == 1:
                start = index
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 1 and start is not None:
                spans.append((start, index + 1))
                start = None
    return spans


def box_token_confidences(token_logprobs: Sequence[tuple[str, float]], box_count: int) -> list[int | None]:
    """Score each box 0-100 from its single least-confident content token.

    Takes the minimum rather than an average so one wrong or uncertain
    character is not diluted by a long run of confident ones. A box's score
    is ``None`` when logprobs are unavailable or the token stream could not
    be split into exactly ``box_count`` spans; treat ``None`` as "unknown".
    """

    if not token_logprobs or box_count <= 0:
        return [None] * box_count

    raw_text = "".join(token for token, _logprob in token_logprobs)
    spans = box_object_spans(raw_text)
    if len(spans) != box_count:
        return [None] * box_count

    offsets: list[tuple[int, int, float]] = []
    cursor = 0
    for token, logprob in token_logprobs:
        offsets.append((cursor, cursor + len(token), logprob))
        cursor += len(token)

    scores: list[int | None] = []
    for start, end in spans:
        content_logprobs = [
            logprob
            for token_start, token_end, logprob in offsets
            if token_start < end
            and token_end > start
            and is_content_token(raw_text[max(token_start, start) : min(token_end, end)])
        ]
        # A valid log-probability is never positive, so this is already
        # bounded to (0, 100]; clamp anyway so a malformed value from a
        # future server/model can never leave the documented 0-100 range.
        scores.append(None if not content_logprobs else max(0, min(100, round(100 * math.exp(min(content_logprobs))))))
    return scores
