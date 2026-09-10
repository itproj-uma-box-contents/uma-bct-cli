"""Shared validation helpers for structured Qwen responses."""

from __future__ import annotations

from typing import Any


def normalize_boxes_envelope(value: Any) -> dict[str, Any]:
    """Wrap complete record forms in the canonical top-level envelope.

    A canonical object is returned by identity. Bare records and top-level
    record arrays are structurally equivalent to that object, while arrays
    containing envelope objects are ambiguous and must not be flattened.
    Record values are never copied, coerced, or repaired here; the adapter's
    existing semantic validator remains authoritative.
    """

    if isinstance(value, dict):
        if "boxes" in value:
            return value
        return {"boxes": [value]}
    if isinstance(value, list):
        if any(isinstance(item, dict) and "boxes" in item for item in value):
            raise TypeError("the response must not contain nested or mixed boxes envelopes")
        return {"boxes": value}
    raise TypeError("the response must be an object, a record array, or adjacent records")


def response_boxes(value: Any, *, maximum: int, limit_subject: str) -> list[Any]:
    """Return the top-level boxes array after common shape and size checks."""

    normalized = normalize_boxes_envelope(value)
    if not isinstance(normalized.get("boxes"), list):
        raise TypeError("the response must be an object containing a boxes array")
    boxes = normalized["boxes"]
    if len(boxes) > maximum:
        raise ValueError(f"{limit_subject} must contain at most {maximum} boxes")
    return boxes


def bounded_string(item: dict[str, Any], *, box_number: int, field: str, maximum_characters: int) -> str:
    """Validate one required bounded string field."""

    value = item.get(field)
    if not isinstance(value, str):
        raise TypeError(f"box {box_number}.{field} must be a string")
    if len(value) > maximum_characters:
        raise ValueError(f"box {box_number}.{field} must contain at most {maximum_characters} characters")
    return value


def bounded_string_list(
    item: dict[str, Any], *, box_number: int, field: str, maximum_items: int, maximum_characters: int
) -> list[str]:
    """Validate a required bounded string array."""

    value = item.get(field)
    if not isinstance(value, list) or not all(isinstance(line, str) for line in value):
        raise ValueError(f"box {box_number}.{field} must be an array of strings")
    if len(value) > maximum_items:
        raise ValueError(f"box {box_number}.{field} must contain at most {maximum_items} lines")
    if any(len(line) > maximum_characters for line in value):
        raise ValueError(f"box {box_number}.{field} lines must contain at most {maximum_characters} characters")
    return list(value)
