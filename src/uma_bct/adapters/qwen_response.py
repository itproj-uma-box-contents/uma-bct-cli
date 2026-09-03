"""Shared validation helpers for structured Qwen responses."""

from __future__ import annotations

from typing import Any

_REQUIRED = object()


def response_boxes(value: Any, *, maximum: int, limit_subject: str) -> list[Any]:
    """Return the top-level boxes array after common shape and size checks."""

    if not isinstance(value, dict) or not isinstance(value.get("boxes"), list):
        raise TypeError("the response must be an object containing a boxes array")
    boxes = value["boxes"]
    if len(boxes) > maximum:
        raise ValueError(f"{limit_subject} must contain at most {maximum} boxes")
    return boxes


def bounded_string(item: dict[str, Any], *, box_number: int, field: str,
    maximum_characters: int) -> str:
    """Validate one required bounded string field."""

    value = item.get(field)
    if not isinstance(value, str):
        raise TypeError(f"box {box_number}.{field} must be a string")
    if len(value) > maximum_characters:
        raise ValueError(
            f"box {box_number}.{field} must contain at most "
            f"{maximum_characters} characters"
        )
    return value


def bounded_string_list(item: dict[str, Any], *, box_number: int, field: str,
    maximum_items: int, maximum_characters: int, default: object = _REQUIRED) -> list[str]:
    """Validate a bounded string array, optionally accepting an omitted field."""

    value = item.get(field) if default is _REQUIRED else item.get(field, default)
    if not isinstance(value, list) or not all(isinstance(line, str) for line in value):
        raise ValueError(f"box {box_number}.{field} must be an array of strings")
    if len(value) > maximum_items:
        raise ValueError(
            f"box {box_number}.{field} must contain at most {maximum_items} lines"
        )
    if any(len(line) > maximum_characters for line in value):
        raise ValueError(
            f"box {box_number}.{field} lines must contain at most "
            f"{maximum_characters} characters"
        )
    return list(value)
