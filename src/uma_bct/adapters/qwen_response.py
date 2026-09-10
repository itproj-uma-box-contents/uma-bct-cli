"""Shared checks for Qwen JSON replies."""

from __future__ import annotations

from typing import Any


def wrap_boxes(value: Any) -> dict[str, Any]:
    """Put accepted reply shapes into one boxes list without changing values."""

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
