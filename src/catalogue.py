"""Map validated box transcriptions into the current catalogue schema."""

from __future__ import annotations

import re
from pathlib import Path

from constants import OUTPUT_COLUMNS

from .domain import BoxTranscription

MISSING_UNIT_FLAG = "Missing unit"


def _non_empty_lines(lines: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(line.strip() for line in lines if line.strip())


def _title_and_remaining_typed_lines(
    transcription: BoxTranscription,
) -> tuple[str, tuple[str, ...]]:
    typed_lines = _non_empty_lines(transcription.typed_lines)
    unit_number = (
        transcription.label.unit_number.strip() if transcription.label else ""
    )
    if not unit_number:
        # A title without its unit number is not a complete identifier. Retain all
        # visible typed text in Description and allow the missing-unit flag to
        # direct an archivist to the record.
        return "", typed_lines

    unit_pattern = re.compile(rf"(?<!\d){re.escape(unit_number)}(?!\d)")
    for index, line in enumerate(typed_lines):
        if unit_pattern.search(line):
            return " ".join(typed_lines[: index + 1]), typed_lines[index + 1 :]

    # Qwen can extract the unit field even when it omits that line from `typed`.
    # Keep any printed label text and append the explicit unit rather than losing
    # either part of the catalogue identifier.
    title_parts = (*typed_lines, f"Unit {unit_number}")
    return " ".join(title_parts), ()


def map_catalogue_record(
    transcription: BoxTranscription,
    image_file_path: str | Path,
) -> dict[str, str | int]:
    """Convert one transcription to the repository's current export columns.

    Start Date and End Date remain empty because date extraction has not yet
    been implemented. Confidence Score is the model's raw transcription score
    (0--100), not a calibrated probability.
    """

    title, remaining_typed_lines = _title_and_remaining_typed_lines(transcription)
    handwritten_lines = _non_empty_lines(transcription.handwritten_lines)
    description = "\n".join((*remaining_typed_lines, *handwritten_lines))
    unit_number = (
        transcription.label.unit_number.strip() if transcription.label else ""
    )

    record = {
        "Node": "Unit",
        "Title": title,
        "Description": description,
        "Start Date": "",
        "End Date": "",
        "Confidence Score": transcription.confidence,
        "Flags": "" if unit_number else MISSING_UNIT_FLAG,
        "Image": str(image_file_path),
    }
    return {column: record[column] for column in OUTPUT_COLUMNS}
