"""Map validated box transcriptions into the client catalogue schema."""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from pathlib import Path

from .constants import (
    AMBIGUOUS_IDENTIFIER_FLAG,
    INVALID_UNIT_FLAG,
    MISSING_IDENTIFIER_FLAG,
    MISSING_UNIT_FLAG,
    MULTIPLE_IDENTIFIERS_FLAG,
    OUTPUT_COLUMNS,
    TRANSCRIPTION_MISMATCH_FLAG,
)
from .domain import BoxTranscription
from .identity import (
    RECORD_IDENTIFIER,
    collect_unit_evidence,
    identifier_evidence_details,
    identity_only_line,
    normalise_identifier,
    normalise_unit_claim,
)

_SPREADSHEET_FORMULA_PREFIX = re.compile(r"^\s*[=+\-@]")


def escape_spreadsheet_formula(value: str) -> str:
    """Prevent spreadsheet applications from evaluating transcribed text."""

    return f"'{value}" if _SPREADSHEET_FORMULA_PREFIX.match(value) else value


def append_flag(current: str, flag: str) -> str:
    """Append one review flag without repeating it."""

    flags = [item.strip() for item in current.split(";") if item.strip()]
    if flag not in flags:
        flags.append(flag)
    return "; ".join(flags)


def non_empty_lines(lines: Iterable[str]) -> tuple[str, ...]:
    return tuple(
        visual_line.strip()
        for value in lines
        for visual_line in value.splitlines()
        if visual_line.strip()
    )


def _structured_identifiers(
    candidates: Iterable[str],
) -> tuple[tuple[str, ...], bool]:
    explicit_identifiers: list[str] = []
    explicit_seen: set[str] = set()
    invalid = False
    for candidate in candidates:
        match = RECORD_IDENTIFIER.fullmatch(candidate.strip())
        if match is None:
            invalid = True
            continue
        identifier = normalise_identifier(match.group(0))
        key = identifier.casefold()
        if key not in explicit_seen:
            explicit_seen.add(key)
            explicit_identifiers.append(identifier)
    return tuple(explicit_identifiers), invalid


def _raw_identifier_mismatch(
    transcription: BoxTranscription, explicit_identifiers: tuple[str, ...]
) -> bool:
    typed_identifiers, malformed_typed = identifier_evidence_details(
        transcription.typed_lines
    )
    handwritten_identifiers, malformed_handwritten = identifier_evidence_details(
        transcription.handwritten_lines
    )
    observed_identifiers = (*typed_identifiers, *handwritten_identifiers)
    observed_keys = {identifier.casefold() for identifier in observed_identifiers}
    explicit_keys = tuple(identifier.casefold() for identifier in explicit_identifiers)
    explicit_key_set = set(explicit_keys)
    order_mismatch = any(
        positions != sorted(positions)
        for source in (typed_identifiers, handwritten_identifiers)
        if (
            positions := [
                explicit_keys.index(identifier.casefold())
                for identifier in source
                if identifier.casefold() in explicit_key_set
            ]
        )
    )
    return (
        observed_keys != explicit_key_set
        or order_mismatch
        or malformed_typed
        or malformed_handwritten
    )


def record_identifiers(
    transcription: BoxTranscription,
) -> tuple[tuple[str, ...], bool, bool]:
    """Return corroborated identifiers, ambiguity evidence, and mismatch state."""

    if not transcription.record_identifiers:
        unexpected, malformed = identifier_evidence_details(
            (*transcription.typed_lines, *transcription.handwritten_lines)
        )
        ambiguous = bool(unexpected) or malformed
        return (), ambiguous, ambiguous

    identifiers, invalid = _structured_identifiers(transcription.record_identifiers)
    if (
        invalid
        or not identifiers
        or _raw_identifier_mismatch(transcription, identifiers)
    ):
        return (), True, True
    return identifiers, False, False


def normalise_unit(transcription: BoxTranscription) -> tuple[str, str | None, bool]:
    raw_unit = transcription.unit_number.strip()
    if not raw_unit:
        return "", MISSING_UNIT_FLAG, False
    normalized = normalise_unit_claim(raw_unit)
    if normalized is None:
        return "", INVALID_UNIT_FLAG, False
    evidence = collect_unit_evidence(
        transcription.typed_lines, transcription.handwritten_lines
    )
    if evidence.ambiguous or evidence.candidates != {normalized}:
        return "", INVALID_UNIT_FLAG, True
    return normalized, None, False


def image_reference(
    image_file_path: str | Path, catalogue_path: str | Path | None
) -> str:
    image_path = Path(image_file_path)
    if catalogue_path is None:
        return image_path.as_posix()

    destination_parent = Path(catalogue_path).parent
    try:
        absolute_image = (
            image_path if image_path.is_absolute() else image_path.resolve()
        )
        absolute_parent = (
            destination_parent
            if destination_parent.is_absolute()
            else destination_parent.resolve()
        )
        return Path(os.path.relpath(absolute_image, absolute_parent)).as_posix()
    except ValueError:
        # Windows cannot represent a relative path across drive letters.
        raise ValueError(
            "The catalogue and review-crop directory must be on the same drive "
            "so Image values remain relative. Choose --path-to-box-images on "
            "the catalogue drive."
        ) from None


def _record_flags(
    transcription: BoxTranscription,
    identifiers: tuple[str, ...],
    ambiguous_identifier: bool,
    identifier_mismatch: bool,
    unit_flag: str | None,
    unit_mismatch: bool,
    additional_flags: Iterable[str],
) -> list[str]:
    flags: list[str] = []
    if not identifiers:
        flags.append(
            AMBIGUOUS_IDENTIFIER_FLAG
            if ambiguous_identifier
            else MISSING_IDENTIFIER_FLAG
        )
    elif len(identifiers) > 1:
        flags.append(MULTIPLE_IDENTIFIERS_FLAG)
    if unit_flag:
        flags.append(unit_flag)
    if transcription.needs_review or identifier_mismatch or unit_mismatch:
        flags.append(TRANSCRIPTION_MISMATCH_FLAG)
    flags.extend(flag for flag in additional_flags if flag not in flags)
    return flags


def _description(transcription: BoxTranscription) -> str:
    values = (
        *non_empty_lines(transcription.handwritten_lines),
        *non_empty_lines(transcription.auxiliary_annotations),
    )
    visible = (
        line
        for line in values
        if not identity_only_line(
            line,
            transcription.record_identifiers,
            transcription.unit_number,
        )
    )
    return escape_spreadsheet_formula("; ".join(visible))


def map_catalogue_record(
    transcription: BoxTranscription,
    image_file_path: str | Path,
    *,
    catalogue_path: str | Path | None = None,
    additional_flags: Iterable[str] = (),
) -> dict[str, str | int]:
    """Convert one transcription to the client import and review columns."""

    identifiers, ambiguous_identifier, identifier_mismatch = record_identifiers(
        transcription
    )
    unit_number, unit_flag, unit_mismatch = normalise_unit(transcription)
    identifier = identifiers[0] if identifiers else ""
    flags = _record_flags(
        transcription,
        identifiers,
        ambiguous_identifier,
        identifier_mismatch,
        unit_flag,
        unit_mismatch,
        additional_flags,
    )
    title = f"{identifier} Unit {unit_number}" if identifier and unit_number else ""
    image = escape_spreadsheet_formula(image_reference(image_file_path, catalogue_path))

    record: dict[str, str | int] = {
        "Node ID": "",
        "Node Type": "Unit",
        "Node Title": title,
        "Description": _description(transcription),
        "Date Range": "",
        "Confidence Score": transcription.confidence,
        "Flags": "; ".join(flags),
        "Image": image,
    }
    return {column: record[column] for column in OUTPUT_COLUMNS}
