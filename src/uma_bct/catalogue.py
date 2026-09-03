"""Map validated box transcriptions into the client catalogue schema."""

from __future__ import annotations

import os
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
    identifier_evidence,
    identity_only_line,
    normalise_identifier,
    normalise_unit_claim,
)
from .validation import escape_spreadsheet_formula


def non_empty_lines(lines: Iterable[str]) -> tuple[str, ...]:
    return tuple(
        visual_line.strip()
        for value in lines
        for visual_line in value.splitlines()
        if visual_line.strip()
    )


def record_identifiers(
    transcription: BoxTranscription,
) -> tuple[tuple[str, ...], bool, bool]:
    """Return corroborated identifiers, ambiguity evidence, and mismatch state."""

    label = transcription.label
    explicit_candidates = label.record_identifiers if label else ()
    if not explicit_candidates:
        # YY/N identifiers are deliberately supported, but the same shape is also
        # common in dates such as "Meeting 12/5".  Never promote arbitrary raw
        # text into identity when the model did not classify it as an identifier.
        return (), False, False

    explicit_identifiers: list[str] = []
    explicit_seen: set[str] = set()
    invalid_candidate = False
    for candidate in explicit_candidates:
        match = RECORD_IDENTIFIER.fullmatch(candidate.strip())
        if match is None:
            invalid_candidate = True
            continue
        identifier = normalise_identifier(match.group(0))
        key = identifier.casefold()
        if key not in explicit_seen:
            explicit_seen.add(key)
            explicit_identifiers.append(identifier)
    if invalid_candidate or not explicit_identifiers:
        return (), True, True

    typed_identifiers = identifier_evidence(transcription.typed_lines)
    handwritten_identifiers = identifier_evidence(transcription.handwritten_lines)
    observed_identifiers = (*typed_identifiers, *handwritten_identifiers)
    observed_keys = {identifier.casefold() for identifier in observed_identifiers}
    explicit_keys = tuple(identifier.casefold() for identifier in explicit_identifiers)
    explicit_key_set = set(explicit_keys)

    # The model supplies global reading order. Typed and handwritten arrays each
    # preserve their own order, but cannot represent how those two modes interleave.
    # Validate order only within each source array and trust the structured list
    # for the otherwise-unrecoverable cross-mode order.
    order_mismatch = False
    for source in (typed_identifiers, handwritten_identifiers):
        positions = [
            explicit_keys.index(identifier.casefold())
            for identifier in source
            if identifier.casefold() in explicit_key_set
        ]
        if positions != sorted(positions):
            order_mismatch = True

    missing_from_raw = any(key not in observed_keys for key in explicit_keys)
    unexpected_raw = any(
        identifier.casefold() not in explicit_key_set
        for identifier in observed_identifiers
    )
    mismatch = missing_from_raw or unexpected_raw or order_mismatch
    if mismatch:
        return (), True, True
    return tuple(explicit_identifiers), False, False


def normalise_unit(transcription: BoxTranscription) -> tuple[str, str | None, bool]:
    raw_unit = transcription.label.unit_number.strip() if transcription.label else ""
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
    identifier = identifiers[0] if identifiers else ""
    unit_number, unit_flag, unit_mismatch = normalise_unit(transcription)

    flags: list[str] = []
    if not identifier:
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

    title = f"{identifier} Unit {unit_number}" if identifier and unit_number else ""
    structured_identifiers = (
        transcription.label.record_identifiers if transcription.label else ()
    )
    structured_unit = transcription.label.unit_number if transcription.label else ""
    handwritten_description = tuple(
        line
        for line in non_empty_lines(transcription.handwritten_lines)
        if not identity_only_line(line, structured_identifiers, structured_unit)
    )
    auxiliary_description = tuple(
        line
        for line in non_empty_lines(transcription.auxiliary_annotations)
        if not identity_only_line(line, structured_identifiers, structured_unit)
    )
    description_lines = (
        *handwritten_description,
        *auxiliary_description,
    )
    description = escape_spreadsheet_formula("; ".join(description_lines))
    image = escape_spreadsheet_formula(image_reference(image_file_path, catalogue_path))

    record: dict[str, str | int] = {
        "Node ID": "",
        "Node Type": "Unit",
        "Node Title": title,
        "Description": description,
        "Date Range": "",
        "Confidence Score": transcription.confidence,
        "Flags": "; ".join(flags),
        "Image": image,
    }
    return {column: record[column] for column in OUTPUT_COLUMNS}
