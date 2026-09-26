"""Map visible label text into the client's Unit Type and Unit Number columns.

Printed text stays available for review even when an identifier is uncertain.
The unit number is exported only when its structured claim matches the label.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

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
    unit_type_text,
)

SPREADSHEET_FORMULA_PREFIX = re.compile(r"^\s*[=+\-@]")


class CatalogueRow(dict[str, str | int]):
    """A row with read-only lookup aliases for the v0.1 draft headings."""

    LOOKUP_ALIASES = {
        "Review Notes": "Review Flags",
        "Crop Image": "Cropped Box Image Link",
    }

    def __missing__(self, key: str) -> str | int:
        alias = self.LOOKUP_ALIASES.get(key)
        if alias is None:
            raise KeyError(key)
        return self[alias]


def escape_spreadsheet_formula(value: str) -> str:
    """Stop spreadsheet apps from running transcribed text as a formula."""

    return f"'{value}" if SPREADSHEET_FORMULA_PREFIX.match(value) else value


def append_flag(current: str, flag: str) -> str:
    """Append one review flag without repeating it."""

    flags = [item.strip() for item in current.split(";") if item.strip()]
    if flag not in flags:
        flags.append(flag)
    return "; ".join(flags)


def non_empty_lines(lines: Iterable[str]) -> tuple[str, ...]:
    return tuple(visual_line.strip() for value in lines for visual_line in value.splitlines() if visual_line.strip())


def structured_identifiers(candidates: Iterable[str]) -> tuple[tuple[str, ...], bool]:
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


def raw_identifier_mismatch(transcription: BoxTranscription, explicit_identifiers: tuple[str, ...]) -> bool:
    """Require the model's identifier list to agree with the visible text and its order."""

    typed_identifiers, malformed_typed = identifier_evidence_details(transcription.typed_lines)
    handwritten_identifiers, malformed_handwritten = identifier_evidence_details(transcription.handwritten_lines)
    observed_identifiers = (*typed_identifiers, *handwritten_identifiers)
    observed_keys = {identifier.casefold() for identifier in observed_identifiers}
    explicit_keys = tuple(identifier.casefold() for identifier in explicit_identifiers)
    explicit_key_set = set(explicit_keys)
    if observed_keys != explicit_key_set or malformed_typed or malformed_handwritten:
        return True

    # Typed and handwritten lines can each list several IDs. Compare their
    # order separately so the first structured ID still means the first seen.
    for source in (typed_identifiers, handwritten_identifiers):
        positions = [
            explicit_keys.index(identifier.casefold())
            for identifier in source
            if identifier.casefold() in explicit_key_set
        ]
        if positions != sorted(positions):
            return True
    return False


def record_identifiers(transcription: BoxTranscription) -> tuple[tuple[str, ...], bool, bool]:
    """Return IDs that match the visible text and say if review is needed."""

    if not transcription.record_identifiers:
        unexpected, malformed = identifier_evidence_details((*transcription.typed_lines, *transcription.handwritten_lines))
        ambiguous = bool(unexpected) or malformed
        return (), ambiguous, ambiguous

    identifiers, invalid = structured_identifiers(transcription.record_identifiers)
    if invalid or not identifiers or raw_identifier_mismatch(transcription, identifiers):
        return (), True, True
    return identifiers, False, False


def normalise_unit(transcription: BoxTranscription) -> tuple[str, str | None, bool]:
    raw_unit = transcription.unit_number.strip()
    if not raw_unit:
        return "", MISSING_UNIT_FLAG, False
    normalized = normalise_unit_claim(raw_unit)
    if normalized is None:
        return "", INVALID_UNIT_FLAG, False
    evidence = collect_unit_evidence(transcription.typed_lines, transcription.handwritten_lines)
    if evidence.ambiguous or evidence.candidates != {normalized}:
        return "", INVALID_UNIT_FLAG, True
    return normalized, None, False


def image_reference(image_file_path: str | Path, catalogue_path: str | Path | None) -> str:
    image_path = Path(image_file_path)
    if catalogue_path is None:
        return image_path.as_posix()

    destination_parent = Path(catalogue_path).parent
    try:
        absolute_image = image_path if image_path.is_absolute() else image_path.resolve()
        absolute_parent = destination_parent if destination_parent.is_absolute() else destination_parent.resolve()
        return Path(os.path.relpath(absolute_image, absolute_parent)).as_posix()
    except ValueError:
        # Windows cannot represent a relative path across drive letters.
        raise ValueError(
            "The catalogue and review-crop directory must be on the same drive "
            "so cropped-image link values remain relative. Choose --path-to-box-images on "
            "the catalogue drive."
        ) from None


def trusted_crop_hyperlink(image_file_path: str | Path, catalogue_path: str | Path) -> str:
    """Build a formula only for a validated, managed ``Box Images`` path."""

    reference = image_reference(image_file_path, catalogue_path)
    relative = PurePosixPath(reference)
    if relative.is_absolute() or len(relative.parts) != 2 or relative.parts[0] != "Box Images":
        raise ValueError("Managed crop links must point to one file inside Box Images.")
    if ".." in relative.parts or any(not part or any(ord(character) < 32 for character in part) for part in relative.parts):
        raise ValueError("Managed crop link contains an unsafe path component.")
    target = reference.replace('"', '""')
    label = relative.name.replace('"', '""')
    return f'=HYPERLINK("{target}","{label}")'


def description(transcription: BoxTranscription) -> str:
    values = (*non_empty_lines(transcription.handwritten_lines), *non_empty_lines(transcription.auxiliary_annotations))
    visible = (
        line for line in values if not identity_only_line(line, transcription.record_identifiers, transcription.unit_number)
    )
    return escape_spreadsheet_formula("; ".join(visible))


def verification_flags(audit: dict[str, object] | None) -> tuple[str, ...]:
    flags = []
    if audit:
        if audit.get("error"):
            flags.append("Verification failed")
        if audit.get("disputed"):
            flags.append("Verification discrepancy")
        if audit.get("unresolved"):
            flags.append("Verification unresolved")
    return tuple(flags)


def map_catalogue_record(transcription: BoxTranscription, image_file_path: str | Path, *,
                         catalogue_path: str | Path | None = None, additional_flags: Iterable[str] = (),
                         trusted_crop_link: bool = False, spreadsheet_safe: bool = True) -> dict[str, str | int]:
    """Keep printed text visible and flag identity claims that need review."""

    identifiers, ambiguous_identifier, identifier_mismatch = record_identifiers(transcription)
    unit_number, unit_flag, unit_mismatch = normalise_unit(transcription)
    unit_type = unit_type_text(transcription.typed_lines)
    flags: list[str] = []
    if not unit_type:
        flags.append(AMBIGUOUS_IDENTIFIER_FLAG if ambiguous_identifier else MISSING_IDENTIFIER_FLAG)
    elif ambiguous_identifier:
        flags.append(AMBIGUOUS_IDENTIFIER_FLAG)
    elif len(identifiers) > 1:
        flags.append(MULTIPLE_IDENTIFIERS_FLAG)
    if unit_flag:
        flags.append(unit_flag)
    if transcription.needs_review or identifier_mismatch or unit_mismatch:
        flags.append(TRANSCRIPTION_MISMATCH_FLAG)
    flags.extend(flag for flag in additional_flags if flag not in flags)

    # Formula links are reserved for validated managed crops. Other paths are
    # exported as escaped text because filenames may be untrusted input.
    if trusted_crop_link:
        if catalogue_path is None:
            raise ValueError("A catalogue path is required for a trusted crop link.")
        image = trusted_crop_hyperlink(image_file_path, catalogue_path)
    else:
        image = escape_spreadsheet_formula(image_reference(image_file_path, catalogue_path))

    record: dict[str, str | int] = {
        "Unit Type": escape_spreadsheet_formula(unit_type),
        "Unit Number": unit_number,
        "Description": description(transcription),
        "Date Range": "",
        "Transcription Confidence (%)": transcription.confidence,
        "Review Flags": "; ".join(flags),
        "Cropped Box Image Link": image,
    }
    if not spreadsheet_safe:
        # HTML escapes at its own boundary; do not display CSV's protective apostrophe.
        record["Unit Type"] = unit_type
        record["Description"] = "; ".join(
            line for line in (*non_empty_lines(transcription.handwritten_lines),
                              *non_empty_lines(transcription.auxiliary_annotations))
            if not identity_only_line(line, transcription.record_identifiers, transcription.unit_number)
        )
        record["Cropped Box Image Link"] = image_reference(image_file_path, catalogue_path)
    return CatalogueRow({column: record[column] for column in OUTPUT_COLUMNS})
