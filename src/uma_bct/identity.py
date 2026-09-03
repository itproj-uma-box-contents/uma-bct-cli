"""Conservative identity evidence shared by transcription and catalogue mapping."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

STRONG_RECORD_IDENTIFIER_PATTERN = r"(?:UMA-(?:AC|SR)E?-\d{8,9}|\d{4}\.\d{4})"
LEGACY_RECORD_IDENTIFIER_PATTERN = r"\d{2}/\d+"
RECORD_IDENTIFIER = re.compile(
    rf"(?<![A-Z0-9/])(?:{STRONG_RECORD_IDENTIFIER_PATTERN}|"
    rf"{LEGACY_RECORD_IDENTIFIER_PATTERN})(?![A-Z0-9/])",
    re.IGNORECASE,
)
UNIT_PREFIX = r"(?:unit|box(?:\s*(?:no\.?|number))?|no\.?)"
UNIT_NUMBER = re.compile(
    rf"(?:{UNIT_PREFIX}\s*[:#-]?\s*)?(\d{{1,4}})",
    re.IGNORECASE,
)
CONTEXTUAL_UNIT = re.compile(
    rf"\b{UNIT_PREFIX}\s*[:#-]?\s*(\d{{1,4}})(?!\d)",
    re.IGNORECASE,
)
IDENTIFIER_LABEL_PREFIX = re.compile(
    r"^\s*(?:(?:records?(?:\s+set)?|accession)"
    r"(?:\s+(?:identifier|id|number|no\.?))?|identifier|id)"
    r"\s*[:#-]?\s*",
    re.IGNORECASE,
)
HANDWRITTEN_UNIT = re.compile(r"^\s*(\d{1,4})(?:\s+[A-Z]\d{1,2})?\s*$", re.I)
AMBIGUOUS_UNIT_CONNECTOR = re.compile(
    rf"\b{UNIT_PREFIX}\s*[:#-]?\s*\d{{1,4}}\s*"
    rf"(?:\?|(?:and\s*/\s*)?or\b|\band\b|\bwas\b|&|\+|/|,|\bto\b|[-–—])\s*"
    rf"(?:{UNIT_PREFIX}\s*[:#-]?\s*)?\d{{1,4}}\b",
    re.IGNORECASE,
)
AMBIGUOUS_UNIT_WORD = re.compile(
    r"\b(?:or|was|maybe|possibly|unclear|unknown|ambiguous|overwritten|"
    r"crossed\s*out|struck\s*out)\b|\band\s*/\s*or\b",
    re.IGNORECASE,
)
IDENTITY_SEPARATORS = re.compile(r"^[\s,;:/|&+()\[\]{}\-–—]*$")


def visual_lines(lines: Iterable[str]) -> Iterable[str]:
    for value in lines:
        yield from value.splitlines()


def normalise_identifier(identifier: str) -> str:
    return (
        identifier.upper() if identifier.casefold().startswith("uma-") else identifier
    )


def normalise_unit_claim(value: str) -> str | None:
    match = UNIT_NUMBER.fullmatch(value.strip())
    if match is None:
        return None
    return match.group(1).zfill(4)


def identifier_list_line(line: str) -> bool:
    """Recognise a line whose semantic content is only record identifiers."""

    body = IDENTIFIER_LABEL_PREFIX.sub("", line, count=1)
    matches = tuple(RECORD_IDENTIFIER.finditer(body))
    if not matches:
        return False
    residue = RECORD_IDENTIFIER.sub(" ", body)
    residue = CONTEXTUAL_UNIT.sub(" ", residue)
    residue = re.sub(r"\band\b", " ", residue, flags=re.IGNORECASE)
    return IDENTITY_SEPARATORS.fullmatch(residue) is not None


def identifier_evidence(lines: Iterable[str]) -> tuple[str, ...]:
    """Return supported raw identifiers, suppressing slash-shaped dates in prose."""

    identifiers: list[str] = []
    seen: set[str] = set()
    for line in visual_lines(lines):
        legacy_is_identity = identifier_list_line(line)
        for match in RECORD_IDENTIFIER.finditer(line):
            identifier = normalise_identifier(match.group(0))
            if "/" in identifier and not legacy_is_identity:
                continue
            key = identifier.casefold()
            if key not in seen:
                seen.add(key)
                identifiers.append(identifier)
    return tuple(identifiers)


@dataclass(frozen=True)
class UnitEvidence:
    candidates: frozenset[str]
    ambiguous: bool


def collect_unit_evidence(typed_lines: Iterable[str], handwritten_lines: Iterable[str]) -> UnitEvidence:
    """
    Collect narrowly supported raw box-number evidence and ambiguity signals.
    """

    candidates: set[str] = set()
    ambiguous = False
    typed = tuple(visual_lines(typed_lines))
    handwritten = tuple(visual_lines(handwritten_lines))
    for line in (*typed, *handwritten):
        contextual_matches = tuple(CONTEXTUAL_UNIT.finditer(line))
        candidates.update(match.group(1).zfill(4) for match in contextual_matches)
        if contextual_matches and (
            AMBIGUOUS_UNIT_CONNECTOR.search(line)
            or "?" in line
            or AMBIGUOUS_UNIT_WORD.search(line)
        ):
            ambiguous = True
    # Some historic printed labels put a bare box number on the visual line
    # immediately after the record identifier. This adjacency is required;
    # arbitrary standalone typed numbers remain unsafe as unit evidence.
    for index, line in enumerate(typed):
        bare_typed_unit = re.fullmatch(r"\s*(\d{1,4})\s*", line)
        if (
            bare_typed_unit is not None
            and index > 0
            and identifier_evidence((typed[index - 1],))
        ):
            candidates.add(bare_typed_unit.group(1).zfill(4))
    for line in handwritten:
        if match := HANDWRITTEN_UNIT.fullmatch(line):
            candidates.add(match.group(1).zfill(4))
    if len(candidates) > 1:
        ambiguous = True
    return UnitEvidence(frozenset(candidates), ambiguous)


def unit_claim_is_corroborated(value: str, typed_lines: Iterable[str], handwritten_lines: Iterable[str]) -> bool:
    normalized = normalise_unit_claim(value)
    if normalized is None:
        return False
    evidence = collect_unit_evidence(typed_lines, handwritten_lines)
    return not evidence.ambiguous and evidence.candidates == {normalized}


def identity_only_line(line: str, structured_identifiers: Iterable[str],
    structured_unit: str) -> bool:
    """
    Return whether a raw line contains identity syntax and no annotation text.
    """

    body = line.strip()
    if not body:
        return False

    structured_keys = {
        normalise_identifier(identifier.strip()).casefold()
        for identifier in structured_identifiers
        if RECORD_IDENTIFIER.fullmatch(identifier.strip())
    }

    removed_identifier = False

    def remove_identifier(match: re.Match[str]) -> str:
        nonlocal removed_identifier
        identifier = normalise_identifier(match.group(0))
        if "/" not in identifier or identifier.casefold() in structured_keys:
            removed_identifier = True
            return " "
        return match.group(0)

    body = RECORD_IDENTIFIER.sub(remove_identifier, body)
    if removed_identifier:
        body = IDENTIFIER_LABEL_PREFIX.sub("", body, count=1)
    body = re.sub(r"\band\b", " ", body, flags=re.IGNORECASE)
    body = body.strip(" \t,;:/|&+()[]{}-–—")
    if not body and removed_identifier:
        return True

    if match := re.match(rf"^\s*{UNIT_PREFIX}\s*[:#-]?\s*", body, re.I):
        unit_body = body[match.end() :]
        unit_body = re.sub(
            r"\b(?:and\s*/\s*or|or|to|maybe|possibly|unclear|unknown|"
            r"ambiguous|overwritten|crossed\s*out|struck\s*out)\b",
            " ",
            unit_body,
            flags=re.IGNORECASE,
        )
        unit_body = re.sub(r"\d{1,4}", " ", unit_body)
        return IDENTITY_SEPARATORS.fullmatch(unit_body) is not None

    normalized_claim = normalise_unit_claim(structured_unit)
    bare_unit = re.fullmatch(r"\d{1,4}", body)
    return bool(
        normalized_claim
        and bare_unit
        and bare_unit.group(0).zfill(4) == normalized_claim
    )
