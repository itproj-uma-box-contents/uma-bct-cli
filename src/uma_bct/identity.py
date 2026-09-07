"""Conservative, readable evidence checks for box identifiers and units."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

STRONG_RECORD_IDENTIFIER_PATTERN = r"(?:UMA-(?:AC|SR)E?-[0-9]{8,9}|[0-9]{4}\.[0-9]{4})"
LEGACY_RECORD_IDENTIFIER_PATTERN = r"[0-9]{2}/[0-9]+"
RECORD_IDENTIFIER_PATTERN = (
    rf"(?:{STRONG_RECORD_IDENTIFIER_PATTERN}|{LEGACY_RECORD_IDENTIFIER_PATTERN})"
)
RECORD_IDENTIFIER = re.compile(
    rf"^{RECORD_IDENTIFIER_PATTERN}$", re.IGNORECASE | re.ASCII
)
RECORD_IDENTIFIER_TOKEN = re.compile(
    rf"(?<![A-Za-z0-9_/])({RECORD_IDENTIFIER_PATTERN})(?![A-Za-z0-9_/])",
    re.IGNORECASE | re.ASCII,
)
IDENTIFIER_LABEL_PREFIX = re.compile(
    r"^\s*(?:(?:records?\s*(?:set\s*)?(?:identifier|id|number|no)?|"
    r"accession(?:\s*(?:identifier|id|number|no))?|identifier|id|ref(?:erence)?)"
    r"\.?\s*[:#-]?\s*)",
    re.IGNORECASE | re.ASCII,
)
MALFORMED_STRONG_IDENTIFIER = re.compile(
    r"(?<![A-Za-z0-9_])(?:UMA-(?:AC|SR)E?-[A-Za-z0-9-]+|[0-9]{4}\.[0-9]+)"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE | re.ASCII,
)
MALFORMED_LEGACY_IDENTIFIER = re.compile(
    r"(?<![A-Za-z0-9_/])[0-9]{2}/[^\s,;:]+", re.ASCII
)

UNIT_PREFIX = r"(?:unit|box(?:\s*(?:no\.?|number))?|no\.?)"
UNIT_NUMBER = re.compile(
    rf"^(?:{UNIT_PREFIX}\s*[:#-]?\s*)?([0-9]{{1,4}})$",
    re.IGNORECASE | re.ASCII,
)
CONTEXTUAL_UNIT = re.compile(
    rf"(?<!\w)(?:{UNIT_PREFIX})\s*[:#-]?\s*([0-9]{{1,4}})(?!\w)",
    re.IGNORECASE | re.ASCII,
)
HANDWRITTEN_UNIT = re.compile(
    r"^\s*([0-9]{1,4})(?:\s+[A-Z][0-9]{1,2})?\s*$",
    re.IGNORECASE | re.ASCII,
)
AMBIGUOUS_UNIT = re.compile(
    r"\b(?:or|maybe|possibly|unclear|unknown|ambiguous|overwritten|"
    r"crossed\s*out|struck\s*out)\b|\?",
    re.IGNORECASE | re.ASCII,
)
IDENTITY_SEPARATORS = re.compile(r"^[\s,;:./|&+()\[\]{}\"'“”‘’\-–—]*$")


def visual_lines(lines: Iterable[str]) -> Iterable[str]:
    for value in lines:
        yield from value.splitlines()


def normalise_identifier(identifier: str) -> str:
    value = identifier.strip()
    return value.upper() if value.casefold().startswith("uma-") else value


def normalise_unit_claim(value: str) -> str | None:
    match = UNIT_NUMBER.fullmatch(value.strip())
    return match.group(1).zfill(4) if match else None


def _identifier_only_line(line: str) -> bool:
    body = IDENTIFIER_LABEL_PREFIX.sub("", line, count=1)
    if not RECORD_IDENTIFIER_TOKEN.search(body):
        return False
    residue = RECORD_IDENTIFIER_TOKEN.sub(" ", body)
    return IDENTITY_SEPARATORS.fullmatch(residue) is not None


def identifier_evidence_details(
    lines: Iterable[str],
) -> tuple[tuple[str, ...], bool]:
    """Return supported visible identifiers and whether malformed evidence exists."""

    identifiers: list[str] = []
    seen: set[str] = set()
    malformed = False
    for line in visual_lines(lines):
        matches = tuple(RECORD_IDENTIFIER_TOKEN.finditer(line))
        valid_spans = {match.span(1) for match in matches}
        legacy_allowed = _identifier_only_line(line)
        for match in matches:
            identifier = normalise_identifier(match.group(1))
            if "/" in identifier and not legacy_allowed:
                continue
            key = identifier.casefold()
            if key not in seen:
                seen.add(key)
                identifiers.append(identifier)

        malformed |= any(
            match.span() not in valid_spans
            for match in MALFORMED_STRONG_IDENTIFIER.finditer(line)
        )
        if IDENTIFIER_LABEL_PREFIX.match(line):
            malformed |= any(
                match.span() not in valid_spans
                for match in MALFORMED_LEGACY_IDENTIFIER.finditer(line)
            )
    return tuple(identifiers), malformed


def identifier_evidence(lines: Iterable[str]) -> tuple[str, ...]:
    identifiers, _malformed = identifier_evidence_details(lines)
    return identifiers


@dataclass(frozen=True)
class UnitEvidence:
    candidates: frozenset[str]
    ambiguous: bool


def collect_unit_evidence(
    typed_lines: Iterable[str], handwritten_lines: Iterable[str]
) -> UnitEvidence:
    """Collect explicit unit labels and narrowly supported handwritten values."""

    typed = tuple(visual_lines(typed_lines))
    handwritten = tuple(visual_lines(handwritten_lines))
    candidates: set[str] = set()
    ambiguous = False

    for line in (*typed, *handwritten):
        matches = tuple(CONTEXTUAL_UNIT.finditer(line))
        candidates.update(match.group(1).zfill(4) for match in matches)
        if matches and AMBIGUOUS_UNIT.search(line):
            ambiguous = True

    for line in handwritten:
        if match := HANDWRITTEN_UNIT.fullmatch(line):
            candidates.add(match.group(1).zfill(4))

    for index, line in enumerate(typed):
        bare = re.fullmatch(r"\s*([0-9]{1,4})\s*", line)
        if bare and index and identifier_evidence((typed[index - 1],)):
            candidates.add(bare.group(1).zfill(4))

    return UnitEvidence(frozenset(candidates), ambiguous or len(candidates) > 1)


def identity_only_line(
    line: str,
    structured_identifiers: Iterable[str],
    structured_unit: str,
) -> bool:
    """Return whether a description line contains only accepted identity text."""

    identifier_keys = {
        normalise_identifier(value).casefold()
        for value in structured_identifiers
        if RECORD_IDENTIFIER.fullmatch(value.strip())
    }

    def remove_identifier(match: re.Match[str]) -> str:
        value = normalise_identifier(match.group(1)).casefold()
        return " " if value in identifier_keys else match.group(0)

    body = IDENTIFIER_LABEL_PREFIX.sub("", line, count=1)
    body = RECORD_IDENTIFIER_TOKEN.sub(remove_identifier, body)
    normalized_unit = normalise_unit_claim(structured_unit)

    def remove_unit(match: re.Match[str]) -> str:
        return " " if match.group(1).zfill(4) == normalized_unit else match.group(0)

    body = CONTEXTUAL_UNIT.sub(remove_unit, body)
    if normalized_unit and re.fullmatch(r"\s*[0-9]{1,4}\s*", body):
        body = " "
    return IDENTITY_SEPARATORS.fullmatch(body) is not None
