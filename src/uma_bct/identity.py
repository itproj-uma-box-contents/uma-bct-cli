"""Check visible identifier and unit-number evidence without assuming a prefix format."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

# Accept a one-to-257-character non-whitespace identifier without leading or trailing whitespace.
RECORD_IDENTIFIER = re.compile(r"^[^\s](?:.{0,255}\S)?$", re.IGNORECASE | re.ASCII)
# Find the supported structured IDs: YYYY.NNNN, YY/N, and UMA-(AC|SR)E?-########.
KNOWN_IDENTIFIER_TOKEN = re.compile(
    r"(?<![A-Za-z0-9_/])((?:UMA-(?:AC|SR)E?-[0-9]{8,9}|[0-9]{4}\.[0-9]{4}|[0-9]{2}/[0-9]+))(?![A-Za-z0-9_/])",
    re.IGNORECASE | re.ASCII,
)
# Find a dotted value shaped like YYYY.NNNN so wrong digit counts cannot fall
# through to the arbitrary-prefix path (for example, 1960.009).
DOTTED_IDENTIFIER_LOOKALIKE = re.compile(
    r"(?<![A-Za-z0-9_/])([0-9]{4}\.[0-9]+)(?![A-Za-z0-9_/])",
    re.IGNORECASE | re.ASCII,
)
# Remove common label prefixes such as "Record ID:", "Accession Number:", or "Reference:".
IDENTIFIER_LABEL_PREFIX = re.compile(
    r"^\s*(?:(?:records?\s*(?:set\s*)?(?:identifier|id|number|no)?|"
    r"accession(?:\s*(?:identifier|id|number|no))?|identifier|id|ref(?:erence)?)"
    r"\.?\s*[:#-]?\s*)",
    re.IGNORECASE | re.ASCII,
)
# Match the labels that can introduce a unit number, including Unit, Box, Box No., and No.
UNIT_PREFIX = r"(?:unit|box(?:\s*(?:no\.?|number))?|no\.?)"
# Match a complete unit label or a bare one-to-four-digit unit number.
UNIT_NUMBER = re.compile(rf"^(?:{UNIT_PREFIX}\s*[:#-]?\s*)?([0-9]{{1,4}})$", re.IGNORECASE | re.ASCII)
# Find a unit label embedded in a transcription line and capture its number.
CONTEXTUAL_UNIT = re.compile(rf"(?<!\w)(?:{UNIT_PREFIX})\s*[:#-]?\s*([0-9]{{1,4}})(?!\w)", re.IGNORECASE | re.ASCII)
# A range, decimal or list cannot corroborate just its first number. "of 9"
# is deliberately absent: that phrase is a sequence total, not another unit.
COMPOUND_UNIT_SUFFIX = re.compile(
    rf"^\s*(?:[/.,&+\-–—]|\b(?:and|or|to)\b)\s*(?:{UNIT_PREFIX}\s*[:#-]?\s*)?(?P<number>[0-9])",
    re.IGNORECASE | re.ASCII,
)
# Match the complete printed unit-number phrase so it can be omitted from Unit Type.
# A sequence total such as "of 9" describes the photographed run, not the unit type.
PRINTED_UNIT_TEXT = re.compile(
    rf"(?<!\w)(?:{UNIT_PREFIX})\s*[:#-]?\s*[0-9]{{1,4}}(?!\w)(?:\s+of\s+[0-9]{{1,4}}(?!\w))?",
    re.IGNORECASE | re.ASCII,
)
# Match a line containing an arbitrary identifier prefix followed by a unit label and number.
PREFIXED_UNIT = re.compile(
    rf"^\s*(?P<prefix>.+?)\s+(?:{UNIT_PREFIX})\s*[:#-]?\s*(?P<number>[0-9]{{1,4}})\s*$", re.IGNORECASE | re.ASCII
)
# Match a handwritten bare unit number, optionally followed by a short letter/code suffix.
HANDWRITTEN_UNIT = re.compile(r"^\s*([0-9]{1,4})(?:\s+[A-Z][0-9]{1,2})?\s*$", re.IGNORECASE | re.ASCII)
# Detect wording or punctuation that makes a unit-number claim ambiguous.
AMBIGUOUS_UNIT = re.compile(
    r"\b(?:or|maybe|possibly|unclear|unknown|ambiguous|overwritten|"
    r"crossed\s*out|struck\s*out)\b|\?",
    re.IGNORECASE | re.ASCII,
)
# Match lines containing only separators around an identifier or unit number.
IDENTITY_SEPARATORS = re.compile(r"^[\s,;:./|&+()\[\]{}\"'“”‘’\-–—]*$")


def visual_lines(lines: Iterable[str]) -> Iterable[str]:
    """Yield each supplied line, including embedded line breaks."""

    for value in lines:
        yield from value.splitlines()


def normalise_identifier(identifier: str) -> str:
    """Trim an identifier while preserving its spelling."""

    value = identifier.strip()
    return value.upper() if value.casefold().startswith("uma-") else value


def normalise_unit_claim(value: str) -> str | None:
    """Return a four-digit unit number when the value is valid."""

    match = UNIT_NUMBER.fullmatch(value.strip())
    return match.group(1).zfill(4) if match else None


def prefixed_unit(line: str) -> tuple[str, str] | None:
    """Extract any prefix and unit number from one label line."""

    match = PREFIXED_UNIT.fullmatch(line)
    if match is None:
        return None
    prefix = " ".join(match.group("prefix").split())
    return prefix, match.group("number").zfill(4)


def standalone_identifier(line: str) -> str | None:
    """Accept a standalone prefix only when followed by a unit line."""

    value = " ".join(line.split())
    if (
        not value
        or not RECORD_IDENTIFIER.fullmatch(value)
        or UNIT_NUMBER.fullmatch(value)
        or IDENTIFIER_LABEL_PREFIX.match(value)
        or value.casefold() in {"record", "origin", "unit", "box", "number", "identifier"}
    ):
        return None
    return value


def identifier_only_line(line: str) -> bool:
    """Check whether a line contains only identifier-like text."""

    body = IDENTIFIER_LABEL_PREFIX.sub("", line, count=1)
    if prefixed_unit(line) is not None:
        return True
    if not KNOWN_IDENTIFIER_TOKEN.search(body):
        return False
    residue = KNOWN_IDENTIFIER_TOKEN.sub(" ", body)
    return IDENTITY_SEPARATORS.fullmatch(residue) is not None


def identifier_evidence_details(lines: Iterable[str]) -> tuple[tuple[str, ...], bool]:
    """Return visible IDs and say if any ID-like text is wrong."""

    identifiers: list[str] = []
    seen: set[str] = set()
    malformed = False
    visible_lines = tuple(visual_lines(lines))
    for index, line in enumerate(visible_lines):
        # Known formats remain supported, but arbitrary prefixes are accepted
        # when the label clearly pairs them with a unit number.
        matches = tuple(KNOWN_IDENTIFIER_TOKEN.finditer(line))
        malformed_dotted = any(
            KNOWN_IDENTIFIER_TOKEN.fullmatch(match.group(1)) is None for match in DOTTED_IDENTIFIER_LOOKALIKE.finditer(line)
        )
        # A dotted ID with the wrong digit count stays visible in Unit Type,
        # but it cannot corroborate a structured identifier without review.
        malformed = malformed or malformed_dotted
        legacy_allowed = identifier_only_line(line)
        for match in matches:
            identifier = normalise_identifier(match.group(1))
            if "/" in identifier and not legacy_allowed:
                continue
            key = identifier.casefold()
            if key not in seen:
                seen.add(key)
                identifiers.append(identifier)

        if (prefixed := prefixed_unit(line)) and not malformed_dotted:
            identifier = normalise_identifier(prefixed[0])
            if identifier.casefold() not in {value.casefold() for value in identifiers}:
                identifiers.append(identifier)
        elif (
            not matches
            and not malformed_dotted
            and index + 1 < len(visible_lines)
            and CONTEXTUAL_UNIT.search(visible_lines[index + 1])
            # An identifier already paired with Unit on the next line needs
            # no preceding prefix. Otherwise an origin heading is misread as
            # another identifier, causing a false mismatch.
            and prefixed_unit(visible_lines[index + 1]) is None
        ):
            if identifier := standalone_identifier(line):
                identifier = normalise_identifier(identifier)
                if identifier.casefold() not in {value.casefold() for value in identifiers}:
                    identifiers.append(identifier)
    return tuple(identifiers), malformed


def identifier_evidence(lines: Iterable[str]) -> tuple[str, ...]:
    """Return identifiers found in visible transcription lines."""

    return identifier_evidence_details(lines)[0]


def unit_type_text(lines: Iterable[str]) -> str:
    """Return all printed label text except complete unit-number phrases."""

    parts: list[str] = []
    # Preserve headings and imperfect IDs exactly as printed. Only an explicit
    # Unit or Box phrase is removed before visual lines become one CSV cell.
    for line in visual_lines(lines):
        without_unit = line if has_compound_unit(line) else PRINTED_UNIT_TEXT.sub(" ", line)
        value = " ".join(without_unit.split())
        if value and IDENTITY_SEPARATORS.fullmatch(value) is None:
            parts.append(value)
    return " ".join(parts)


def has_compound_unit(line: str) -> bool:
    """Recognize a labelled unit whose following text adds another number."""

    for match in CONTEXTUAL_UNIT.finditer(line):
        suffix = COMPOUND_UNIT_SUFFIX.match(line[match.end():])
        if suffix is None:
            continue
        # Label order varies: "Unit 7 - 1960.0009" still names one box.
        # Exempt complete known identifiers, never a partial or malformed ID.
        following_number = match.end() + suffix.start("number")
        if KNOWN_IDENTIFIER_TOKEN.match(line, following_number) is None:
            return True
    return False


@dataclass(frozen=True)
class UnitEvidence:
    """Collected unit candidates and whether they conflict."""

    candidates: frozenset[str]
    ambiguous: bool


def collect_unit_evidence(typed_lines: Iterable[str], handwritten_lines: Iterable[str]) -> UnitEvidence:
    """Collect unit numbers from printed and handwritten text."""

    typed = tuple(visual_lines(typed_lines))
    handwritten = tuple(visual_lines(handwritten_lines))
    candidates: set[str] = set()
    ambiguous = False

    for line in (*typed, *handwritten):
        matches = tuple(CONTEXTUAL_UNIT.finditer(line))
        candidates.update(match.group(1).zfill(4) for match in matches)
        if matches and (AMBIGUOUS_UNIT.search(line) or has_compound_unit(line)):
            ambiguous = True

    for line in handwritten:
        if match := HANDWRITTEN_UNIT.fullmatch(line):
            candidates.add(match.group(1).zfill(4))

    for index, line in enumerate(typed):
        bare = re.fullmatch(r"\s*([0-9]{1,4})\s*", line)
        if bare and index and identifier_evidence((typed[index - 1],)):
            candidates.add(bare.group(1).zfill(4))

    return UnitEvidence(frozenset(candidates), ambiguous or len(candidates) > 1)


def resolve_unit_claim(
    structured_unit: str, typed_lines: Iterable[str], handwritten_lines: Iterable[str]
) -> tuple[str, bool]:
    """Fill one omitted unit from exact raw text and say if review is needed.

    An explicit structured value is never replaced. A missing value can be
    recovered only from an explicit printed Unit or Box label, when every
    recognised raw occurrence names the same unit and none of the surrounding
    text marks it as uncertain or crossed out. Bare or handwritten numbers are
    too easy to misinterpret without looking at the image, so they stay blank.
    The returned digits preserve the visible spelling instead of inventing
    padding that was not present in the transcription.
    """

    if structured_unit.strip():
        return structured_unit, False

    typed = tuple(visual_lines(typed_lines))
    handwritten = tuple(visual_lines(handwritten_lines))
    evidence = collect_unit_evidence(typed, handwritten)
    if not evidence.candidates:
        return "", False
    if evidence.ambiguous:
        return "", True

    normalized = next(iter(evidence.candidates))
    for line in typed:
        for match in CONTEXTUAL_UNIT.finditer(line):
            if match.group(1).zfill(4) == normalized:
                return match.group(1), True

    # Other evidence can still force review, but it is not promoted without an
    # explicit printed label.
    return "", True


def identity_only_line(line: str, structured_identifiers: Iterable[str], structured_unit: str) -> bool:
    """Check whether a line contains only the box ID or unit number."""

    identifier_keys = {
        normalise_identifier(value).casefold()
        for value in structured_identifiers
        if RECORD_IDENTIFIER.fullmatch(value.strip())
    }

    def remove_identifier(match: re.Match[str]) -> str:
        value = normalise_identifier(match.group(1)).casefold()
        return " " if value in identifier_keys else match.group(0)

    normalized_unit = normalise_unit_claim(structured_unit)

    def remove_unit(match: re.Match[str]) -> str:
        return " " if match.group(1).zfill(4) == normalized_unit else match.group(0)

    # Remove known identity text while leaving any additional annotation for
    # Description. A whole prefix-plus-unit line is identity-only as well.
    body = IDENTIFIER_LABEL_PREFIX.sub("", line, count=1)
    body = " " if body.strip().casefold() in identifier_keys else KNOWN_IDENTIFIER_TOKEN.sub(remove_identifier, body)
    if prefixed := prefixed_unit(line):
        body = line.replace(prefixed[0], " ", 1)
        body = CONTEXTUAL_UNIT.sub(remove_unit, body)
        return IDENTITY_SEPARATORS.fullmatch(body) is not None

    body = CONTEXTUAL_UNIT.sub(remove_unit, body)
    if normalized_unit and re.fullmatch(r"\s*[0-9]{1,4}\s*", body):
        body = " "
    return IDENTITY_SEPARATORS.fullmatch(body) is not None
