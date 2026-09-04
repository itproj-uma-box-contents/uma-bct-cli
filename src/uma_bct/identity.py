"""Conservative identity evidence shared by transcription and catalogue mapping."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date
from typing import Literal

STRONG_RECORD_IDENTIFIER_PATTERN = r"(?:UMA-(?:AC|SR)E?-[0-9]{8,9}|[0-9]{4}\.[0-9]{4})"
LEGACY_RECORD_IDENTIFIER_PATTERN = r"[0-9]{2}/[0-9]+"
RECORD_IDENTIFIER_BODY = (
    rf"(?a:(?:{STRONG_RECORD_IDENTIFIER_PATTERN}|{LEGACY_RECORD_IDENTIFIER_PATTERN}))"
)
UNIT_TOKEN_END = r"(?=$|[\s,;:!?)}\]]|\.(?=$|\s))"
RECORD_IDENTIFIER = re.compile(RECORD_IDENTIFIER_BODY, re.IGNORECASE)
_RECORD_IDENTIFIER_CANDIDATE = re.compile(
    rf"(?<![\w/])(?<!\d\.){RECORD_IDENTIFIER_BODY}",
    re.IGNORECASE,
)
_CLEAR_TOKEN_SEPARATORS = frozenset(",;:&+|")
_TERMINAL_TOKEN_PUNCTUATION = frozenset(".!?)}]\"'’”")
_DASH_IDENTIFIER_SEPARATORS = frozenset("-–—")
_OPENING_TOKEN_WRAPPERS = frozenset("\"'“‘([{")
_OPENING_TOKEN_WRAPPER_PATTERN = r"[\"'“‘(\[{]"
UNIT_PREFIX = r"(?:unit|box(?:\s*(?:no\.?|number))?|no\.?)"
UNIT_NUMBER = re.compile(
    rf"(?:{UNIT_PREFIX}\s*[:#-]?\s*)?([0-9]{{1,4}})",
    re.IGNORECASE | re.ASCII,
)
CONTEXTUAL_UNIT = re.compile(
    rf"(?<!\w)(?a:{UNIT_PREFIX})\s*[:#-]?\s*([0-9]{{1,4}})"
    rf"{UNIT_TOKEN_END}",
    re.IGNORECASE,
)
IDENTIFIER_LABEL_PREFIX = re.compile(
    rf"^\s*(?:(?:records?\b(?:\s+set\b)?|accession\b)"
    r"(?:\s+(?:identifier\b\.?|id\b\.?|number\b\.?|no\b\.?))?|"
    r"identifier\b\.?|id\b\.?|ref(?:erence)?\b\.?)"
    rf"(?:\s*[:#-]\s*|\s+)?"
    rf"(?=(?:{_OPENING_TOKEN_WRAPPER_PATTERN}\s*)*{RECORD_IDENTIFIER_BODY})",
    re.IGNORECASE | re.ASCII,
)
HANDWRITTEN_UNIT = re.compile(
    r"^\s*([0-9]{1,4})(?:\s+[A-Z][0-9]{1,2})?\s*$", re.I | re.ASCII
)
AMBIGUOUS_UNIT_CONNECTOR = re.compile(
    rf"\b{UNIT_PREFIX}\s*[:#-]?\s*[0-9]{{1,4}}\s*"
    rf"(?:\?|(?:and\s*/\s*)?or\b|\band\b|\bwas\b|&|\+|/|,|\bto\b|[-–—])\s*"
    rf"(?:{UNIT_PREFIX}\s*[:#-]?\s*)?[0-9]{{1,4}}\b",
    re.IGNORECASE | re.ASCII,
)
AMBIGUOUS_UNIT_WORD = re.compile(
    r"\b(?:or|was|maybe|possibly|unclear|unknown|ambiguous|overwritten|"
    r"crossed\s*out|struck\s*out)\b|\band\s*/\s*or\b",
    re.IGNORECASE | re.ASCII,
)
IDENTITY_SEPARATORS = re.compile(r"^[\s,;:./|&+()\[\]{}\"'“”‘’\-–—]*$")
CALENDAR_DATE = re.compile(
    r"(?<![0-9/])(?P<day>0?[1-9]|[12][0-9]|3[01])/"
    r"(?P<month>0?[1-9]|1[0-2])/"
    r"(?P<year>[0-9]{4}|[0-9]{2})",
    re.ASCII,
)
_TokenTermination = Literal["boundary", "dash", "invalid"]


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


def _skip_whitespace(value: str, start: int) -> int:
    index = start
    while index < len(value) and value[index].isspace():
        index += 1
    return index


def _skip_opening_token_wrappers(value: str, start: int) -> int:
    index = _skip_whitespace(value, start)
    while index < len(value) and value[index] in _OPENING_TOKEN_WRAPPERS:
        index = _skip_whitespace(value, index + 1)
    return index


def _terminal_punctuation_end(value: str, start: int) -> int:
    """Return the index after one sequence of closing punctuation."""

    index = start
    while index < len(value) and value[index] in _TERMINAL_TOKEN_PUNCTUATION:
        index += 1
    return index


def _token_termination(
    value: str,
    end: int,
    *,
    spaced_dash_is_chain: bool,
) -> tuple[_TokenTermination, int]:
    """Classify what follows a complete identifier or calendar date."""

    if end == len(value):
        return "boundary", end

    following = value[end]
    if following.isspace():
        next_index = _skip_whitespace(value, end)
        if (
            spaced_dash_is_chain
            and next_index < len(value)
            and value[next_index] in _DASH_IDENTIFIER_SEPARATORS
        ):
            return "dash", next_index
        return "boundary", next_index
    if following in _CLEAR_TOKEN_SEPARATORS:
        return "boundary", end
    if following in _TERMINAL_TOKEN_PUNCTUATION:
        next_index = _terminal_punctuation_end(value, end)
        if next_index == len(value) or value[next_index] in _CLEAR_TOKEN_SEPARATORS:
            return "boundary", next_index
        if value[next_index].isspace():
            next_index = _skip_whitespace(value, next_index)
            if (
                spaced_dash_is_chain
                and next_index < len(value)
                and value[next_index] in _DASH_IDENTIFIER_SEPARATORS
            ):
                return "dash", next_index
            return "boundary", next_index
        if value[next_index] in _DASH_IDENTIFIER_SEPARATORS:
            return "dash", next_index
        return "invalid", next_index
    if following in _DASH_IDENTIFIER_SEPARATORS:
        return "dash", end
    return "invalid", end


def _record_identifier_matches(value: str) -> tuple[re.Match[str], ...]:
    """Return exact supported identifier tokens from arbitrary visible text."""

    candidates = tuple(_RECORD_IDENTIFIER_CANDIDATE.finditer(value))
    valid_starts: set[int] = set()
    for match in reversed(candidates):
        termination, index = _token_termination(
            value,
            match.end(),
            spaced_dash_is_chain=False,
        )
        if termination == "boundary":
            valid_starts.add(match.start())
        elif termination == "dash":
            next_start = _skip_opening_token_wrappers(value, index + 1)
            if next_start in valid_starts:
                valid_starts.add(match.start())
    return tuple(match for match in candidates if match.start() in valid_starts)


def _replace_record_identifiers(
    value: str,
    replacement: str | Callable[[re.Match[str]], str],
) -> str:
    """Replace only exact identifier tokens, leaving invalid lookalikes intact."""

    parts: list[str] = []
    cursor = 0
    for match in _record_identifier_matches(value):
        parts.append(value[cursor : match.start()])
        parts.append(replacement(match) if callable(replacement) else replacement)
        cursor = match.end()
    parts.append(value[cursor:])
    return "".join(parts)


def _calendar_date_at(
    value: str,
    start: int,
    cache: dict[int, bool],
) -> bool:
    """Return whether ``start`` begins a calendar-valid slash date."""

    if start in cache:
        return cache[start]

    pending: list[int] = []
    current = start
    while current not in cache:
        match = CALENDAR_DATE.match(value, current)
        if match is None:
            cache[current] = False
            break
        raw_year = match.group("year")
        year = int(raw_year) + (2000 if len(raw_year) == 2 else 0)
        try:
            date(year, int(match.group("month")), int(match.group("day")))
        except ValueError:
            cache[current] = False
            break

        pending.append(current)
        termination, index = _token_termination(
            value,
            match.end(),
            spaced_dash_is_chain=True,
        )
        if termination == "boundary":
            cache[current] = True
            break
        if termination == "invalid":
            cache[current] = False
            break
        current = _skip_opening_token_wrappers(value, index + 1)

    valid = cache[current]
    for pending_start in pending:
        cache[pending_start] = valid
    return valid


def _malformed_identifier_candidate(
    line: str,
    match: re.Match[str],
    valid_spans: set[tuple[int, int]],
    date_cache: dict[int, bool],
) -> bool:
    """Identify malformed strong IDs and legacy IDs used as identity text."""

    if (match.start(), match.end()) in valid_spans:
        return False
    if "/" not in match.group(0):
        return True
    if IDENTIFIER_LABEL_PREFIX.match(line) is not None:
        return True
    if _calendar_date_at(line, match.start(), date_cache):
        return False
    leading_text = _replace_record_identifiers(line[: match.start()], " ")
    leading_text = re.sub(
        r"\b(?:and|or)\b", " ", leading_text, flags=re.IGNORECASE | re.ASCII
    )
    return IDENTITY_SEPARATORS.fullmatch(leading_text) is not None


def _identifier_list_line(line: str) -> bool:
    """Recognise a line whose semantic content is only record identifiers."""

    body = IDENTIFIER_LABEL_PREFIX.sub("", line, count=1)
    matches = _record_identifier_matches(body)
    if not matches:
        return False
    residue = _replace_record_identifiers(body, " ")
    residue = CONTEXTUAL_UNIT.sub(" ", residue)
    residue = re.sub(r"\band\b", " ", residue, flags=re.IGNORECASE)
    return IDENTITY_SEPARATORS.fullmatch(residue) is not None


def identifier_evidence_details(
    lines: Iterable[str],
) -> tuple[tuple[str, ...], bool]:
    """Return supported identifiers plus malformed strong-identifier evidence."""

    identifiers: list[str] = []
    seen: set[str] = set()
    malformed = False
    for line in visual_lines(lines):
        legacy_is_identity = _identifier_list_line(line)
        valid_matches = _record_identifier_matches(line)
        valid_spans = {(match.start(), match.end()) for match in valid_matches}
        date_cache: dict[int, bool] = {}
        malformed = malformed or any(
            _malformed_identifier_candidate(line, match, valid_spans, date_cache)
            for match in RECORD_IDENTIFIER.finditer(line)
        )
        for match in valid_matches:
            identifier = normalise_identifier(match.group(0))
            if "/" in identifier and not legacy_is_identity:
                continue
            key = identifier.casefold()
            if key not in seen:
                seen.add(key)
                identifiers.append(identifier)
    return tuple(identifiers), malformed


def identifier_evidence(lines: Iterable[str]) -> tuple[str, ...]:
    """Return supported raw identifiers, suppressing slash-shaped dates in prose."""

    identifiers, _malformed = identifier_evidence_details(lines)
    return identifiers


@dataclass(frozen=True)
class UnitEvidence:
    candidates: frozenset[str]
    ambiguous: bool


def collect_unit_evidence(
    typed_lines: Iterable[str], handwritten_lines: Iterable[str]
) -> UnitEvidence:
    """Collect narrowly supported raw box-number evidence and ambiguity signals."""

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
        bare_typed_unit = re.fullmatch(r"\s*([0-9]{1,4})\s*", line)
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


def unit_claim_is_corroborated(
    value: str,
    typed_lines: Iterable[str],
    handwritten_lines: Iterable[str],
) -> bool:
    normalized = normalise_unit_claim(value)
    if normalized is None:
        return False
    evidence = collect_unit_evidence(typed_lines, handwritten_lines)
    return not evidence.ambiguous and evidence.candidates == {normalized}


def identity_only_line(
    line: str,
    structured_identifiers: Iterable[str],
    structured_unit: str,
) -> bool:
    """Return whether a raw line contains identity syntax and no annotation text."""

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

    body = _replace_record_identifiers(body, remove_identifier)
    if removed_identifier:
        body = IDENTIFIER_LABEL_PREFIX.sub("", body, count=1)
    body = re.sub(r"\band\b", " ", body, flags=re.IGNORECASE)
    body = body.strip(" \t,;:/|&+()[]{}-–—\"'“”‘’")
    if not body and removed_identifier:
        return True

    if match := re.match(rf"^\s*{UNIT_PREFIX}\s*[:#-]?\s*", body, re.I | re.ASCII):
        unit_body = body[match.end() :]
        unit_body = re.sub(
            r"\b(?:and\s*/\s*or|or|to|maybe|possibly|unclear|unknown|"
            r"ambiguous|overwritten|crossed\s*out|struck\s*out)\b",
            " ",
            unit_body,
            flags=re.IGNORECASE,
        )
        unit_body = re.sub(r"[0-9]{1,4}", " ", unit_body)
        return IDENTITY_SEPARATORS.fullmatch(unit_body) is not None

    normalized_claim = normalise_unit_claim(structured_unit)
    bare_unit = re.fullmatch(r"[0-9]{1,4}", body)
    return bool(
        normalized_claim
        and bare_unit
        and bare_unit.group(0).zfill(4) == normalized_claim
    )
