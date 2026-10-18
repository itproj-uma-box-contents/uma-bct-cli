"""Extract dates with the selected model and validate its reply locally."""

import re
from dataclasses import dataclass, replace
from typing import Any

from ..ports import DateExtraction, VisionModelClient
from ..util.date_util import Date, DateRange
from .regex_date_detector import MONTH_STRINGS, ambiguous_year_spans, fill_prefix, spaced_month_dates

DATE_EXTRACTION_PROMPT = """Extract dates and date ranges from the supplied archive-box text.
Return each date's exact source text in raw_text and its zero-based character
position in starting_index. Keep entries in reading order without overlapping
spans. Do not also list the individual dates contained within a date range.

For the text "Filed 10/12/1950; 2021-23", return:
{"dates": [
  {"raw_text": "10/12/1950", "starting_index": 6,
   "start_date": {"day": 10, "month": 12, "year": 1950}, "end_date": {}},
  {"raw_text": "2021-23", "starting_index": 17,
   "start_date": {"year": 2021}, "end_date": {"year": 2023}}
]}

Recognize full dates, month/year, year-only dates, month names, and ranges such
as dd-dd/mm/yyyy, mm-mm/yyyy, yyyy-yyyy and yyyy-y. Omit missing components;
never guess ambiguous dates. A single date has an empty end_date object.
Dates must be valid calendar dates, and range ends must not precede starts.
Omit years with an ambiguous century, including standalone two-digit years;
never choose a century. A shortened range end is allowed only when a complete
first year supplies its prefix, such as 2021-23. Keep other unambiguous dates.
Identifiers and box numbers are not dates. Return an empty dates array when
no unambiguous dates are present.

Extract dates from the following text:
"""

MAX_DATES = 100
# Full month-name ranges need more space than numeric dates.
MAX_DATE_LEN = 100
MAX_YEAR = 2050
SOURCE_NUMBER = re.compile(r"(?<!\w)([0-9]+)(?:st|nd|rd|th)?(?!\w)", re.IGNORECASE)
SOURCE_WORD = re.compile(r"\b[A-Za-z]+\b")
MONTH_NUMBERS = {name: number for number, names in MONTH_STRINGS.items() for name in names} | {"sept": 9}
RANGE_EVIDENCE = re.compile(r"[-–—~/.]|\b(?:to|until|through|between|from)\b", re.IGNORECASE)


def get_result_schema(string: str) -> dict[str, Any]:
    """Describe the same bounds enforced by the local response validator."""

    date_schema = {
        "type": "object",
        "properties": {
            "day": {"type": "integer", "minimum": 1, "maximum": 31},
            "month": {"type": "integer", "minimum": 1, "maximum": 12},
            "year": {"type": "integer", "minimum": 1, "maximum": MAX_YEAR},
        },
        "additionalProperties": False,
    }
    # Empty text can only produce an empty array. Keep its item schema valid
    # too, since some providers inspect it even when maxItems is zero.
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "dates": {
                "type": "array",
                "maxItems": MAX_DATES if string else 0,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "raw_text": {"type": "string", "minLength": 1, "maxLength": MAX_DATE_LEN},
                        "starting_index": {"type": "integer", "minimum": 0, "maximum": max(0, len(string) - 1)},
                        "start_date": date_schema,
                        "end_date": date_schema,
                    },
                    "required": ["raw_text", "starting_index", "start_date", "end_date"],
                },
            }
        },
        "required": ["dates"],
    }


def parse_date(value: Any, *, allow_empty: bool = False, unresolved_year: bool = False) -> Date:
    """Accept omitted components, but reject invalid or invented structure."""

    if not isinstance(value, dict) or set(value) - {"day", "month", "year"}:
        raise ValueError("Model date components must be a day/month/year object.")
    if not value and not allow_empty:
        raise ValueError("A model date must contain at least one component.")
    if any(type(component) is not int for component in value.values()):
        raise ValueError("Model date components must be integers.")

    # Calendar checks distinguish a missing component from zero or false.
    # Keep the existing model year ceiling without changing regex formats.
    date = Date(**value)
    calendar_date = replace(date, year=None) if unresolved_year else date
    if not calendar_date.is_valid() or (date.year is not None and not 1 <= date.year <= MAX_YEAR):
        raise ValueError("The model returned an invalid calendar date.")
    return date


def source_year_matches(year: int, number: str, first_year: int | None, *, full_year_present: bool) -> bool:
    """Corroborate complete years and suffixes anchored to a complete first year."""

    # A visible complete year outranks shorter tokens that may be days/months.
    if int(number) == year and (not full_year_present or len(number) == 4):
        return True
    if not full_year_present and len(number) == 2 and year % 100 == int(number):
        return True
    if first_year is None or len(number) >= len(str(first_year)):
        return False
    prefix = str(first_year)
    return int(prefix[-len(number):]) < int(number) and str(year) == prefix[:-len(number)] + number


def source_component_tokens(date: Date, numbers: tuple[str, ...], months: tuple[int, ...],
                            first_year: int | None = None) -> list[set[int]]:
    """Find separate source witnesses for each supplied date component."""

    candidates: list[set[int]] = []
    full_year_present = any(len(value) == 4 for value in numbers)
    for field, component in (("day", date.day), ("month", date.month), ("year", date.year)):
        if component is None:
            continue
        if field == "year":
            matches = {index for index, value in enumerate(numbers) if source_year_matches(
                component, value, first_year, full_year_present=full_year_present,
            )}
        else:
            matches = {index for index, value in enumerate(numbers) if len(value) <= 2 and int(value) == component}
            if field == "month":
                matches.update(len(numbers) + index for index, value in enumerate(months) if value == component)
        candidates.append(matches)
    return candidates


def distinct_source_tokens(candidates: list[set[int]], used: frozenset[int] = frozenset()) -> bool:
    """One visible number cannot create both a day and a month."""

    if not candidates:
        return True
    first, *remaining = sorted(candidates, key=len)
    return any(distinct_source_tokens(remaining, used | {token}) for token in first - used)


def validate_source_components(raw_text: str, start: Date, end: Date | None) -> None:
    """Reject unsupported components while leaving span interpretation to AI."""

    evidence = spaced_month_dates(raw_text)
    numbers = tuple(match.group(1) for match in SOURCE_NUMBER.finditer(evidence))
    months = tuple(MONTH_NUMBERS[word] for match in SOURCE_WORD.finditer(evidence)
                   if (word := match.group().casefold()) in MONTH_NUMBERS)
    first = source_component_tokens(start, numbers, months)
    supported = distinct_source_tokens(first)
    if end is not None:
        # Range endpoints may share a printed year or month.
        last = source_component_tokens(end, numbers, months, start.year)
        supported &= bool(RANGE_EVIDENCE.search(raw_text)) and distinct_source_tokens(last)
    if not supported:
        raise ValueError("The model date components are not supported by its source text.")


def parse_qwen_result(result: Any, string: str | None = None) -> tuple[Date | DateRange, ...]:
    """Preserve the tuple API for source-validated, unambiguous dates."""

    return parse_qwen_extraction(result, string).dates


def parse_qwen_entry(entry: Any, string: str | None,
                     ambiguous_spans: tuple[tuple[int, int], ...]
                     ) -> tuple[int, int, Date | DateRange | None, tuple[int, int] | None]:
    """Validate one source span; None means its century requires human review."""

    if not isinstance(entry, dict) or set(entry) != {"raw_text", "starting_index", "start_date", "end_date"}:
        raise ValueError("A model date entry has invalid fields.")
    raw_text, position = entry["raw_text"], entry["starting_index"]
    if not isinstance(raw_text, str) or not raw_text.strip() or len(raw_text) > MAX_DATE_LEN:
        raise ValueError("A model date entry has invalid source text.")
    if type(position) is not int or position < 0:
        raise ValueError("A model date entry has an invalid source position.")
    # Check quoted evidence without echoing private contents in an error.
    end_position = position + len(raw_text)
    if string is not None and (end_position > len(string) or string[position:end_position] != raw_text):
        raise ValueError("A model date does not match its source text.")
    numbers = tuple(match.group(1) for match in SOURCE_NUMBER.finditer(spaced_month_dates(raw_text)))
    claimed_year = any(isinstance(value, dict) and "year" in value
                       for value in (entry["start_date"], entry["end_date"]))
    ambiguous = string is not None and (
        any(position < right and end_position > left for left, right in ambiguous_spans)
        or (claimed_year and any(len(value) <= 2 for value in numbers)
            and not any(len(value) == 4 for value in numbers))
    )
    start = parse_date(entry["start_date"], unresolved_year=ambiguous)
    end = parse_date(entry["end_date"], allow_empty=True, unresolved_year=ambiguous)
    date = DateRange(start, end) if entry["end_date"] else start
    resolved = anchored_end_span(raw_text, position, start, end, ambiguous_spans) if ambiguous else None
    # The anchor must explain every ambiguous date in this quoted span.
    # An unrelated short year keeps the entire candidate out of the catalogue.
    if resolved is not None and any(
        position < right and end_position > left and not (left < resolved[1] and right > resolved[0])
        for left, right in ambiguous_spans
    ):
        resolved = None
    if resolved is not None and date.is_valid():
        try:
            validate_source_components(raw_text, start, end)
        except ValueError:
            resolved = None
        else:
            ambiguous = False
    if not ambiguous and not date.is_valid():
        raise ValueError("The model returned a reversed date range.")
    if string is not None:
        source_start = replace(start, year=None) if ambiguous else start
        source_end = replace(end, year=None) if ambiguous else end
        validate_source_components(raw_text, source_start, source_end if entry["end_date"] else None)
    return position, end_position, None if ambiguous else date, resolved if not ambiguous else None


def anchored_end_span(raw_text: str, position: int, start: Date, end: Date,
                      ambiguous_spans: tuple[tuple[int, int], ...]) -> tuple[int, int] | None:
    """Resolve one short ending only after its literal complete start year."""

    if start.year is None or end.year is None:
        return None
    anchor = re.search(rf"(?<!\d){start.year:04d}(?!\d)", raw_text)
    if anchor is None:
        return None
    for left, right in sorted(ambiguous_spans):
        if not position + anchor.end() <= left < right <= position + len(raw_text):
            continue
        evidence = spaced_month_dates(raw_text[left - position:right - position])
        numbers = [match.group(1) for match in SOURCE_NUMBER.finditer(evidence)]
        if numbers and len(numbers[-1]) < 4 and fill_prefix(str(start.year), numbers[-1]) == str(end.year):
            return left, right
    return None


def parse_qwen_extraction(result: Any, string: str | None = None) -> DateExtraction:
    """Keep malformed replies as failures and century ambiguity as review data."""

    if not isinstance(result, dict) or set(result) != {"dates"}:
        raise ValueError("The model date response must contain a dates array.")
    entries = result["dates"]
    if not isinstance(entries, list) or len(entries) > MAX_DATES:
        raise ValueError("The model date response has an invalid dates array.")

    ambiguous_spans = ambiguous_year_spans(string) if string is not None else ()
    needs_review = False
    resolved_spans: set[tuple[int, int]] = set()
    results_with_position: list[tuple[int, int, Date | DateRange | None]] = []
    for entry in entries:
        position, end_position, date, resolved = parse_qwen_entry(entry, string, ambiguous_spans)
        needs_review |= date is None
        if resolved is not None:
            # A compact end can also produce a spurious overlapping fragment,
            # such as "1950-15" inside "12May1950-15June51".
            resolved_spans.update(span for span in ambiguous_spans
                                  if position <= span[0] < resolved[1] and resolved[0] < span[1] <= end_position)
        results_with_position.append((position, end_position, date))

    # Sorting must not turn duplicate or overlapping evidence into separate
    # catalogue dates. Reject the reply so the original reading stays visible.
    results_with_position.sort(key=lambda item: item[0])
    previous_end = 0
    for position, end_position, _ in results_with_position:
        if position < previous_end:
            raise ValueError("The model returned overlapping date entries.")
        previous_end = end_position
    needs_review |= any(span not in resolved_spans for span in ambiguous_spans)
    return DateExtraction(tuple(date for _, _, date in results_with_position if date is not None), needs_review)


@dataclass(frozen=True)
class QwenDateDetector:
    client: VisionModelClient
    name: str = "qwen"

    def extract_dates(self, string: str) -> tuple[Date | DateRange, ...]:
        """Preserve the tuple API while omitting dates that need century review."""

        return self.extract_dates_with_review(string).dates

    def extract_dates_with_review(self, string: str) -> DateExtraction:
        """Extract dates and return source ambiguity without retrying the model."""

        if not string.strip():
            return DateExtraction(())

        # Validate inside the client as well as here: real clients validate
        # parsed replies, while simple adapters may only return ModelResponse.
        result = self.client.query(
            None, DATE_EXTRACTION_PROMPT + string, get_result_schema(string),
            validator=lambda value: parse_qwen_extraction(value, string), logprobs=False,
        )
        return parse_qwen_extraction(result.value, string)
