"""Extract dates with the selected model and validate its reply locally."""

from dataclasses import dataclass
from typing import Any

from ..ports import VisionModelClient
from ..util.date_util import Date, DateRange

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
Identifiers and box numbers are not dates. Return an empty dates array when
no unambiguous dates are present.

Extract dates from the following text:
"""

MAX_DATES = 100
# Full month-name ranges need more space than numeric dates.
MAX_DATE_LEN = 100
MAX_YEAR = 2050


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


def parse_date(value: Any, *, allow_empty: bool = False) -> Date:
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
    if not date.is_valid() or (date.year is not None and date.year > MAX_YEAR):
        raise ValueError("The model returned an invalid calendar date.")
    return date


def parse_qwen_result(result: Any, string: str | None = None) -> tuple[Date | DateRange, ...]:
    """Reject malformed dates and retain the source order of valid replies."""

    if not isinstance(result, dict) or set(result) != {"dates"}:
        raise ValueError("The model date response must contain a dates array.")
    entries = result["dates"]
    if not isinstance(entries, list) or len(entries) > MAX_DATES:
        raise ValueError("The model date response has an invalid dates array.")

    results_with_position: list[tuple[int, int, Date | DateRange]] = []
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"raw_text", "starting_index", "start_date", "end_date"}:
            raise ValueError("A model date entry has invalid fields.")
        raw_text, position = entry["raw_text"], entry["starting_index"]
        if not isinstance(raw_text, str) or not raw_text.strip() or len(raw_text) > MAX_DATE_LEN:
            raise ValueError("A model date entry has invalid source text.")
        if type(position) is not int or position < 0:
            raise ValueError("A model date entry has an invalid source position.")

        # A model-supplied position is evidence only when its text really
        # occurs there. Error messages never echo private label contents.
        end_position = position + len(raw_text)
        if string is not None and (end_position > len(string) or string[position:end_position] != raw_text):
            raise ValueError("A model date does not match its source text.")
        start = parse_date(entry["start_date"])
        end = parse_date(entry["end_date"], allow_empty=True)
        date = DateRange(start, end) if entry["end_date"] else start
        if not date.is_valid():
            raise ValueError("The model returned a reversed date range.")
        results_with_position.append((position, end_position, date))

    # Sorting must not turn duplicate or overlapping evidence into separate
    # catalogue dates. Reject the reply so the original reading stays visible.
    results_with_position.sort(key=lambda item: item[0])
    previous_end = 0
    for position, end_position, _ in results_with_position:
        if position < previous_end:
            raise ValueError("The model returned overlapping date entries.")
        previous_end = end_position
    return tuple(date for _, _, date in results_with_position)


@dataclass(frozen=True)
class QwenDateDetector:
    client: VisionModelClient
    name: str = "qwen"

    def extract_dates(self, string: str) -> tuple[Date | DateRange, ...]:
        if not string.strip():
            return ()

        # Validate inside the client as well as here: real clients validate
        # parsed replies, while simple adapters may only return ModelResponse.
        result = self.client.query(
            None, DATE_EXTRACTION_PROMPT + string, get_result_schema(string),
            validator=lambda value: parse_qwen_result(value, string), logprobs=False,
        )
        return parse_qwen_result(result.value, string)
