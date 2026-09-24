from dataclasses import dataclass
from ..ports import VisionModelClient
from .ollama_client import OllamaVisionModel


DATE_EXTRACTION_PROMPT = """You will be provided with text taken from a box in a University archive.
The box contains a number of dates (e.g. 19/12/1950) and ranges of dates (e.g. 2024-5). Extract these from the text and report them in the provided json format.
In the case of incomplete dates, ommit the missing parts, do not attempt to guess ambiguous cases.
Also ensure you include the original date or date range string under the raw_text field, and the position of the first character under starting_index.
All dates and date ranges should be non overlapping
Dates which are a part of a date range should not be listed as indivial dates.

For example, the string "10/12/1950. In 5/1921, 1850. Confrence 3.1930 to 5.1933, then 2021-23, 5-9/1811" becomes:
{
"dates": [
{"raw_text": "10/12/1950", "starting_index": 0, "day": "10", "month": "12", "year": "1950"},
{"raw_text": "5/1921", "starting_index": 15, "month": "5", "year": "1921"},
{"raw_text": "1850", "starting_index": 23, "year": "1850"}
],

"date_ranges": [
{"raw_text": "3.1930 to 5.1933", "starting_index": 39, "start_date": {"month": "3", "year": "1930"}, "end_date": {"month": "5", "year": "1933"}},
{"raw_text": "2021-23", "starting_index": 62, "start_date": {"year": "2021"}, "end_date": {"year" : "2023"}},
{"raw_text": "5-9/1811", "starting_index": 71, "start_date": {"month": "5", "year": "1811"}, "end_date": {"month": "9", "year": "1811"}}
]
}


Extract the dates from the folllowing string:
"""

MAX_DATES = 100
MAX_DATE_LEN = 30

DATE_RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "dates": {
            "type": "array",
            "maxItems": MAX_DATES,
            "items": {
                "type": "object",
                "properties": {
                    "raw_text": {"type": "string", "maxLength": MAX_DATE_LEN},
                    "starting_index": {"type": "integer"},
                    "day": {"type": "string", "maxLength": 2},
                    "month": {"type": "string", "maxLength": 2},
                    "year": {"type": "string", "maxLength": 4}
                },
                "required": ["raw_text", "starting_index"]
            }
        },
        "date_ranges": {
            "type": "array",
            "items": {
                "type": "object",
                "maxItems": MAX_DATES,
                "properties": {
                    "raw_text": {"type": "string", "maxLength": MAX_DATE_LEN},
                    "starting_index": {"type": "integer"},
                    "start_date": {
                        "type": "object",
                        "properties": {
                            "day": {"type": "string", "maxLength": 2},
                            "month": {"type": "string", "maxLength": 2},
                            "year": {"type": "string", "maxLength": 4}
                        },
                        "required": []
                    },
                    "end_date": {
                        "type": "object",
                        "properties": {
                            "day": {"type": "string", "maxLength": 2},
                            "month": {"type": "string", "maxLength": 2},
                            "year": {"type": "string", "maxLength": 4}
                        },
                        "required": []
                    }
                },
                "required": ["raw_text", "starting_index", "start_date", "end_date"]
            }
        }
    },
    "required": ["dates", "date_ranges"],
}

def parse_qwen_result(result: dict[str, any]) -> tuple[str, ...]:
    results_with_position = []
    for date_result in result["dates"]:
        results_with_position.append((f"{date_result.get("day", "")}/{date_result.get("month", "")}/{date_result.get("year", "")}", date_result['starting_index']))

    for date_result in result["date_ranges"]:
        results_with_position.append((f"{date_result['start_date'].get("day", "")}/{date_result['start_date'].get("month", "")}/{date_result['start_date'].get("year", "")} - {date_result['end_date'].get("day", "")}/{date_result['end_date'].get("month", "")}/{date_result['end_date'].get("year", "")}", date_result['starting_index']))

    results_with_position.sort(key = lambda x: x[1])
    return tuple(result[0] for result in results_with_position)

@dataclass(frozen=True)
class QwenDateDetector:
    client: VisionModelClient
    name: str = "qwen"

    def extract_dates(self, string: str) -> tuple[str, ...]:
        result = self.client.query(None, DATE_EXTRACTION_PROMPT+string, DATE_RESULT_SCHEMA, validator=None, logprobs=False)
        return parse_qwen_result(result.value)
