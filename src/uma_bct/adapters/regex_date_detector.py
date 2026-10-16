import re
from dataclasses import dataclass, field

from ..util.date_util import Date, DateRange

ESCAPE_SQEUNCE = "#"

DATE_DELINIATOR_SYMBOL = "/"
RANGE_DELINIATOR_SYMBOL = "~"

MONTH_STRINGS = {
    1: ["jan", "january"],
    2: ["feb", "february"],
    3: ["mar", "march"],
    4: ["apr", "april"],
    5: ["may"],
    6: ["jun", "june"],
    7: ["jul", "july"],
    8: ["aug", "august"],
    9: ["sep", "september"],
    10: ["oct", "october"],
    11: ["nov", "november"],
    12: ["dec", "december"],
}

# Replace longer placeholders before their suffixes, such as yyyy before yy.
DATE_COMPONENT_REGEXES = {
    "dd": "(?:[12][0-9])|(?:3[01])|(?:0?[1-9])",
    "mm": "(?:1[012])|(?:0?[1-9])",
    "mon": "|".join(["|".join(month_forms) for month_forms in MONTH_STRINGS.values()]),
    "yyyy": "(?:1[5-9][0-9][0-9])|(?:20[0-3][0-9])",
    "yy": "[0-9][0-9]",
    "y": "[0-9]?[0-9]",
}

DEFAULT_DATE_DELINIATORS = ("/", ".")

DEFAULT_RANGE_DELINIATORS = ("-", "/", ".", "to")

DEFAULT_DATE_FORMATS = (
    "dd/mm/yyyy",
    "mm/dd/yyyy",
    "mm/yyyy",
    "mm/yy",
    "dd/mm",
    "dd mon yyyy",
    "ddth mon yyyy",
    "1st mon yyyy",
    "2nd mon yyyy",
    "dd mon yy",
    "ddth mon yy",
    "1st mon yy",
    "2nd mon yy",
    "mon yyyy",
    "yyyy",
)

DEFAULT_CUSTOM_RANGE_FORMATS = (
    "dd~dd/mm/yyyy",
    "dd~dd/mm",
    "dd~dd/mon/yyyy",
    "dd~dd/mon",
    "mm~mm/yyyy",
    "mm~mm/yy",
    "mon~mon/yyyy",
    "mon~mon/yy",
    "yyyy~y",
)


def fill_prefix(start: str, end: str) -> str:
    """Expand an increasing shorthand suffix: 1942-49 or 1942-9 ends in 1949.

    A non-increasing suffix stays unchanged, so reversed ranges are rejected.
    """
    if start == "":
        return end
    if end == "":
        return start
    start_suffix = start[len(start) - len(end) :]
    if int(start_suffix) < int(end):
        return start[: len(start) - len(end)] + end
    else:
        return end


def complete_numeric_token(match: re.Match[str]) -> bool:
    """Reject date fragments inside digit-dot-digit tokens, such as record IDs."""

    text = match.string
    start, end = match.span()
    if start == end:
        return False
    # A month name after numbered punctuation ("1.May 1950") remains valid;
    # only an adjacent digit-dot-digit sequence makes the match incomplete.
    starts_inside_number = (
        start >= 2 and text[start].isdigit() and text[start - 1] == "." and text[start - 2].isdigit()
    )
    ends_inside_number = (
        end + 1 < len(text) and text[end - 1].isdigit() and text[end] == "." and text[end + 1].isdigit()
    )
    return not starts_inside_number and not ends_inside_number


@dataclass
class RegexDateDetector:
    name: str = "regex"

    date_range_delineators: tuple[str, ...] = field(default=DEFAULT_RANGE_DELINIATORS)
    date_delineators: tuple[str, ...] = field(default=DEFAULT_DATE_DELINIATORS)
    date_formats: tuple[str, ...] = field(default=DEFAULT_DATE_FORMATS)
    custom_range_formats: tuple[str, ...] = field(default=DEFAULT_CUSTOM_RANGE_FORMATS)

    range_delineator_regex: str = field(init=False)
    date_delineator_regex: str = field(init=False)

    def __post_init__(self) -> None:
        # Build alternatives for the configured range separators.
        self.range_delineator_regex = " ?(?:" + re.escape(self.date_range_delineators[0])
        for i in range(1, len(self.date_range_delineators)):
            self.range_delineator_regex += "|" + re.escape(self.date_range_delineators[i])
        self.range_delineator_regex += ") ?"

        # Date separators are separate from the separators between two dates.
        self.date_delineator_regex: str = " ?(?:" + re.escape(self.date_delineators[0])
        for i in range(1, len(self.date_delineators)):
            self.date_delineator_regex += "|" + re.escape(self.date_delineators[i])
        self.date_delineator_regex += ") ?"

    def escape_date_format(self, format: str) -> str:
        """Mark date placeholders before escaping literal regex characters."""
        result = format.replace(ESCAPE_SQEUNCE, ESCAPE_SQEUNCE + ESCAPE_SQEUNCE)
        result = result.replace(DATE_DELINIATOR_SYMBOL, ESCAPE_SQEUNCE + DATE_DELINIATOR_SYMBOL)
        result = result.replace(RANGE_DELINIATOR_SYMBOL, ESCAPE_SQEUNCE + RANGE_DELINIATOR_SYMBOL)
        for component in DATE_COMPONENT_REGEXES:
            result = result.replace(component, ESCAPE_SQEUNCE + component)
        return re.escape(result)

    def get_date_component_to_regex_map(self) -> dict[str, str]:
        """Map date components and separators to their matching patterns."""
        return DATE_COMPONENT_REGEXES | {
            DATE_DELINIATOR_SYMBOL: self.date_delineator_regex,
            RANGE_DELINIATOR_SYMBOL: self.range_delineator_regex,
        }

    def date_format_to_regex(self, format: str) -> str:
        """Give repeated date components distinct named capture groups."""
        regex = self.escape_date_format(format)
        for component, regex_component in self.get_date_component_to_regex_map().items():
            if component in DATE_COMPONENT_REGEXES:
                # A range can contain the same component twice; number each
                # occurrence so its start and end values stay distinguishable.
                i = 1
                while i != 0:
                    new_regex = regex.replace(
                        self.escape_date_format(component), f"(?P<{component}{i}>{regex_component})", 1
                    )
                    if new_regex == regex:
                        i = 0
                    else:
                        i += 1
                    regex = new_regex
            else:
                regex = regex.replace(self.escape_date_format(component), f"(?:{regex_component})")
        # Match whole numeric components rather than fragments of longer IDs.
        # Dotted-token boundaries are checked separately after matching.
        return r"(?<!\d)(?:" + regex + r")(?!\d)"

    def standardise_match_result(self, match: re.Match) -> Date | DateRange:
        """Map captured components to one date or the endpoints of a range."""
        dates = [{"d": "", "m": "", "y": ""}, {"d": "", "m": "", "y": ""}]

        def add_to_dates(part, value):
            if dates[0][part] == "":
                dates[0][part] = value
            else:
                dates[1][part] = value

        # First occurrences describe the start date; repeated components
        # describe the end date.
        for part, value in match.groupdict().items():
            if "yyyy" in part:
                add_to_dates("y", value)
            elif "yy" in part:
                add_to_dates("y", value)
            elif "y" in part:
                add_to_dates("y", value)
            elif "mm" in part:
                add_to_dates("m", value)
            elif "mon" in part:
                m = ""
                for index, month_strings in MONTH_STRINGS.items():
                    if value.lower() in month_strings:
                        m = index
                        break
                add_to_dates("m", m)
            elif "dd" in part:
                add_to_dates("d", value)

        # Preserve the established interpretation of complete dates with
        # two-digit years 01-29; partial dates keep their printed precision.
        if dates[0]["d"] != "" and dates[0]["m"] != "" and len(dates[0]["y"]) == 2 and 0 < int(dates[0]["y"]) < 30:
            dates[0]["y"] = "20" + dates[0]["y"]

        if dates[1]["y"] != "" or dates[1]["m"] != "" or dates[1]["d"] != "":
            # Expand an abbreviated day only when both dates share a month.
            if dates[1]["y"] == dates[0]["y"] and dates[1]["m"] == dates[0]["m"] and dates[1]["d"] != "":
                dates[1]["d"] = fill_prefix(dates[0]["d"], dates[1]["d"])

            # Expand an abbreviated year, then carry forward omitted parts.
            dates[1]["y"] = fill_prefix(dates[0]["y"], dates[1]["y"])

            for component in dates[1]:
                if dates[1][component] == "":
                    dates[1][component] = dates[0][component]

            return DateRange(
                start=Date(
                    day=int(dates[0]["d"]) if dates[0]["d"] else None,
                    month=int(dates[0]["m"]) if dates[0]["m"] else None,
                    year=int(dates[0]["y"]) if dates[0]["y"] else None,
                ),
                end=Date(
                    day=int(dates[1]["d"]) if dates[1]["d"] else None,
                    month=int(dates[1]["m"]) if dates[1]["m"] else None,
                    year=int(dates[1]["y"]) if dates[1]["y"] else None,
                ),
            )
        else:
            return Date(
                day=int(dates[0]["d"]) if dates[0]["d"] else None,
                month=int(dates[0]["m"]) if dates[0]["m"] else None,
                year=int(dates[0]["y"]) if dates[0]["y"] else None,
            )

    def extract_dates(self, string: str) -> tuple[Date | DateRange, ...]:
        """Extract configured dates, preferring longer matches and custom ranges.

        Stable ordering favours individual dates over equal-length ranges.
        Contained matches are removed before restoring source reading order.
        """
        dates: list[re.Match] = []
        matches: list[re.Match] = []

        # Collect single dates first so equal-length matches keep that priority.
        for format in self.date_formats:
            date_regex = self.date_format_to_regex(format)
            matches.extend(re.finditer(date_regex, string, flags=re.IGNORECASE))

        # Explicit range formats take precedence over generic combinations.
        for format in self.custom_range_formats:
            date_range_regex = self.date_format_to_regex(format)
            matches.extend(re.finditer(date_range_regex, string, flags=re.IGNORECASE))

        # Any two configured date formats can form a generic range.
        for format1 in self.date_formats:
            for format2 in self.date_formats:
                date_range_regex = self.date_format_to_regex(format1 + "~" + format2)
                matches.extend(re.finditer(date_range_regex, string, flags=re.IGNORECASE))

        # Keep complete dotted dates/ranges, but never extract only the year
        # from a longer dotted identifier such as 1980.0950.
        matches = [match for match in matches if complete_numeric_token(match)]
        # Stable sorting preserves the format priorities for equal lengths.
        matches.sort(key=lambda x: len(x.group(0)), reverse=True)
        i = 0

        while i < len(matches):
            # A full date must not also produce its contained day/month match.
            if self.standardise_match_result(matches[i]).is_valid():
                dates.append(matches[i])

            for j in reversed(range(i + 1, len(matches))):
                if matches[j].start() >= matches[i].start() and matches[j].end() <= matches[i].end():
                    matches.pop(j)
            i += 1

        # Restore left-to-right reading order after selecting complete matches.
        dates.sort(key=lambda x: x.start())

        return tuple(self.standardise_match_result(date) for date in dates)
