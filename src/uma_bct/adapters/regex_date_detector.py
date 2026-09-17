from dataclasses import dataclass, field
import re

DATE_DELINIATOR_SYMBOL = "/"
RANGE_DELINIATOR_SYMBOL = "~"

# A list of date formats, ensure substring are after thier parent string (i.e. yyyy before yy)
DATE_COMPONENT_REGEXS = {
    "dd": "(?:[12][0-9])|(?:3[01])|(?:0?[1-9])",
    "mm": "(?:1[012])|(?:0?[1-9])",
    "mon": "jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec",
    "yyyy": "(?:1[5-9][0-9][0-9])|(?:20[0-3][0-9])",
    "yy": "[0-9][0-9]",
    "y": "[0-9]?[0-9]",
}

DEFAULT_RANGE_DELINIATORS = (
    "-",
    "to",
)

DEFAULT_DATE_DELINIATORS = (
    "/",
    ".",
)

DEFAULT_DATE_FORMATS = (
    "dd/mm/yyyy",
    "mm/dd/yyyy",
    "mm/yyyy",
    "mm/yy",
    
    "ddth mon yyyy",
    "ddth mon yy",
    "1st mon yyyy",
    "2nd mon yy",
    "mon yyyy",
    "mon yy",

    "yyyy",
)

DEFAULT_CUSTOM_RANGE_FORMATS = (
    "dd~dd/mm/yyyy",
    "mm~mm/yyyy",
    "yyyy~y",
)

@dataclass
class RegexDateDetector:
    date_range_delineators: tuple[str, ...] = field(default=DEFAULT_RANGE_DELINIATORS)
    date_delineators: tuple[str, ...] = field(default=DEFAULT_DATE_DELINIATORS)
    date_formats: tuple[str, ...] = field(default=DEFAULT_DATE_FORMATS)
    custom_range_formats: tuple[str, ...] = field(default=DEFAULT_CUSTOM_RANGE_FORMATS)

    range_delineator_regex: str = field(init=False)
    date_delineator_regex: str = field(init=False)

    def __post_init__(self) -> None:
        #consturct regex for range deliniators
        self.range_delineator_regex = " ?(?:"+re.escape(self.date_range_delineators[0])
        for i in range(1,len(self.date_range_delineators)):
            self.range_delineator_regex+="|"+re.escape(self.date_range_delineators[i])
        self.range_delineator_regex+=") ?"

        #consturct regex for date deliniators
        self.date_delineator_regex: str = " ?(?:"+re.escape(self.date_delineators[0])
        for i in range(1,len(self.date_delineators)):
            self.date_delineator_regex+="|"+re.escape(self.date_delineators[i])
        self.date_delineator_regex+=") ?"

    def get_date_component_to_regex_map(self) -> dict[str, str]:
        return DATE_COMPONENT_REGEXS | {re.escape(DATE_DELINIATOR_SYMBOL): self.date_delineator_regex, re.escape(RANGE_DELINIATOR_SYMBOL): self.range_delineator_regex}

    def date_format_to_regex(self, format: str) -> str:
        regex = re.escape(format)
        for (date_compnent, regex_component) in self.get_date_component_to_regex_map().items():
            regex = regex.replace(date_compnent, "(?:"+regex_component+")")
        return "(?:" + regex + ")"

    def regex_extract_dates(self, string: str) -> list[str]:
        dates = []
        matches: list[re.Match] = []

        #find all possible dates:
        
        #Check generic ranges
        for format1 in self.date_formats:
            for format2 in self.date_formats:
                date_range_regex = self.date_format_to_regex(format1+"~"+format2)
                matches.extend(re.finditer(date_range_regex, string, flags=re.IGNORECASE))

        #Check custon range formats
        for format in self.custom_range_formats:
            date_range_regex = self.date_format_to_regex(format)
            matches.extend(re.finditer(date_range_regex, string, flags=re.IGNORECASE))

        #Check indiviaul dates
        for format in self.date_formats:
            date_regex = self.date_format_to_regex(format)
            matches.extend(re.finditer(date_regex, string, flags=re.IGNORECASE))

        #find dates, prioritising longer matches
        matches.sort(key = lambda x: (len(x.group(0) ), x.group(0)), reverse=True)
        i=0
        while i<len(matches):
            #remove substrings (i.e dd/mm/yyyy contains the reduntant mach dd/mm) 
            dates.append(matches[i])
            for j in reversed(range(i+1, len(matches))):
                if matches[j].start()>=matches[i].start() and matches[j].end()<=matches[i].end():
                    matches.pop(j)
            i+=1

        #sort back to original order
        dates.sort(key = lambda x: x.start())
        return [date.group(0) for date in dates]