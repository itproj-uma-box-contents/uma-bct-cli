from dataclasses import dataclass, field
import re

ESCAPE_SQEUNCE = "#"

DATE_DELINIATOR_SYMBOL = "/"
RANGE_DELINIATOR_SYMBOL = "~"

MONTH_STRINGS = {
    1:["jan", "january"],
    2:["feb", "february"],
    3:["mar", "march"],
    4:["apr", "april"],
    5:["may"],
    6:["jun", "june"],
    7:["jul", "july"],
    8:["aug", "august"],
    9:["sep", "september"],
    10:["oct", "october"],
    11:["nov", "november"],
    12:["dec", "december"]
}

# A list of date formats, ensure substring are after thier parent string (i.e. yyyy before yy)
DATE_COMPONENT_REGEXES = {
    "dd": "(?:[12][0-9])|(?:3[01])|(?:0?[1-9])",
    "mm": "(?:1[012])|(?:0?[1-9])",
    "mon": "|".join(["|".join(month_forms) for month_forms in MONTH_STRINGS.values()]),
    "yyyy": "(?:1[5-9][0-9][0-9])|(?:20[0-3][0-9])",
    "yy": "[0-9][0-9]",
    "y": "[0-9]?[0-9]",
}

DEFAULT_DATE_DELINIATORS = (
    "/",
    ".",
)

DEFAULT_RANGE_DELINIATORS = (
    "-",
    "/",
    ".",
    "to",
)

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
    "mon yy",

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

#take two integers of length n, and use the first to deduce the second, secifically in the context of date ranges, i.e. given 1942-49 or 1942-9 obtain 1949
#only continues if second number is a greater suffixs, i.e. 1945-41, 1945-1 don't work, they return end instead
def fill_prefix(start: str, end: str) -> str:
    if start=="": return end
    if end=="": return start
    start_suffix = start[len(start)-len(end):]
    if int(start_suffix)<int(end):
        return start[:len(start)-len(end)]+end
    else: return end


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

    #makes a date format safe for use in other functions
    def escape_date_format(self, format: str) -> str:
            result = format.replace(ESCAPE_SQEUNCE, ESCAPE_SQEUNCE+ESCAPE_SQEUNCE)
            result = result.replace(DATE_DELINIATOR_SYMBOL, ESCAPE_SQEUNCE+DATE_DELINIATOR_SYMBOL)
            result = result.replace(RANGE_DELINIATOR_SYMBOL, ESCAPE_SQEUNCE+RANGE_DELINIATOR_SYMBOL)
            for date_compnent in DATE_COMPONENT_REGEXES:
                result = result.replace(date_compnent, ESCAPE_SQEUNCE+date_compnent)
            return re.escape(result)

    #gets the dictionary which maps date components to thier respective regexes 
    def get_date_component_to_regex_map(self) -> dict[str, str]:
        return DATE_COMPONENT_REGEXES | {DATE_DELINIATOR_SYMBOL: self.date_delineator_regex, RANGE_DELINIATOR_SYMBOL: self.range_delineator_regex}

    #converts a date format to a regex which matches it
    def date_format_to_regex(self, format: str) -> str:
        regex = self.escape_date_format(format)
        for (date_compnent, regex_component) in self.get_date_component_to_regex_map().items():
            if date_compnent in DATE_COMPONENT_REGEXES:
                #Create the capture groups keeping track of what component they are
                i=1
                while i!=0:
                    new_regex = regex.replace(self.escape_date_format(date_compnent), f"(?P<{date_compnent}{i}>{regex_component})", 1)
                    if new_regex == regex:
                        i=0
                    else:
                        i+=1
                    regex=new_regex
            else: 
                regex = regex.replace(self.escape_date_format(date_compnent), f"(?:{regex_component})")
        return "(?:" + regex + ")"

    #turns a match result from a date format regex to a date of the form dd/mm/yyyy or a date range dd/mm/yyyy-dd/mm/yyyy
    def standardise_match_result(self, match: re.Match) -> str:
            dates = [{"d": "", "m": "", "y": ""}, {"d": "", "m": "", "y": ""}]

            def add_to_dates(part, value):
                if dates[0][part] == "": dates[0][part]=value
                else: dates[1][part]=value

            #populate two dates, assume the first day/month/year you encounter is for the first date, and second for the second date
            for (part,value) in match.groupdict().items():
                if "yyyy" in part:
                    add_to_dates("y", value)
                elif "yy" in part:
                    add_to_dates("y", value)
                elif "yy" in part:
                    add_to_dates("y", value)
                elif "y" in part:
                    add_to_dates("y", value)
                elif "mm" in part:
                    add_to_dates("m", value)
                elif "mon" in part:
                    m=""
                    for (index, month_strings) in MONTH_STRINGS.items():
                        if value.lower() in month_strings:
                            m = index
                            break
                    add_to_dates("m", m)
                elif "dd" in part:
                    add_to_dates("d", value)

            #If first year is short and of reasonable size, assume its in the 2000s
            if len(dates[0]["y"])==2 and 0<int(dates[0]["y"])<30:
                dates[0]["y"] = "20"+dates[0]["y"]

            if dates[1]["y"]!="" or dates[1]["m"]!="" or dates[1]["d"]!="":
                #Deduce second date from first
                
                #Complete day range
                if dates[1]["y"]==dates[0]["y"] and dates[1]["m"]==dates[0]["m"] and dates[1]["d"]!="":
                    dates[1]["d"] = fill_prefix(dates[0]["d"], dates[1]["d"])

                #Complete year range    
                dates[1]["y"] = fill_prefix(dates[0]["y"], dates[1]["y"])

                #Complete empty
                for component in dates[1]:
                    if dates[1][component] == "": dates[1][component] = dates[0][component]

                return f"{dates[0]['d']}/{dates[0]['m']}/{dates[0]['y']} - {dates[1]['d']}/{dates[1]['m']}/{dates[1]['y']}"
            else:
                return f"{dates[0]['d']}/{dates[0]['m']}/{dates[0]['y']}"

    """
    Extacts dates from the provided string. 
    Searches for dates of the form in the date_formats attribute 
    Searches for dates ranges of the form in the custom_range_formats attribute, or formats automatically generated by taking two date formats in the date_formats attribute
    
    Longer matches are prioritised
    Given a substring which is recognised both a single date and date range, the date is returned date

    Custom date formats are prioritised over automatically generated formats 
    Dates with another date or date range are not returned, but two partly overlapping potential dates/date ranges are both returned 
    """
    def extract_dates(self, string: str) -> list[str]:
        dates: list[re.Match]  = []
        matches: list[re.Match] = []

        #find all possible dates, added in order given in documentation:

        #Check indiviaul dates
        for format in self.date_formats:
            date_regex = self.date_format_to_regex(format)
            matches.extend(re.finditer(date_regex, string, flags=re.IGNORECASE))

        #Check custon range formats
        for format in self.custom_range_formats:
            date_range_regex = self.date_format_to_regex(format)
            matches.extend(re.finditer(date_range_regex, string, flags=re.IGNORECASE))
        
        #Check generic ranges
        for format1 in self.date_formats:
            for format2 in self.date_formats:
                date_range_regex = self.date_format_to_regex(format1+"~"+format2)
                matches.extend(re.finditer(date_range_regex, string, flags=re.IGNORECASE))

        #find dates, prioritising longer matches, stable search ensure priority matches that given in documentation
        matches.sort(key = lambda x: len(x.group(0)), reverse=True)
        i=0

        #select matches
        while i<len(matches):
            #remove substrings (i.e dd/mm/yyyy contains the reduntant match dd/mm) 
            dates.append(matches[i])
            for j in reversed(range(i+1, len(matches))):
                if matches[j].start()>=matches[i].start() and matches[j].end()<=matches[i].end():
                    matches.pop(j)
            i+=1

        #sort back to original order
        dates.sort(key = lambda x: x.start())

        return [self.standardise_match_result(date) for date in dates]

