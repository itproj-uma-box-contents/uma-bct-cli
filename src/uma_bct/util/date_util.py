from dataclasses import dataclass
from datetime import datetime

class InvalidDate(Exception):
    """Provided parameters do not form a valid date."""

def component_to_string(component: int | None, width: int) -> str:
    return f"{component:0{width}}" if component != None else ""

@dataclass(frozen=True)
class Date:
    day: int | None = None
    month: int | None = None
    year: int | None = None

    def to_string(self) -> str:
        return f"{component_to_string(self.day, 2)}/{component_to_string(self.month, 2)}/{component_to_string(self.year, 4)}"

    def is_valid(self) -> bool:
        try: 
            datetime(self.year if self.year else 2000, 
                    self.month if self.month else 1, 
                    self.day if self.day else 1)
        except ValueError: 
            return False
        return True

@dataclass(frozen=True)
class DateRange:
    start: Date = Date()
    end: Date = Date()
    
    def to_string(self) -> str:
        return f"{self.start.to_string()} - {self.end.to_string()}"

    def is_valid(self) -> bool:
        if not self.start.is_valid(): 
            return False
        if not self.end.is_valid(): 
            return False
        if self.start.year and self.end.year and self.start.year > self.end.year: 
            return False
        if self.start.year == self.end.year:
            if self.start.month and self.end.month and self.start.month > self.end.month: 
                return False
            if self.start.month == self.end.month:
                if self.start.day and self.end.day and self.start.day > self.end.day: 
                    return False
        return True

def get_earliest_date(dates: tuple[Date | DateRange, ...]) -> Date:
    #convert date ranges to two dates
    all_dates = []
    for date in dates:
        if isinstance(date, Date): 
            all_dates.append(date)
        if isinstance(date, DateRange): 
            all_dates.append(date.start)
            all_dates.append(date.end)
    if all_dates: 
        return min(all_dates, key = lambda date: (date.year if date.year != None else 5000, 
                                                  date.month if date.month != None else 13, 
                                                  date.day if date.day != None else 31))
    else: 
        return Date()

def get_latest_date(dates: tuple[Date | DateRange, ...]) -> Date:
    #convert date ranges to two dates
    all_dates = []
    for date in dates:
        if isinstance(date, Date):
            all_dates.append(date)
        if isinstance(date, DateRange): 
            all_dates.append(date.start)
            all_dates.append(date.end)
    if all_dates: 
        return max(all_dates, key = lambda date: (date.year if date.year != None else 0,
                                                  date.month if date.month != None else 0, 
                                                  date.day if date.day != None else 0))
    else: 
        return Date()
