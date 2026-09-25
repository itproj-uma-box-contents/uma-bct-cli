from dataclasses import dataclass
from datetime import datetime


class InvalidDate(Exception):
    """Provided parameters do not form a valid date."""

@dataclass(frozen=True)
class Date:
    day: str = ""
    month: str = ""
    year: str = ""

    def to_string(self) -> str:
        return f"{self.day}/{self.month}/{self.year}"

    def __post_init__(self) -> None:
        try:
            if self.day: 
                int(self.day)
        except ValueError as err: 
            raise InvalidDate(f"Provided day string is not an integer, {self.day}") from err

        try:
            if self.month: 
                int(self.month)
        except ValueError as err: 
            raise InvalidDate(f"Provided day string is not an integer, {self.month}") from err

        try:
            if self.year: 
                int(self.year)
        except ValueError as err: 
            raise InvalidDate(f"Provided day string is not an integer, {self.year}") from err

    def is_valid(self) -> bool:
        try: 
            datetime(int(self.year) if self.year else 2000, 
                    int(self.month) if self.month else 1, 
                    int(self.day) if self.day else 1)
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
        return min(all_dates, key = lambda date: (date.year if date.year else "5000", date.month if date.month else "13", date.day if date.day else "31"))
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
        return max(all_dates, key = lambda date: (date.year if date.year else "0000", date.month if date.month else "00", date.day if date.day else "00"))
    else: 
        return Date()
