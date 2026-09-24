from dataclasses import dataclass

import datetime
  
@dataclass(frozen=True)
class Date():
    day: str = ""
    month: str = ""
    year: str = ""

    def to_string(self) -> str:
        return f"{self.day}/{self.month}/{self.year}"

@dataclass(frozen=True)
class DateRange():
    start: Date = Date()
    end: Date = Date()
    
    def to_string(self) -> str:
        return f"{self.start.to_string()} - {self.end.to_string()}"

def get_earliest_date(dates: tuple[Date | DateRange, ...]) -> Date:
    #convert date ranges to two dates
    all_dates = []
    for date in dates:
        if isinstance(date, Date): all_dates.append(date)
        if isinstance(date, DateRange): 
            all_dates.append(date.start)
            all_dates.append(date.end)
    if all_dates: return min(all_dates, key = lambda date: (date.year if date.year else "5000", date.month if date.month else "13", date.day if date.day else "31"))
    else: return Date()

def get_latest_date(dates: tuple[Date | DateRange, ...]) -> Date:
    #convert date ranges to two dates
    all_dates = []
    for date in dates:
        if isinstance(date, Date): all_dates.append(date)
        if isinstance(date, DateRange): 
            all_dates.append(date.start)
            all_dates.append(date.end)
    if all_dates: return max(all_dates, key = lambda date: (date.year if date.year else "0000", date.month if date.month else "00", date.day if date.day else "00"))
    else: return Date()