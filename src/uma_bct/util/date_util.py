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