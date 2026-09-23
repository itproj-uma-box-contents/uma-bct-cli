from dataclasses import dataclass

@dataclass(frozen=True)
class EmptyDateDetector:
    name: str = "none"

    def extract_dates(self, string: str) -> tuple[str, ...]:
        return ()
