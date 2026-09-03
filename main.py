"""Compatibility entry point for `uv run main.py`."""

from src.cli import main
from src.duplicates import check_duplicate_rows

__all__ = ["check_duplicate_rows", "main"]

if __name__ == "__main__":
    raise SystemExit(main())
