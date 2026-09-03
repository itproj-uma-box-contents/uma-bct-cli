"""Compatibility entry point for `uv run main.py`."""

from uma_bct.cli import main
from uma_bct.duplicates import check_duplicate_rows

__all__ = ["check_duplicate_rows", "main"]

if __name__ == "__main__":
    raise SystemExit(main())
