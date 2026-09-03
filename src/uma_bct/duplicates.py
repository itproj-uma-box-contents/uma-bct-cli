"""Duplicate catalogue-record flagging retained for the CSV export stage."""

from __future__ import annotations

import pandas as pd

from .constants import DUPLICATE_FLAG
from .validation import append_flag


def check_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Flag every repeated non-empty canonical title without removing rows."""

    required_columns = {"Node Title", "Flags"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Duplicate checking requires columns: {sorted(missing_columns)}"
        )

    result = df.set_index("Node Title", drop=False).copy()
    normalized_titles = result["Node Title"].astype("string").str.strip().str.casefold()
    duplicate_rows = normalized_titles.notna() & normalized_titles.ne("")
    duplicate_rows &= normalized_titles.duplicated(keep=False)

    result.loc[duplicate_rows, "Flags"] = result.loc[duplicate_rows, "Flags"].map(
        lambda current: append_flag(current, DUPLICATE_FLAG)
    )
    return result
