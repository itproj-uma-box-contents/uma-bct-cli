"""Duplicate catalogue-record flagging retained for the CSV export stage."""

from __future__ import annotations

import pandas as pd

DUPLICATE_FLAG = "Potential Duplicate"


def _append_flag(current_flags: object, flag: str) -> str:
    if pd.isna(current_flags) or not str(current_flags).strip():
        return flag
    flags = [item.strip() for item in str(current_flags).split(";") if item.strip()]
    if flag not in flags:
        flags.append(flag)
    return "; ".join(flags)


def check_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Flag every repeated non-empty Node Title without removing any rows."""

    required_columns = {"Node Title", "Flags"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Duplicate checking requires columns: {sorted(missing_columns)}"
        )

    result = df.set_index("Node Title", drop=False).copy()
    normalized_titles = result["Node Title"].astype("string").str.strip()
    duplicate_rows = normalized_titles.notna() & normalized_titles.ne("")
    duplicate_rows &= normalized_titles.duplicated(keep=False)

    result.loc[duplicate_rows, "Flags"] = result.loc[
        duplicate_rows, "Flags"
    ].map(lambda current: _append_flag(current, DUPLICATE_FLAG))
    return result
