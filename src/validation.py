"""Batch-level review-flag validation shared by catalogue exporters."""

from __future__ import annotations

import pandas as pd

LOW_CONFIDENCE_FLAG = "Low confidence"
LOW_CONFIDENCE_THRESHOLD = 50


def append_flag(current_flags: object, flag: str) -> str:
    """Append one flag once, preserving deterministic existing flag order."""

    if pd.isna(current_flags) or not str(current_flags).strip():
        return flag
    flags = [item.strip() for item in str(current_flags).split(";") if item.strip()]
    if flag not in flags:
        flags.append(flag)
    return "; ".join(flags)


def check_low_confidence_rows(
    dataframe: pd.DataFrame,
    threshold: int = LOW_CONFIDENCE_THRESHOLD,
) -> pd.DataFrame:
    """Flag records whose transcription confidence is strictly below ``threshold``."""

    if not 0 <= threshold <= 100:
        raise ValueError("Low-confidence threshold must be between 0 and 100.")
    required_columns = {"Confidence Score", "Flags"}
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise ValueError(
            f"Low-confidence checking requires columns: {sorted(missing_columns)}"
        )

    result = dataframe.copy()
    scores = pd.to_numeric(result["Confidence Score"], errors="coerce")
    low_confidence_rows = scores.notna() & scores.lt(threshold)
    result.loc[low_confidence_rows, "Flags"] = result.loc[
        low_confidence_rows, "Flags"
    ].map(lambda current: append_flag(current, LOW_CONFIDENCE_FLAG))
    return result
