"""Create catalogue records and write complete batches to CSV or XLSX."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from .catalogue import map_catalogue_record
from .constants import FALLBACK_USED_FLAG, OUTPUT_COLUMNS, SUPPORTED_OUTPUT_EXTENSIONS
from .duplicates import check_duplicate_rows
from .pipeline import ImageResult
from .validation import check_low_confidence_rows

EXCEL_CELL_CHARACTER_LIMIT = 32_767


def _validate_xlsx_cell_lengths(dataframe: pd.DataFrame) -> None:
    """Reject values that Excel would silently truncate."""

    for column in dataframe.columns:
        for value in dataframe[column]:
            if isinstance(value, str) and len(value) > EXCEL_CELL_CHARACTER_LIMIT:
                raise ValueError(
                    "XLSX cells cannot exceed 32,767 characters; export this "
                    "catalogue as CSV to preserve the complete text."
                )


def catalogue_dataframe(
    results: Iterable[ImageResult],
    *,
    minimum_confidence: int = 80,
    catalogue_path: str | Path | None = None,
) -> pd.DataFrame:
    """Map a batch and flag duplicates across every successfully processed image."""

    records: list[dict[str, str | int]] = []
    for result in results:
        requested_detector = result.requested_detector or result.detector
        actual_detector = result.actual_detector or result.detector
        fallback_path = result.fallback_path or (requested_detector,)
        used_fallback = (
            len(fallback_path) > 1
            or actual_detector.casefold() != requested_detector.casefold()
        )
        transcriptions_by_id = {
            transcription.box_id: transcription
            for transcription in result.transcriptions
        }
        if len(transcriptions_by_id) != len(result.transcriptions):
            raise ValueError(
                f"Duplicate transcription box IDs for {result.input_image.file_path}"
            )

        for box in result.boxes:
            try:
                transcription = transcriptions_by_id.pop(box.box_id)
            except KeyError as error:
                raise ValueError(
                    f"No transcription found for box {box.box_id}"
                ) from error
            records.append(
                map_catalogue_record(
                    transcription,
                    box.image_file_path,
                    catalogue_path=catalogue_path,
                    additional_flags=(FALLBACK_USED_FLAG,) if used_fallback else (),
                )
            )

        if transcriptions_by_id:
            unexpected = ", ".join(sorted(transcriptions_by_id))
            raise ValueError(f"Transcriptions do not match boxes: {unexpected}")

    dataframe = pd.DataFrame.from_records(records, columns=OUTPUT_COLUMNS)
    duplicate_checked = check_duplicate_rows(dataframe)
    checked = check_low_confidence_rows(
        duplicate_checked,
        threshold=minimum_confidence,
    ).reset_index(drop=True)
    if checked.empty:
        return checked

    checked["_source_order"] = range(len(checked))
    normalized_titles = checked["Node Title"].astype("string").str.strip()
    checked["_incomplete"] = normalized_titles.isna() | normalized_titles.eq("")
    return (
        checked.sort_values(
            ["_incomplete", "Node Title", "_source_order"],
            kind="stable",
            na_position="last",
        )
        .drop(columns=["_source_order", "_incomplete"])
        .reset_index(drop=True)
    )


def export_catalogue(
    results: Iterable[ImageResult],
    output_path: str | Path,
    *,
    minimum_confidence: int = 80,
) -> pd.DataFrame:
    """Atomically write a mapped batch as CSV or XLSX and return its DataFrame."""

    destination = Path(output_path)
    extension = destination.suffix.casefold()
    if extension not in SUPPORTED_OUTPUT_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_OUTPUT_EXTENSIONS))
        raise ValueError(
            f"Unsupported output extension {destination.suffix or '<none>'}; "
            f"expected one of: {supported}"
        )

    dataframe = catalogue_dataframe(
        results,
        minimum_confidence=minimum_confidence,
        catalogue_path=destination,
    )
    if extension == ".xlsx":
        _validate_xlsx_cell_lengths(dataframe)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        if extension == ".csv":
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                dir=destination.parent,
                prefix=".uma-bct-catalogue-",
                suffix=extension,
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                dataframe.to_csv(temporary_file, index=False)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
        else:
            with tempfile.NamedTemporaryFile(
                mode="w+b",
                dir=destination.parent,
                prefix=".uma-bct-catalogue-",
                suffix=extension,
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                dataframe.to_excel(temporary_file, index=False, engine="openpyxl")
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
        temporary_path.replace(destination)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise

    return dataframe
