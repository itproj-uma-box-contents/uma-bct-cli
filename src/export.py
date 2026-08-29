"""Create catalogue records and write complete batches to CSV or XLSX."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Iterable

import pandas as pd

from constants import OUTPUT_COLUMNS

from .catalogue import map_catalogue_record
from .duplicates import check_duplicate_rows
from .pipeline import ImageResult
from .validation import check_low_confidence_rows

SUPPORTED_OUTPUT_EXTENSIONS = {".csv", ".xlsx"}


def catalogue_dataframe(results: Iterable[ImageResult]) -> pd.DataFrame:
    """Map a batch and flag duplicates across every successfully processed image."""

    records: list[dict[str, str | int]] = []
    for result in results:
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
                map_catalogue_record(transcription, box.image_file_path)
            )

        if transcriptions_by_id:
            unexpected = ", ".join(sorted(transcriptions_by_id))
            raise ValueError(f"Transcriptions do not match boxes: {unexpected}")

    dataframe = pd.DataFrame.from_records(records, columns=OUTPUT_COLUMNS)
    duplicate_checked = check_duplicate_rows(dataframe)
    return check_low_confidence_rows(duplicate_checked).reset_index(drop=True)


def export_catalogue(
    results: Iterable[ImageResult], output_path: str | Path
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

    dataframe = catalogue_dataframe(results)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.stem}-",
            suffix=extension,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

        if extension == ".csv":
            dataframe.to_csv(temporary_path, index=False)
        else:
            dataframe.to_excel(temporary_path, index=False, engine="openpyxl")
        temporary_path.replace(destination)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise

    return dataframe
