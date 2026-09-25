"""Build and safely write the catalogue CSV.

Rows are built first so duplicate and confidence flags can be added. The file
is written beside the final path and replaced only when it is complete.
"""

from __future__ import annotations

import csv
import os
import tempfile
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from .atomic import rename_new_file
from .catalogue import append_flag, map_catalogue_record
from .constants import DUPLICATE_FLAG, FALLBACK_USED_FLAG, LOW_CONFIDENCE_FLAG, OUTPUT_COLUMNS
from .path_safety import check_path
from .pipeline import ImageResult


def catalogue_records(results: Iterable[ImageResult], *, minimum_confidence: int = 80, catalogue_path: str | Path | None = None,
    trusted_crop_links: bool = False) -> list[dict[str, str | int]]:
    """Build catalogue rows and keep the physical box order."""

    if not 0 <= minimum_confidence <= 100:
        raise ValueError("Low-confidence threshold must be between 0 and 100.")

    records: list[dict[str, str | int]] = []
    for result in results:
        used_fallback = (
            len(result.fallback_path) > 1 or result.actual_detector.casefold() != result.requested_detector.casefold()
        )
        for box_result in result.records:
            records.append(
                map_catalogue_record(
                    box_result.transcription,
                    box_result.box.image_file_path,
                    catalogue_path=catalogue_path,
                    additional_flags=(FALLBACK_USED_FLAG,) if used_fallback else (),
                    trusted_crop_link=trusted_crop_links,
                )
            )

    # Two boxes can legitimately share a title. Keep both rows and mark them
    # for review after seeing the complete photo set.
    title_counts = Counter(
        str(record["Node Title"]).strip().casefold() for record in records if str(record["Node Title"]).strip()
    )
    for record in records:
        title = str(record["Node Title"]).strip()
        flags = str(record["Review Flags"])
        if title and title_counts[title.casefold()] > 1:
            flags = append_flag(flags, DUPLICATE_FLAG)
        if int(record["Transcription Confidence (%)"]) < minimum_confidence:
            flags = append_flag(flags, LOW_CONFIDENCE_FLAG)
        record["Review Flags"] = flags

    return records


def export_catalogue(results: Iterable[ImageResult], output_path: str | Path, *, minimum_confidence: int = 80,
    replace_existing: bool = True, trusted_crop_links: bool = False) -> list[dict[str, str | int]]:
    """Safely write the UTF-8 CSV and return its rows."""

    destination = Path(output_path)
    if destination.suffix.casefold() != ".csv":
        raise ValueError("Catalogue output must end in .csv.")

    records = catalogue_records(
        results,
        minimum_confidence=minimum_confidence,
        catalogue_path=destination,
        trusted_crop_links=trusted_crop_links,
    )
    check_path(destination, "Catalogue output")
    if destination.exists() and not replace_existing:
        raise FileExistsError(f"Catalogue output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    check_path(destination, "Catalogue output")
    temporary_path: Path | None = None
    # A sibling temporary file keeps readers from seeing a half-written CSV.
    # Recheck the destination after writing because another process may have
    # changed the output path while rows were being built.
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=destination.parent,
            prefix=".uma-bct-catalogue-",
            suffix=".csv",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            writer = csv.DictWriter(temporary_file, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()
            writer.writerows(records)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        check_path(destination, "Catalogue output")
        if destination.exists() and not replace_existing:
            raise FileExistsError(f"Catalogue output already exists: {destination}")
        if replace_existing:
            temporary_path.replace(destination)
        else:
            rename_new_file(temporary_path, destination, existing_message=f"Catalogue output already exists: {destination}")
            temporary_path = None
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return records
