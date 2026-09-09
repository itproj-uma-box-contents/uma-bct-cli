"""Build and atomically write the client CSV catalogue.

``catalogue_records`` runs in two passes because some flags depend on the
full batch rather than one box: duplicate-title detection compares every
record's title against every other record's, and the low-confidence flag
needs the per-record confidence already computed by
``catalogue.map_catalogue_record`` in the first pass. ``export_catalogue``
writes to a temporary file in the same directory as the destination,
``fsync``s it, and only then ``replace()``s the destination, so a crash or
interruption during export can never leave a half-written or corrupt
catalogue CSV behind.
"""

from __future__ import annotations

import csv
import os
import tempfile
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from .catalogue import append_flag, map_catalogue_record
from .constants import (
    DUPLICATE_FLAG,
    FALLBACK_USED_FLAG,
    LOW_CONFIDENCE_FLAG,
    OUTPUT_COLUMNS,
)
from .pipeline import ImageResult


def catalogue_records(
    results: Iterable[ImageResult],
    *,
    minimum_confidence: int = 80,
    catalogue_path: str | Path | None = None,
) -> list[dict[str, str | int]]:
    """Map results and add batch flags without changing physical-box order."""

    if not 0 <= minimum_confidence <= 100:
        raise ValueError("Low-confidence threshold must be between 0 and 100.")

    records: list[dict[str, str | int]] = []
    for result in results:
        used_fallback = (
            len(result.fallback_path) > 1
            or result.actual_detector.casefold() != result.requested_detector.casefold()
        )
        for box_result in result.records:
            records.append(
                map_catalogue_record(
                    box_result.transcription,
                    box_result.box.image_file_path,
                    catalogue_path=catalogue_path,
                    additional_flags=(FALLBACK_USED_FLAG,) if used_fallback else (),
                )
            )

    title_counts = Counter(
        str(record["Node Title"]).strip().casefold()
        for record in records
        if str(record["Node Title"]).strip()
    )
    for record in records:
        title = str(record["Node Title"]).strip()
        flags = str(record["Review Notes"])
        if title and title_counts[title.casefold()] > 1:
            flags = append_flag(flags, DUPLICATE_FLAG)
        if int(record["Transcription Confidence (0-100)"]) < minimum_confidence:
            flags = append_flag(flags, LOW_CONFIDENCE_FLAG)
        record["Review Notes"] = flags

    return records


def export_catalogue(
    results: Iterable[ImageResult],
    output_path: str | Path,
    *,
    minimum_confidence: int = 80,
) -> list[dict[str, str | int]]:
    """Atomically write a UTF-8 CSV and return its catalogue records."""

    destination = Path(output_path)
    if destination.suffix.casefold() != ".csv":
        raise ValueError("Catalogue output must end in .csv.")

    records = catalogue_records(
        results,
        minimum_confidence=minimum_confidence,
        catalogue_path=destination,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
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
        temporary_path.replace(destination)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return records
