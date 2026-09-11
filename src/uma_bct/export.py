"""Build and safely write the catalogue CSV.

Rows are built first so duplicate and confidence flags can be added. The file
is written beside the final path and replaced only when it is complete.
"""

from __future__ import annotations

import csv
import os
import re
import tempfile
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from .catalogue import append_flag, map_catalogue_record
from .constants import DUPLICATE_FLAG, FALLBACK_USED_FLAG, LOW_CONFIDENCE_FLAG, MISSING_BOX_FLAG, OUTPUT_COLUMNS
from .path_safety import check_path
from .pipeline import ImageResult

NODE_TITLE = re.compile(r"^(.+) Unit (\d{4})$")


def missing_box_records(records: Iterable[dict[str, str | int]]) -> list[dict[str, str]]:
    """Return blank review rows for gaps between recorded box titles."""

    units_by_identifier: dict[str, set[int]] = {}
    for record in records:
        match = NODE_TITLE.fullmatch(str(record["Node Title"]).strip())
        if match:
            units_by_identifier.setdefault(match.group(1), set()).add(int(match.group(2)))

    missing: list[dict[str, str]] = []
    for identifier, units in units_by_identifier.items():
        for unit in range(min(units), max(units) + 1):
            if unit not in units:
                row = {column: "" for column in OUTPUT_COLUMNS}
                row["Node Title"] = f"{identifier} Unit {unit:04d}"
                row["Review Notes"] = MISSING_BOX_FLAG
                missing.append(row)
    return missing


def catalogue_records(
    results: Iterable[ImageResult], *, minimum_confidence: int = 80, catalogue_path: str | Path | None = None
) -> list[dict[str, str | int]]:
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
                )
            )

    # These checks can only be made after every source row has been built.
    records.extend(missing_box_records(records))
    title_counts = Counter(
        str(record["Node Title"]).strip().casefold() for record in records if str(record["Node Title"]).strip()
    )
    for record in records:
        title = str(record["Node Title"]).strip()
        flags = str(record["Review Notes"])
        if title and title_counts[title.casefold()] > 1:
            flags = append_flag(flags, DUPLICATE_FLAG)
        confidence = record["Transcription Confidence (0-100)"]
        if confidence != "" and int(confidence) < minimum_confidence:
            flags = append_flag(flags, LOW_CONFIDENCE_FLAG)
        record["Review Notes"] = flags

    return records


def export_catalogue(
    results: Iterable[ImageResult], output_path: str | Path, *, minimum_confidence: int = 80, replace_existing: bool = True
) -> list[dict[str, str | int]]:
    """Safely write the UTF-8 CSV and return its rows."""

    destination = Path(output_path)
    if destination.suffix.casefold() != ".csv":
        raise ValueError("Catalogue output must end in .csv.")

    records = catalogue_records(results, minimum_confidence=minimum_confidence, catalogue_path=destination)
    check_path(destination, "Catalogue output")
    if destination.exists() and not replace_existing:
        raise FileExistsError(f"Catalogue output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    check_path(destination, "Catalogue output")
    temporary_path: Path | None = None
    # Write beside the final file, then swap it in only when complete.
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
        temporary_path.replace(destination)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return records
