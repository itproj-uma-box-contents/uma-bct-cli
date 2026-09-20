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

NODE_TITLE = re.compile(r"^(?P<prefix>.+?)\s+Unit\s*#?\s*(?P<unit>\d+)\s*$", re.IGNORECASE)


def _missing_box_row(prefix: str, unit: int) -> dict[str, str]:
    return {
        **{column: "" for column in OUTPUT_COLUMNS},
        "Node Title": f"{prefix} Unit {unit:04d}",
        "Review Notes": MISSING_BOX_FLAG,
    }


def missing_box_records(records: Iterable[dict[str, str | int]]) -> list[dict[str, str]]:
    """Return blank review rows for gaps within each identifier prefix."""

    units_by_prefix: dict[str, set[int]] = {}

    for record in records:
        match = NODE_TITLE.fullmatch(str(record["Node Title"]).strip())
        if match:
            prefix = match.group("prefix")
            units_by_prefix.setdefault(prefix, set()).add(int(match.group("unit")))

    return [
        _missing_box_row(prefix, unit)
        for prefix in sorted(units_by_prefix, key=str.casefold)
        for unit in range(min(units_by_prefix[prefix]), max(units_by_prefix[prefix]) + 1)
        if unit not in units_by_prefix[prefix]
    ]


def _sorted_records(records: Iterable[dict[str, str | int]]) -> list[dict[str, str | int]]:
    """Sort valid box titles by prefix and unit, leaving unparseable rows last."""

    indexed_records = list(enumerate(records))

    def sort_key(item: tuple[int, dict[str, str | int]]) -> tuple[int, str, int, int]:
        index, record = item
        match = NODE_TITLE.fullmatch(str(record["Node Title"]).strip())
        if match is None:
            return (1, "", 0, index)
        prefix = match.group("prefix")
        return (0, prefix.casefold(), int(match.group("unit")), index)

    return [record for _, record in sorted(indexed_records, key=sort_key)]


def _insert_missing_box_records(records: list[dict[str, str | int]]) -> list[dict[str, str | int]]:
    """Insert missing rows directly between consecutive units in each prefix group."""

    output: list[dict[str, str | int]] = []
    for index, record in enumerate(records):
        output.append(record)
        current = NODE_TITLE.fullmatch(str(record["Node Title"]).strip())
        if current is None or index + 1 == len(records):
            continue
        following = NODE_TITLE.fullmatch(str(records[index + 1]["Node Title"]).strip())
        if following is None or following.group("prefix") != current.group("prefix"):
            continue
        current_unit = int(current.group("unit"))
        following_unit = int(following.group("unit"))
        for unit in range(current_unit + 1, following_unit):
            output.append(_missing_box_row(current.group("prefix"), unit))
    return output


def catalogue_records(
    results: Iterable[ImageResult],
    *,
    minimum_confidence: int = 80,
    catalogue_path: str | Path | None = None,
    detect_missing_boxes: bool = True,
) -> list[dict[str, str | int]]:
    """Build catalogue rows sorted by box prefix and unit number."""

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

    records = _sorted_records(records)
    if detect_missing_boxes:
        records = _insert_missing_box_records(records)
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
    results: Iterable[ImageResult],
    output_path: str | Path,
    *,
    minimum_confidence: int = 80,
    replace_existing: bool = True,
    detect_missing_boxes: bool = True,
) -> list[dict[str, str | int]]:
    """Safely write the UTF-8 CSV and return its rows."""

    destination = Path(output_path)
    if destination.suffix.casefold() != ".csv":
        raise ValueError("Catalogue output must end in .csv.")

    records = catalogue_records(
        results, minimum_confidence=minimum_confidence, catalogue_path=destination, detect_missing_boxes=detect_missing_boxes
    )
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
