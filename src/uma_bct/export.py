"""Sort catalogue rows, flag duplicates, and stage a catalogue for publication.

Missing-box rows are review suggestions between observed boxes with the same
Unit Type. The catalogue is written beside its destination before replacement.
"""

from __future__ import annotations

import csv
import os
import re
import tempfile
from collections import Counter
from collections.abc import Iterable, Sequence
from datetime import datetime
from itertools import pairwise
from pathlib import Path

import xlsxwriter

from .atomic import rename_new_file
from .catalogue import CatalogueRow, append_flag, map_catalogue_record, verification_flags
from .constants import (
    DEFAULT_MIN_TRANSCRIPTION_CONFIDENCE,
    DUPLICATE_FLAG,
    FALLBACK_USED_FLAG,
    LOW_CONFIDENCE_FLAG,
    MISSING_BOX_FLAG,
    OUTPUT_COLUMNS,
)
from .path_safety import check_path
from .pipeline import ImageResult

UNIT_NUMBER = re.compile(r"^\d{1,4}$")
MAX_GAP_SIZE = 100


def _missing_box_row(prefix: str, unit: int) -> CatalogueRow:
    """Build a blank catalogue row for one predicted missing unit."""

    return CatalogueRow({
        **{column: "" for column in OUTPUT_COLUMNS},
        "Unit Type": prefix,
        "Unit Number": f"{unit:04d}",
        "Review Flags": MISSING_BOX_FLAG,
    })


def missing_box_records(records: Iterable[dict[str, str | int]]) -> list[dict[str, str]]:
    """Return bounded internal gap suggestions for each case-insensitive Unit Type."""

    units_by_type: dict[str, tuple[str, set[int]]] = {}

    for record in _sorted_records(records):
        unit_type = str(record["Unit Type"]).strip()
        unit = str(record["Unit Number"]).strip()
        if unit_type and UNIT_NUMBER.fullmatch(unit):
            key = unit_type.casefold()
            if key not in units_by_type:
                units_by_type[key] = (unit_type, set())
            units_by_type[key][1].add(int(unit))

    missing: list[dict[str, str]] = []
    for key in sorted(units_by_type):
        unit_type, observed = units_by_type[key]
        ordered = sorted(observed)
        for lower, upper in pairwise(ordered):
            gap_size = upper - lower - 1
            if 0 < gap_size <= MAX_GAP_SIZE:
                missing.extend(_missing_box_row(unit_type, unit) for unit in range(lower + 1, upper))
    return missing


def _sorted_records(records: Iterable[dict[str, str | int]]) -> list[dict[str, str | int]]:
    """Sort rows with a Unit Type and number, leaving incomplete rows stable at the end."""

    indexed_records = list(enumerate(records))

    def sort_key(item: tuple[int, dict[str, str | int]]) -> tuple[int, str, int, int]:
        # Keep incomplete rows stable after sortable unit rows.
        index, record = item
        prefix = str(record["Unit Type"]).strip()
        unit = str(record["Unit Number"]).strip()
        if not prefix or UNIT_NUMBER.fullmatch(unit) is None:
            return (1, "", 0, index)
        return (0, prefix.casefold(), int(unit), index)

    return [record for _, record in sorted(indexed_records, key=sort_key)]


def _insert_missing_box_records(records: list[dict[str, str | int]]) -> list[dict[str, str | int]]:
    """Insert missing rows between consecutive units in each prefix group, unless the gap exceeds MAX_GAP_SIZE."""

    output: list[dict[str, str | int]] = []
    for index, record in enumerate(records):
        output.append(record)
        current_prefix = str(record["Unit Type"]).strip()
        current_unit = str(record["Unit Number"]).strip()
        if not current_prefix or UNIT_NUMBER.fullmatch(current_unit) is None or index + 1 == len(records):
            continue
        following_prefix = str(records[index + 1]["Unit Type"]).strip()
        following_unit = str(records[index + 1]["Unit Number"]).strip()
        if (
            not following_prefix
            or UNIT_NUMBER.fullmatch(following_unit) is None
            or following_prefix.casefold() != current_prefix.casefold()
        ):
            continue

        gap_size = int(following_unit) - int(current_unit) - 1
        if gap_size <= 0 or gap_size > MAX_GAP_SIZE:
            continue

        for unit in range(int(current_unit) + 1, int(following_unit)):
            output.append(_missing_box_row(current_prefix, unit))
    return output


def catalogue_records(
    results: Iterable[ImageResult],
    *,
    minimum_confidence: int = DEFAULT_MIN_TRANSCRIPTION_CONFIDENCE,
    catalogue_path: str | Path | None = None,
    detect_missing_boxes: bool = True,
    trusted_crop_links: bool = False,
    spreadsheet_safe: bool = True,
    legacy_identity_mapping: bool = False,
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
                    additional_flags=((FALLBACK_USED_FLAG,) if used_fallback else ()) + verification_flags(
                        box_result.verification
                    ),
                    trusted_crop_link=trusted_crop_links,
                    spreadsheet_safe=spreadsheet_safe,
                    legacy_identity_mapping=legacy_identity_mapping,
                )
            )

    # Sorting by the complete printed Unit Type makes internal numeric gaps
    # visible without joining boxes whose headings or identifiers differ.
    records = _sorted_records(records)
    if detect_missing_boxes:
        records = _insert_missing_box_records(records)
    # Identical Unit Type and number can still describe two physical boxes.
    # Keep both rows and mark the matching identities for human review.
    title_counts = Counter(
        (str(record["Unit Type"]).strip().casefold(), str(record["Unit Number"]).strip())
        for record in records
        if str(record["Unit Type"]).strip() and str(record["Unit Number"]).strip()
    )
    for record in records:
        identity = (str(record["Unit Type"]).strip().casefold(), str(record["Unit Number"]).strip())
        flags = str(record["Review Flags"])
        if identity[0] and identity[1] and title_counts[identity] > 1:
            flags = append_flag(flags, DUPLICATE_FLAG)
        confidence = record["Transcription Confidence (%)"]
        if confidence != "" and int(confidence) < minimum_confidence:
            flags = append_flag(flags, LOW_CONFIDENCE_FLAG)
        record["Review Flags"] = flags

    return records


def _write_excel(path: Path, records: list[dict[str, str | int]], columns: Sequence[str]) -> None:
    """Write literal cell values and relative links to the actual crop files."""
    with xlsxwriter.Workbook(path, {
        "in_memory": True, "strings_to_formulas": False, "strings_to_urls": False,
    }) as workbook:
        # Resume verifies file hashes, so workbook timestamps must stay fixed.
        workbook.set_properties({"created": datetime(1980, 1, 1)})
        sheet = workbook.add_worksheet("Catalogue")
        text = workbook.add_format({"text_wrap": True, "valign": "top"})
        heading = workbook.add_format({"bold": True, "text_wrap": True, "valign": "top"})
        link = workbook.add_format({"font_color": "blue", "underline": True, "text_wrap": True, "valign": "top"})
        widths = dict(zip(OUTPUT_COLUMNS, (42, 14, 52, 32, 18, 18, 20, 36, 40), strict=True))
        widths["Date Range"] = 16  # Historical staged catalogues retain their original layout.
        for column, name in enumerate(columns):
            sheet.set_column(column, column, widths[name], text)
        sheet.write_row(0, 0, columns, heading)
        sheet.freeze_panes(1, 0)
        for row_number, record in enumerate(records, start=1):
            for column_number, column in enumerate(columns):
                value = record[column]
                if column == "Cropped Box Image Link" and value:
                    reference = str(value)
                    # Preserve Unicode; escape literal % before URL anchors.
                    target = "external:./" + reference.replace("%", "%25").replace("#", "%23")
                    status = sheet.write_url(row_number, column_number, target, link, Path(reference).name)
                else:
                    status = sheet.write(row_number, column_number, value)
                if status != 0:
                    raise ValueError("Catalogue exceeds Excel's cell or hyperlink limits.")


def export_catalogue(
    results: Iterable[ImageResult],
    output_path: str | Path,
    *,
    minimum_confidence: int = DEFAULT_MIN_TRANSCRIPTION_CONFIDENCE,
    replace_existing: bool = True,
    detect_missing_boxes: bool = True,
    trusted_crop_links: bool = False,
    output_columns: Sequence[str] = OUTPUT_COLUMNS,
    legacy_identity_mapping: bool = False,
) -> list[dict[str, str | int]]:
    """Safely write an Excel workbook or UTF-8 CSV and return its rows."""

    destination = Path(output_path)
    if destination.suffix.casefold() not in {".xlsx", ".csv"}:
        raise ValueError("Catalogue output must end in .xlsx or .csv.")
    excel = destination.suffix.casefold() == ".xlsx"

    records = catalogue_records(
        results, minimum_confidence=minimum_confidence, catalogue_path=destination,
        detect_missing_boxes=detect_missing_boxes, trusted_crop_links=trusted_crop_links and not excel,
        spreadsheet_safe=not excel,
        legacy_identity_mapping=legacy_identity_mapping,
    )
    if "Date Range" in output_columns:
        for record in records:
            record["Date Range"] = ""
    check_path(destination, "Catalogue output")
    if destination.exists() and not replace_existing:
        raise FileExistsError(f"Catalogue output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    check_path(destination, "Catalogue output")
    temporary_path: Path | None = None
    # A sibling temporary file keeps readers from seeing a half-written export.
    # Recheck the destination after writing because another process may have
    # changed the output path while rows were being built.
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=destination.parent,
            prefix=".uma-bct-catalogue-",
            suffix=destination.suffix,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            if not excel:
                writer = csv.DictWriter(temporary_file, fieldnames=output_columns, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(records)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
        if excel:
            # Close the temporary handle first; Windows prevents reopening it.
            _write_excel(temporary_path, records, output_columns)
            with temporary_path.open("r+b") as workbook_file:
                os.fsync(workbook_file.fileno())
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
