"""A portable, read-only catalogue view with local crop links and no web services."""
from __future__ import annotations

import html
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from urllib.parse import quote

from .atomic import rename_new_file
from .catalogue import image_reference
from .constants import OUTPUT_COLUMNS
from .export import catalogue_records
from .path_safety import check_path
from .pipeline import ImageResult

STYLE = """
:root { color-scheme: light; font: 15px/1.5 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
  color: #233746; background: #f5f8fa; }
body { margin: 0; padding: 2rem clamp(1rem, 3vw, 3rem); }
header { max-width: 76rem; margin: 0 auto 1.5rem; }
h1 { font: 600 2rem/1.2 Georgia, serif; margin: 0 0 .65rem; color: #173e58; }
p { max-width: 75ch; margin: .5rem 0; }
a { color: #135978; text-underline-offset: .2em; }
a:focus-visible, summary:focus-visible { outline: 3px solid #b86a09; outline-offset: 3px; }
.catalogue { overflow-x: auto; background: white; border: 1px solid #c7d5df; border-radius: 6px; }
table { border-collapse: collapse; table-layout: fixed; width: 100%; min-width: 850px; }
caption { text-align: left; font-weight: 600; padding: 1rem; }
th, td { text-align: left; vertical-align: top; padding: .8rem; overflow-wrap: anywhere; }
th { background: #e5eef4; border-bottom: 2px solid #8eaabc; }
td { border-bottom: 1px solid #dce5eb; }
tr:last-child td { border-bottom: 0; }
tr:nth-child(even) { background: #f5f8fa; }
col:nth-child(1) { width: 24%; } col:nth-child(2) { width: 7%; }
col:nth-child(3) { width: 26%; } col:nth-child(4) { width: 8%; }
col:nth-child(5) { width: 9%; } col:nth-child(6) { width: 17%; } col:nth-child(7) { width: 9%; }
summary { cursor: pointer; color: #135978; margin-top: .6rem; }
dd { margin: 0 0 .5rem; } dt { font-weight: 600; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; font: inherit; margin: .5rem 0; }
.readings { margin: 1rem 0; border-left: 3px solid #8eaabc; padding-left: .8rem; }
@media print { body { padding: 0; } .catalogue { overflow: visible; } table { min-width: 0; } }
"""


def escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def comparison(audit: dict[str, object] | None) -> str:
    if not audit:
        return ""
    if not audit.get("disputed") and not audit.get("error"):
        return ""
    sections = []
    if audit.get("error"):
        sections.append(f'<p>Verification failed: {escape(audit["error"])}</p>')
    labels = {"position": "Box position", "typed_lines": "Printed text", "unit_number": "Unit number",
              "record_identifiers": "Identifiers", "handwritten_lines": "Handwriting",
              "auxiliary_annotations": "Other annotations"}
    for key, label in (("original", "First reading"), ("second", "Second reading"), ("third", "Discrepancy check")):
        if key not in audit:
            continue
        values = []
        for record in audit[key]:
            for field, title in labels.items():
                value = record.get(field, "")
                if isinstance(value, (list, tuple)):
                    value = "; ".join(value)
                values.append(f'<dt>{title}</dt><dd>{escape(value) or "Blank"}</dd>')
        sections.append(f'<div class="readings"><strong>{label}</strong><dl>{"".join(values)}</dl></div>')
    return '<details><summary>Compare readings</summary>' + "".join(sections) + '</details>'


def render_review(results: Iterable[ImageResult], catalogue_path: Path, *, minimum_confidence: int = 80,
                  detect_missing_boxes: bool = True) -> str:
    results = list(results)
    rows = catalogue_records(results, catalogue_path=catalogue_path, minimum_confidence=minimum_confidence,
                             detect_missing_boxes=detect_missing_boxes, spreadsheet_safe=False)
    audits = {image_reference(record.box.image_file_path, catalogue_path): record.verification
              for result in results for record in result.records}
    rendered = []
    for row in rows:
        cells = []
        crop = str(row["Cropped Box Image Link"])
        for column in OUTPUT_COLUMNS:
            content = escape(row[column])
            if column == "Cropped Box Image Link" and crop:
                # Prefix ./ so a colon in a filename cannot become a URL scheme.
                content = f'<a href="{escape(quote("./" + crop, safe="/"))}">View crop</a>'
            if column == "Review Flags":
                content += comparison(audits.get(crop))
            cells.append(f"<td>{content}</td>")
        rendered.append("<tr>" + "".join(cells) + "</tr>")
    headings = "".join(f'<th scope="col">{escape(column)}</th>' for column in OUTPUT_COLUMNS)
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            '<meta name="referrer" content="no-referrer">'
            '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; style-src \'unsafe-inline\'; '
            'base-uri \'none\'; form-action \'none\'">'
            f'<title>Archive box catalogue</title><style>{STYLE}</style></head><body><header>'
            '<h1>Archive box catalogue</h1>'
            '<p>Compare each transcription with its cropped photograph. Review flagged rows before importing the catalogue.</p>'
            f'<p><a href="{escape(quote("./" + catalogue_path.name, safe="/"))}">Open catalogue CSV</a> '
            'to make corrections in your spreadsheet application.</p>'
            '<p>This page shows the generated readings. Later CSV edits are preserved but do not update this page. '
            'Keep this file and the crop folder together when moving results.</p></header>'
            f'<main class="catalogue"><table><caption>{len(rows)} catalogue rows</caption>'
            f'<colgroup>{"<col>" * len(OUTPUT_COLUMNS)}</colgroup><thead><tr>{headings}</tr></thead>'
            f'<tbody>{"".join(rendered)}</tbody></table></main></body></html>\n')


def export_review(results: Iterable[ImageResult], destination: Path, *, catalogue_path: Path,
                  minimum_confidence: int = 80, detect_missing_boxes: bool = True) -> None:
    """Stage one complete HTML file without clobbering an existing destination."""
    content = render_review(results, catalogue_path, minimum_confidence=minimum_confidence,
                            detect_missing_boxes=detect_missing_boxes)
    check_path(destination, "Review output")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=destination.parent,
                                         prefix=".uma-bct-review-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        check_path(destination, "Review output")
        rename_new_file(temporary, destination, existing_message=f"Review output already exists: {destination}")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
