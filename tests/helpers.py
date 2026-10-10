from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

from uma_bct.domain import BoundingBox, Box, BoxResult, BoxTranscription, InputImage
from uma_bct.pipeline import ImageResult


def read_excel(path: Path) -> tuple[list[list[str]], dict[str, str]]:
    """Read catalogue cells and link targets without another Excel dependency."""
    namespace = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with ZipFile(path) as workbook:
        strings = ElementTree.fromstring(workbook.read("xl/sharedStrings.xml"))
        shared = ["".join(item.itertext()) for item in strings]
        sheet = ElementTree.fromstring(workbook.read("xl/worksheets/sheet1.xml"))
        rows = []
        for row in sheet.findall("m:sheetData/m:row", namespace):
            values = {cell.attrib["r"].rstrip("0123456789"): cell for cell in row}
            cells = []
            for column in "ABCDEFG":
                cell = values.get(column)
                value = "" if cell is None else cell.findtext("m:v", default="", namespaces=namespace)
                cells.append(shared[int(value)] if cell is not None and cell.get("t") == "s" else value)
            rows.append(cells)
        relationships = {}
        name = "xl/worksheets/_rels/sheet1.xml.rels"
        if name in workbook.namelist():
            relationships = {item.get("Id"): item.get("Target") for item in ElementTree.fromstring(workbook.read(name))}
        links = {item.get("ref"): relationships[item.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")]
                 for item in sheet.findall("m:hyperlinks/m:hyperlink", namespace)}
    return rows, links


def make_image(path: Path | None = None) -> InputImage:
    return InputImage(path or Path("shelf.jpg"), width=200, height=300)


def make_box(*, image: InputImage | None = None, box_id: str = "shelf:001", bbox: BoundingBox | None = None,
             crop_path: Path | None = None, confidence: float = 0.95) -> Box:
    source = image or make_image()
    return Box(
        box_id=box_id,
        source_image=source,
        bounding_box=bbox or BoundingBox((10, 20), (90, 180)),
        image_file_path=crop_path or Path("crops/shelf/box_001.png"),
        position="middle-left",
        detector_confidence=confidence,
    )


def make_transcription(box: Box, *, identifiers: tuple[str, ...] = ("1960.0009",), unit: str = "4",
                       typed: tuple[str, ...] | None = None, handwritten: tuple[str, ...] = (),
                       auxiliary: tuple[str, ...] = (), confidence: int = 95, needs_review: bool = False) -> BoxTranscription:
    # Default printed lines corroborate the structured fields. Individual
    # tests can pass even an empty tuple to model missing visible evidence.
    if typed is None:
        typed = (identifiers[0], f"Unit {unit}") if identifiers else (f"Unit {unit}",)
    return BoxTranscription(
        box_id=box.box_id,
        position=box.position,
        confidence=confidence,
        record_identifiers=identifiers,
        unit_number=unit,
        typed_lines=typed,
        handwritten_lines=handwritten,
        auxiliary_annotations=auxiliary,
        needs_review=needs_review,
    )


def make_result(*records: BoxResult, requested: str = "qwen", actual: str = "qwen",
                route: tuple[str, ...] = ("qwen",), status: str = "complete", warnings: tuple[str, ...] = ()) -> ImageResult:
    image = records[0].box.source_image if records else make_image()
    return ImageResult(
        input_image=image,
        requested_detector=requested,
        actual_detector=actual,
        fallback_path=route,
        transcriber="qwen",
        status=status,
        records=records,
        warnings=warnings,
        model_metadata={"model": "qwen-test", "model_digest": "sha256:test"},
    )
